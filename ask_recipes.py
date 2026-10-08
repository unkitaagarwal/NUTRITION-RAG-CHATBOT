"""POST /ask-recipes — turn a request ("High protein meals") into 1-5 complete recipes.

Spec: "Ask Recipes API" (Oct 7, 2026). Powers the "Or ask RecipeVault" card and the
"Ask RecipeVault" results chat.

    1. Authenticate   Firebase ID token -> uid (same as /customize-recipe)   (401)
    2. Validate       headers, sizes, shapes                                 (400)
    3. Check limit    meal_plan_chef_users/{uid}/usage/ask_recipes_{date}    (403/429)
    4. Build messages history (merged, leading assistant dropped) + final user turn
    5. Call Claude    structured JSON: status/message/recipes/followUps      (502)
    6. Nutrition      each recipe through the /extract-recipe nutrition step (parallel)
    7. Ids + images   r_ ids, ask_recipes_cache/{id}, images generated in background
    8. Count + log    only when status == "results"

Steps 4-6 must finish within ASK_RECIPES_TIMEOUT_SECONDS (45 s) or we return 504.

Also:
    GET /ask-recipes/image/<recipeId>   poll a pending image
    GET /ask-recipes/suggestions        starter "Try saying..." cards

Auth, error shape, usage counter, premium check and the Claude client are shared with
customize_recipe.py so both endpoints behave the same way.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from flask import Blueprint, jsonify, request

import customize_recipe as cr
from customize_recipe import ApiError, _env_int, _error_body, _is_str

# ── Config ────────────────────────────────────────────────────────────────────

TIMEOUT_SECONDS = float(os.getenv("ASK_RECIPES_TIMEOUT_SECONDS", "45"))
MAX_BODY_BYTES = _env_int("ASK_RECIPES_MAX_BODY_BYTES", 64 * 1024)
MAX_PROMPT_CHARS = 500
MAX_HISTORY = 10
MAX_HISTORY_TEXT_CHARS = 4000
MAX_EXCLUDE = 50
MAX_ASSISTANT_MESSAGE_CHARS = 400  # "at most 2 sentences"

FREE_DAILY_LIMIT = _env_int("ASK_RECIPES_FREE_DAILY_LIMIT", 3)        # 0 => premium-only (403)
PREMIUM_DAILY_LIMIT = _env_int("ASK_RECIPES_PREMIUM_DAILY_LIMIT", 30)
USAGE_PREFIX = "ask_recipes"

# Haiku 4.5 by default for cost/latency (3 full recipes are ~2-3k output tokens). Set
# ASK_RECIPES_CLAUDE_MODEL=claude-sonnet-5-5 for the spec's original model.
CLAUDE_MODEL = os.getenv("ASK_RECIPES_CLAUDE_MODEL", "claude-haiku-4-5-20251001")
CLAUDE_MAX_TOKENS = _env_int("ASK_RECIPES_CLAUDE_MAX_TOKENS", 8000)          # hard cap
TOKENS_PER_RECIPE = _env_int("ASK_RECIPES_TOKENS_PER_RECIPE", 1400)          # budget scales with count

# Whole-answer cache for repeatable first turns (suggestion cards / fresh chats with no history
# or exclusions): same prompt + count + preferences + model -> reuse the recipes, no Claude call.
RESULTS_CACHE_COLLECTION = os.getenv("ASK_RECIPES_RESULTS_CACHE_COLLECTION", "ask_recipes_results_cache")
RESULTS_CACHE_HOURS = _env_int("ASK_RECIPES_RESULTS_CACHE_HOURS", 24)        # 0 disables
# Image library: one image per dish name, reused across users and requests.
IMAGE_LIBRARY_COLLECTION = os.getenv("ASK_RECIPES_IMAGE_LIBRARY_COLLECTION", "ask_recipes_image_library")
IMAGE_GIVEUP_SECONDS = _env_int("ASK_RECIPES_IMAGE_GIVEUP_SECONDS", 180)

CACHE_COLLECTION = os.getenv("ASK_RECIPES_CACHE_COLLECTION", "ask_recipes_cache")
CACHE_TTL_DAYS = _env_int("ASK_RECIPES_CACHE_TTL_DAYS", 30)
IMAGES_ENABLED = (os.getenv("ASK_RECIPES_IMAGES") or "1").strip().lower() in ("1", "true", "yes")
SAVED_RECIPES_SUBCOLLECTION = os.getenv("ASK_RECIPES_SAVED_SUBCOLLECTION", "recipes")
SAVED_IMAGE_FIELD = os.getenv("ASK_RECIPES_SAVED_IMAGE_FIELD", "imageUrl")
SUGGESTIONS_CONFIG_DOC = os.getenv("ASK_RECIPES_SUGGESTIONS_DOC", "app_configs/ask_recipes")

ALLOWED_MODES = {"suggestion", "chat", "follow_up"}
import re as _re
_SUGGESTION_ID_RE = _re.compile(r"^[A-Za-z0-9_-]{1,50}$")
# The built-in starter cards (used for defaults only; any well-formed id is accepted).
ALLOWED_SUGGESTIONS = {"5_ingredient", "breakfast", "asian", "meal_prep", "global", "high_protein"}
ALLOWED_CLIENTS = {"ios-app", "android-app"}
ALLOWED_STATUSES = {"results", "clarify", "declined"}
DIFFICULTIES = ["Easy", "Medium", "Hard"]
MEAL_TYPES = ["Breakfast", "Lunch", "Dinner", "Snack", "Dessert"]

DEFAULT_SUGGESTIONS = [
    {"id": "high_protein", "label": "High protein meals", "prompt": "High protein meals", "imageUrl": None},
    {"id": "5_ingredient", "label": "5-ingredient dinners", "prompt": "5-ingredient dinners", "imageUrl": None},
    {"id": "breakfast", "label": "Quick breakfasts", "prompt": "Quick breakfasts", "imageUrl": None},
    {"id": "asian", "label": "Asian-inspired dishes", "prompt": "Asian-inspired dishes", "imageUrl": None},
    {"id": "meal_prep", "label": "Meal prep for the week", "prompt": "Meal prep for the week", "imageUrl": None},
    {"id": "global", "label": "Dishes from around the world", "prompt": "Dishes from around the world", "imageUrl": None},
]

SYSTEM_PROMPT = (
    "You suggest home-cooking recipes for the RecipeVault app. Return exactly count distinct, "
    "realistic recipes that match the request and the user's preferences. Never include an "
    "ingredient from allergies. Use realistic amounts for the requested servings and clear, "
    "numbered steps. Write assistantMessage as 1-2 friendly sentences introducing the results. "
    "Suggest 2-4 short follow-ups, the first being 'Show more like these'. If the request is too "
    "vague, set status to clarify and ask one short question. If it isn't about food or cooking, "
    "set status to declined."
    # Output-format guidance (not part of the product prompt above):
    "\n\nPrefer returning results. Use status clarify ONLY when you cannot reasonably pick any "
    "recipes (e.g. 'food', 'something nice', 'help'). A request that names an audience, meal, "
    "ingredient, cuisine, diet, time or goal (e.g. 'toddler meals', 'paneer', 'quick lunch', "
    "'high protein') is specific enough: make sensible assumptions (e.g. a mix of meal types), "
    "return recipes, and offer refine follow-ups that narrow it down instead of asking."
    "\n\nRespond only with JSON matching the required schema. Apply the user's preferences "
    "unless the request explicitly overrides them; allergies can never be overridden. Do not "
    "repeat any recipe listed under 'Do not repeat'. For each ingredient give name, amount "
    "(string, may be empty), unit (may be empty) and note (preparation, may be empty). "
    "Steps are in order; give durationMinutes only when a step has a clear duration. tags are "
    "up to 3 short display labels (e.g. 'High protein', 'Quick'); dietaryTags are lowercase "
    "machine values such as vegetarian, vegan, gluten_free, dairy_free. nutrition is your "
    "per-serving estimate (the server recomputes it). followUps: type 'more' for 'Show more "
    "like these', 'refine' for anything that narrows or changes the results. When status is "
    "not results, return recipes as [] and followUps as []."
)

_INGREDIENT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"}, "amount": {"type": "string"},
        "unit": {"type": "string"}, "note": {"type": "string"},
    },
    "required": ["name", "amount", "unit", "note"],
    "additionalProperties": False,
}
_STEP_SCHEMA = {
    "type": "object",
    "properties": {"instruction": {"type": "string"}, "durationMinutes": {"type": "integer"}},
    "required": ["instruction"],
    "additionalProperties": False,
}
_RECIPE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
        "prepMinutes": {"type": "integer"},
        "cookMinutes": {"type": "integer"},
        "servings": {"type": "integer"},
        "difficulty": {"type": "string", "enum": DIFFICULTIES},
        "cuisine": {"type": "string"},
        "mealType": {"type": "string", "enum": MEAL_TYPES},
        "tags": {"type": "array", "items": {"type": "string"}},
        "dietaryTags": {"type": "array", "items": {"type": "string"}},
        "ingredients": {"type": "array", "items": _INGREDIENT_SCHEMA},
        "steps": {"type": "array", "items": _STEP_SCHEMA},
        "nutrition": {
            "type": "object",
            "properties": {
                "calories": {"type": "number"}, "protein": {"type": "number"},
                "carbs": {"type": "number"}, "fat": {"type": "number"},
            },
            "required": ["calories", "protein", "carbs", "fat"],
            "additionalProperties": False,
        },
    },
    "required": ["title", "description", "prepMinutes", "cookMinutes", "servings", "difficulty",
                 "cuisine", "mealType", "tags", "dietaryTags", "ingredients", "steps", "nutrition"],
    "additionalProperties": False,
}
# Claude's response: everything except id / imageUrl / imageStatus (spec step 5), and
# without the derived fields the server fills in (totalMinutes, display, order).
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": sorted(ALLOWED_STATUSES)},
        "assistantMessage": {"type": "string"},
        "recipes": {"type": "array", "items": _RECIPE_SCHEMA},
        "followUps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"}, "prompt": {"type": "string"},
                    "type": {"type": "string", "enum": ["more", "refine"]},
                },
                "required": ["label", "prompt", "type"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["status", "assistantMessage", "recipes", "followUps"],
    "additionalProperties": False,
}


# ── Validation ────────────────────────────────────────────────────────────────

def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _str_list(v, field: str, *, max_items: int, max_len: int = 100) -> list[str]:
    if v is None:
        return []
    if not isinstance(v, list) or len(v) > max_items:
        raise ApiError(400, "invalid_request", f"{field} must be an array of at most {max_items} strings.")
    out = []
    for item in v:
        if not _is_str(item) or len(item) > max_len:
            raise ApiError(400, "invalid_request", f"{field} must contain strings.")
        if item.strip():
            out.append(item.strip())
    return out


def _validate_headers(headers) -> None:
    ctype = (headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if ctype != "application/json":
        raise ApiError(400, "invalid_request", "Content-Type must be application/json.")
    if (headers.get("X-Client") or "").strip() not in ALLOWED_CLIENTS:
        raise ApiError(400, "invalid_request", "X-Client must be ios-app or android-app.")
    try:
        uuid.UUID((headers.get("X-Request-Id") or "").strip())
    except ValueError:
        raise ApiError(400, "invalid_request", "X-Request-Id must be a UUID.")


def _validate_body(raw: bytes) -> dict:
    if len(raw) > MAX_BODY_BYTES:
        raise ApiError(400, "invalid_request", "Request body exceeds 64 KB.")
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ApiError(400, "invalid_request", "Body must be valid UTF-8 JSON.")
    if not isinstance(body, dict):
        raise ApiError(400, "invalid_request", "Body must be a JSON object.")

    def bad(msg: str):
        raise ApiError(400, "invalid_request", msg)

    mode = body.get("mode")
    if mode not in ALLOWED_MODES:
        bad("mode must be suggestion, chat or follow_up.")
    # Cards can change without an app release (app_configs/ask_recipes), so accept any
    # well-formed id rather than a fixed list; the prompt is what drives the results.
    sid = body.get("suggestionId")
    if mode == "suggestion" and (not _is_str(sid) or not _SUGGESTION_ID_RE.match(sid)):
        bad("suggestionId is required for suggestion mode (letters, numbers, _ or -, max 50).")
    if sid is not None and not _is_str(sid):
        bad("suggestionId must be a string.")
    prompt = body.get("prompt")
    if not _is_str(prompt) or not prompt.strip():
        bad("prompt is required.")
    if len(prompt) > MAX_PROMPT_CHARS:
        bad(f"prompt must be at most {MAX_PROMPT_CHARS} characters.")

    conv = body.get("conversationId")
    if conv is not None and (not _is_str(conv) or not conv.strip() or len(conv) > 100):
        bad("conversationId must be a string or null.")

    history = body.get("history") or []
    if not isinstance(history, list) or len(history) > MAX_HISTORY:
        bad(f"history must be an array of at most {MAX_HISTORY} items.")
    clean_history = []
    for i, turn in enumerate(history):
        if (not isinstance(turn, dict) or turn.get("role") not in ("user", "assistant")
                or not _is_str(turn.get("text")) or len(turn["text"]) > MAX_HISTORY_TEXT_CHARS):
            bad(f"history[{i}] must be {{role: user|assistant, text, recipeIds?}}.")
        ids = _str_list(turn.get("recipeIds"), f"history[{i}].recipeIds", max_items=10)
        clean_history.append({"role": turn["role"], "text": turn["text"], "recipeIds": ids})

    exclude = _str_list(body.get("excludeRecipeIds"), "excludeRecipeIds", max_items=MAX_EXCLUDE)

    count = body.get("count", 3)
    if count is None:
        count = 3
    if not _is_int(count) or not 1 <= count <= 5:
        bad("count must be an integer from 1 to 5.")

    prefs = body.get("preferences") or {}
    if not isinstance(prefs, dict):
        bad("preferences must be an object.")
    servings = prefs.get("servings")
    if servings is None:
        servings = 2
    if not _is_int(servings) or not 1 <= servings <= 20:
        bad("preferences.servings must be an integer from 1 to 20.")
    max_minutes = prefs.get("maxMinutes")
    if max_minutes is not None and (not _is_int(max_minutes) or max_minutes <= 0):
        bad("preferences.maxMinutes must be a positive integer or null.")
    clean_prefs = {
        "servings": servings,
        "dietaryTags": [t.lower() for t in _str_list(prefs.get("dietaryTags"), "preferences.dietaryTags", max_items=10)],
        "allergies": _str_list(prefs.get("allergies"), "preferences.allergies", max_items=20),
        "maxMinutes": max_minutes,
        "cuisines": _str_list(prefs.get("cuisines"), "preferences.cuisines", max_items=10),
    }
    return {
        "mode": mode, "suggestionId": body.get("suggestionId"), "prompt": prompt.strip(),
        "conversationId": conv.strip() if conv else None, "history": clean_history,
        "excludeRecipeIds": exclude, "count": count, "preferences": clean_prefs,
    }


# ── Messages (spec step 4) ────────────────────────────────────────────────────

def _build_messages(body: dict, titles: dict[str, str]) -> list[dict]:
    """History first (text -> content, same-role neighbours merged, leading assistant
    dropped), then one final user turn with prompt, preferences and titles to avoid."""
    msgs: list[dict] = []
    for turn in body["history"]:
        content = turn["text"].strip()
        if turn["role"] == "assistant" and turn["recipeIds"]:
            shown = [titles[i] for i in turn["recipeIds"] if i in titles]
            if shown:
                content += "\n(Recipes shown: " + "; ".join(shown) + ")"
        if not content:
            continue
        if not msgs and turn["role"] == "assistant":
            continue
        if msgs and msgs[-1]["role"] == turn["role"]:
            msgs[-1]["content"] += "\n\n" + content
        else:
            msgs.append({"role": turn["role"], "content": content})

    p = body["preferences"]
    lines = [
        f"Request: {body['prompt']}",
        *(["(This is a preset suggestion card: always return results, never clarify.)"]
          if body["mode"] == "suggestion" else []),
        f"count: {body['count']}",
        f"Servings: {p['servings']}",
    ]
    if p["dietaryTags"]:
        lines.append("Dietary preferences: " + ", ".join(p["dietaryTags"]))
    lines.append("allergies (never include): " + (", ".join(p["allergies"]) if p["allergies"] else "none"))
    if p["maxMinutes"]:
        lines.append(f"Max total time: {p['maxMinutes']} minutes")
    if p["cuisines"]:
        lines.append("Preferred cuisines (soft preference): " + ", ".join(p["cuisines"]))
    avoid = [titles[i] for i in body["excludeRecipeIds"] if i in titles]
    if avoid:
        lines.append("Do not repeat: " + "; ".join(avoid))
    final = "\n".join(lines)
    if msgs and msgs[-1]["role"] == "user":
        msgs[-1]["content"] += "\n\n" + final
    else:
        msgs.append({"role": "user", "content": final})
    return msgs


# ── Output normalization ──────────────────────────────────────────────────────

def _int(v, default=0, lo=0, hi=10_000) -> int:
    try:
        n = int(round(float(v)))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def _ingredient_display(ing: dict) -> str:
    parts = [ing["amount"], ing["unit"], ing["name"]]
    text = " ".join(p for p in parts if p)
    return f"{text} ({ing['note']})" if ing["note"] else text


def _allergy_terms(allergies: list[str]) -> list[str]:
    terms = set()
    for a in allergies:
        a = a.lower().strip()
        if len(a) >= 3:
            terms.add(a)
            if a.endswith("s") and len(a) > 3:
                terms.add(a[:-1])
    return sorted(terms)


def _normalize_recipe(raw: dict, servings_default: int) -> dict | None:
    if not isinstance(raw, dict) or not _is_str(raw.get("title")) or not raw["title"].strip():
        return None
    ingredients = []
    for ing in raw.get("ingredients") or []:
        if not isinstance(ing, dict) or not _is_str(ing.get("name")) or not ing["name"].strip():
            continue
        item = {k: (str(ing.get(k) or "").strip()) for k in ("name", "amount", "unit", "note")}
        item["display"] = _ingredient_display(item)
        ingredients.append(item)
    steps = []
    for st in raw.get("steps") or []:
        if not isinstance(st, dict) or not _is_str(st.get("instruction")) or not st["instruction"].strip():
            continue
        step = {"order": len(steps) + 1, "instruction": st["instruction"].strip()}
        if _is_int(st.get("durationMinutes")) and st["durationMinutes"] > 0:
            step["durationMinutes"] = st["durationMinutes"]
        steps.append(step)
    if not ingredients or not steps:
        return None
    prep = _int(raw.get("prepMinutes"))
    cook = _int(raw.get("cookMinutes"))
    return {
        "title": raw["title"].strip(),
        "description": str(raw.get("description") or "").strip(),
        "prepMinutes": prep,
        "cookMinutes": cook,
        "totalMinutes": prep + cook,
        "servings": _int(raw.get("servings"), servings_default, 1, 50) or servings_default,
        "difficulty": raw.get("difficulty") if raw.get("difficulty") in DIFFICULTIES else "Easy",
        "cuisine": str(raw.get("cuisine") or "").strip(),
        "mealType": raw.get("mealType") if raw.get("mealType") in MEAL_TYPES else "Dinner",
        "tags": [t.strip() for t in (raw.get("tags") or []) if _is_str(t) and t.strip()][:3],
        "dietaryTags": sorted({t.strip().lower().replace(" ", "_").replace("-", "_")
                               for t in (raw.get("dietaryTags") or []) if _is_str(t) and t.strip()}),
        "ingredients": ingredients,
        "steps": steps,
        "_model_nutrition": raw.get("nutrition") if isinstance(raw.get("nutrition"), dict) else None,
    }


def _normalize_follow_ups(raw) -> list[dict]:
    out, seen = [], set()
    for f in raw or []:
        if not isinstance(f, dict) or not _is_str(f.get("label")) or not f["label"].strip():
            continue
        label = f["label"].strip()
        prompt = (f.get("prompt") or label).strip() if _is_str(f.get("prompt")) else label
        kind = f.get("type") if f.get("type") in ("more", "refine") else "refine"
        if label.lower() in seen:
            continue
        seen.add(label.lower())
        out.append({"label": label, "prompt": prompt, "type": kind})
    more = {"label": "Show more like these", "prompt": "Show more like these", "type": "more"}
    out = [f for f in out if f["type"] != "more"]
    out.insert(0, more)
    if len(out) < 2:
        out.append({"label": "Something quicker", "prompt": "Something quicker", "type": "refine"})
    return out[:4]


def _parse_model_json(resp) -> dict:
    for block in getattr(resp, "content", None) or []:
        if getattr(block, "type", None) == "text" and (getattr(block, "text", "") or "").strip():
            try:
                data = json.loads(block.text)
            except json.JSONDecodeError:
                raise ApiError(502, "model_error", "The model returned invalid JSON.")
            if isinstance(data, dict):
                return data
    raise ApiError(502, "model_error", "The model returned no answer.")


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception as e:
        print(f"[ask-recipes] {getattr(fn, '__name__', 'call')} failed: {type(e).__name__}: {e}")
        return None


def _max_tokens(count: int) -> int:
    return min(CLAUDE_MAX_TOKENS, 600 + TOKENS_PER_RECIPE * count)


def _slug(title: str) -> str:
    """Dish key for the image library: lowercase words joined by '-'."""
    import re
    words = re.findall(r"[a-z0-9]+", (title or "").lower())
    return "-".join(words)[:120] or "recipe"


def _results_cache_key(body: dict) -> str | None:
    """Key for the whole-answer cache, or None when the request is not repeatable."""
    if RESULTS_CACHE_HOURS <= 0 or body["history"] or body["excludeRecipeIds"] or body["mode"] == "follow_up":
        return None
    import hashlib
    prefs = body["preferences"]
    norm = {
        "prompt": " ".join(body["prompt"].lower().split()),
        "count": body["count"],
        "servings": prefs["servings"],
        "dietaryTags": sorted(prefs["dietaryTags"]),
        "allergies": sorted(a.lower() for a in prefs["allergies"]),
        "maxMinutes": prefs["maxMinutes"],
        "cuisines": sorted(c.lower() for c in prefs["cuisines"]),
        "model": CLAUDE_MODEL,
        "v": 2,  # bump when the response shape changes (v2: micronutrients)
    }
    return hashlib.sha256(json.dumps(norm, sort_keys=True).encode()).hexdigest()


# ── Firestore cache / images ──────────────────────────────────────────────────

def _results_cache_get(db, key: str) -> dict | None:
    snap = db.collection(RESULTS_CACHE_COLLECTION).document(key).get()
    if not snap.exists:
        return None
    d = snap.to_dict() or {}
    exp = d.get("expireAt")
    if exp is not None and exp < datetime.now(timezone.utc):
        return None
    return d if isinstance(d.get("recipes"), list) and d["recipes"] else None


def _results_cache_put(db, key: str, entry: dict) -> None:
    from google.cloud import firestore as gcf
    db.collection(RESULTS_CACHE_COLLECTION).document(key).set({
        **entry, "createdAt": gcf.SERVER_TIMESTAMP,
        "expireAt": datetime.now(timezone.utc) + timedelta(hours=RESULTS_CACHE_HOURS),
    })


def _image_library_lookup(db, slugs: list[str]) -> dict[str, str]:
    slugs = list(dict.fromkeys(slugs))
    if not slugs:
        return {}
    refs = [db.collection(IMAGE_LIBRARY_COLLECTION).document(s) for s in slugs]
    out = {}
    for snap in db.get_all(refs):
        if snap.exists and _is_str((snap.to_dict() or {}).get("imageUrl")):
            out[snap.id] = snap.to_dict()["imageUrl"]
    return out


def _image_library_store(db, slug: str, title: str, url: str) -> None:
    from google.cloud import firestore as gcf
    db.collection(IMAGE_LIBRARY_COLLECTION).document(slug).set(
        {"imageUrl": url, "title": title, "createdAt": gcf.SERVER_TIMESTAMP})


def _cache_titles(db, uid: str, ids: list[str]) -> dict[str, str]:
    """Look up titles for recipe ids this user was shown (one batched read)."""
    ids = [i for i in dict.fromkeys(ids) if i.startswith("r_")]
    if not ids:
        return {}
    refs = [db.collection(CACHE_COLLECTION).document(i) for i in ids]
    out = {}
    for snap in db.get_all(refs):
        if snap.exists:
            d = snap.to_dict() or {}
            if d.get("uid") == uid and _is_str((d.get("recipe") or {}).get("title")):
                out[snap.id] = d["recipe"]["title"]
    return out


def _cache_store(db, uid: str, conversation_id: str, recipes: list[dict]) -> None:
    from google.cloud import firestore as gcf
    batch = db.batch()
    expire = datetime.now(timezone.utc) + timedelta(days=CACHE_TTL_DAYS)
    for r in recipes:
        batch.set(db.collection(CACHE_COLLECTION).document(r["id"]), {
            "uid": uid, "conversationId": conversation_id, "recipe": r,
            "imageUrl": r["imageUrl"], "imageStatus": r["imageStatus"],
            "createdAt": gcf.SERVER_TIMESTAMP, "expireAt": expire,
        })
    batch.commit()


def _cache_image_update(db, recipe_id: str, url: str | None) -> None:
    db.collection(CACHE_COLLECTION).document(recipe_id).set({
        "imageUrl": url, "imageStatus": "ready",
        "recipe": {"imageUrl": url, "imageStatus": "ready"},
    }, merge=True)


def _backfill_saved_recipe_image(db, uid: str, recipe_id: str, url: str) -> int:
    """If the user saved this recipe before its image finished, write the Storage URL onto
    the saved copy so it is never lost. Finds the saved doc either by document id == recipe
    id, or by an `askRecipeId` field. Only fills an empty image field. Returns docs updated."""
    col = db.collection(cr.USERS_COLLECTION).document(uid).collection(SAVED_RECIPES_SUBCOLLECTION)
    refs = {}
    snap = col.document(recipe_id).get()
    if snap.exists:
        refs[snap.reference.path] = snap
    for s in col.where("askRecipeId", "==", recipe_id).limit(5).stream():
        refs[s.reference.path] = s
    updated = 0
    for s in refs.values():
        if not (s.to_dict() or {}).get(SAVED_IMAGE_FIELD):
            s.reference.set({SAVED_IMAGE_FIELD: url, "imageStatus": "ready"}, merge=True)
            updated += 1
    return updated


def _default_suggestions(db) -> list[dict]:
    try:
        col, doc = SUGGESTIONS_CONFIG_DOC.split("/", 1)
        snap = db.collection(col).document(doc).get()
        items = (snap.to_dict() or {}).get("suggestions") if snap.exists else None
        if isinstance(items, list) and items:
            return [s for s in items if isinstance(s, dict) and s.get("id") and s.get("prompt")]
    except Exception as e:
        print(f"[ask-recipes] suggestions config read failed: {type(e).__name__}: {e}")
    return DEFAULT_SUGGESTIONS


# ── Blueprint ─────────────────────────────────────────────────────────────────

def create_ask_recipes_blueprint(
    *,
    recompute_nutrition: Callable[[dict], dict | None] | None = None,
    generate_image: Callable[[str, dict], str | None] | None = None,
    verify_token: Callable[[str], str] = cr._default_verify_token,
    get_db: Callable[[], Any] = cr._default_firestore,
    claude_call: Callable[..., Any] = cr._default_claude_call,
    is_premium: Callable[[Any, str], bool] = cr._is_premium,
    read_usage: Callable[[Any, str], int] = lambda db, uid: cr._read_usage_count(db, uid, USAGE_PREFIX),
    increment_usage: Callable[[Any, str, str], int] = lambda db, uid, rid: cr._increment_usage(db, uid, rid, USAGE_PREFIX),
    cache_titles: Callable[[Any, str, list], dict] = _cache_titles,
    cache_store: Callable[[Any, str, str, list], None] = _cache_store,
    cache_image_update: Callable[[Any, str, str | None], None] = _cache_image_update,
    backfill_saved_image: Callable[[Any, str, str, str], int] = _backfill_saved_recipe_image,
    load_suggestions: Callable[[Any], list] = _default_suggestions,
    results_cache_get: Callable[[Any, str], dict | None] = _results_cache_get,
    results_cache_put: Callable[[Any, str, dict], None] = _results_cache_put,
    image_library_lookup: Callable[[Any, list], dict] = _image_library_lookup,
    image_library_store: Callable[[Any, str, str, str], None] = _image_library_store,
    timeout_seconds: float | None = None,
    images_enabled: bool | None = None,
) -> Blueprint:
    bp = Blueprint("ask_recipes", __name__)
    budget = TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    images_on = (IMAGES_ENABLED if images_enabled is None else images_enabled) and generate_image is not None

    executor = ThreadPoolExecutor(max_workers=_env_int("ASK_RECIPES_WORKERS", 8), thread_name_prefix="ask-recipes")
    nutrition_pool = ThreadPoolExecutor(max_workers=_env_int("ASK_RECIPES_NUTRITION_WORKERS", 10),
                                        thread_name_prefix="ask-nutrition")
    image_pool = ThreadPoolExecutor(max_workers=_env_int("ASK_RECIPES_IMAGE_WORKERS", 4),
                                    thread_name_prefix="ask-image")
    # Images finished on this worker (fast path for polling before Firestore catches up).
    image_results: dict[str, tuple[str, str | None]] = {}  # recipe id -> (uid, url) when generation ends
    # Every recipe this worker returned: id -> (uid, url known at response time, e.g. a reused
    # library image). Lets the poll answer with the real Storage URL even if Firestore is slow.
    issued: dict[str, tuple[str, str | None]] = {}
    inflight_slugs: set[str] = set()   # dishes whose image this worker is generating now
    inflight_lock = threading.Lock()

    # Duplicate-retry handling (same as /customize-recipe): uid + X-Request-Id.
    dedupe_ttl = _env_int("ASK_RECIPES_DEDUPE_TTL_SECONDS", 900)
    dedupe: dict[str, tuple[float, Future]] = {}
    dedupe_lock = threading.Lock()

    def _dedupe_claim(key: str) -> tuple[Future, bool]:
        now = time.monotonic()
        with dedupe_lock:
            for k in [k for k, (ts, f) in dedupe.items() if f.done() and now - ts > dedupe_ttl]:
                dedupe.pop(k, None)
            if key in dedupe:
                return dedupe[key][1], False
            fut: Future = Future()
            dedupe[key] = (now, fut)
            return fut, True

    def _dedupe_release(key: str, fut: Future, result) -> None:
        with dedupe_lock:
            if result is not None and result[1] == 200:
                dedupe[key] = (time.monotonic(), fut)
                fut.set_result(result)
            else:
                dedupe.pop(key, None)
                fut.set_result(None)

    def _authenticate() -> str:
        auth_header = request.headers.get("Authorization") or ""
        if not auth_header.lower().startswith("bearer ") or not auth_header[7:].strip():
            raise ApiError(401, "unauthenticated", "Missing Firebase ID token.")
        try:
            uid = verify_token(auth_header[7:].strip())
        except Exception as e:
            print(f"[ask-recipes] token verification failed: {type(e).__name__}: {e}")
            raise ApiError(401, "unauthenticated", "Your session has expired. Please sign in again.")
        if not uid:
            raise ApiError(401, "unauthenticated", "Invalid Firebase ID token.")
        return uid

    def _generate_image_bg(recipe_id: str, recipe: dict, db, uid: str) -> None:
        url = None
        slug = _slug(recipe.get("title"))
        try:
            url = generate_image(recipe_id, recipe)
        except Exception as e:
            print(f"[ask-recipes] image generation failed for {recipe_id}: {type(e).__name__}: {e}")
        finally:
            with inflight_lock:
                inflight_slugs.discard(slug)
        image_results[recipe_id] = (uid, url)
        if url:
            try:
                image_library_store(db, slug, recipe.get("title") or "", url)
            except Exception as e:
                print(f"[ask-recipes] image library write failed for {slug}: {type(e).__name__}: {e}")
        try:
            cache_image_update(db, recipe_id, url)
        except Exception as e:
            print(f"[ask-recipes] image cache update failed for {recipe_id}: {type(e).__name__}: {e}")
        if url:
            try:
                n = backfill_saved_image(db, uid, recipe_id, url)
                if n:
                    print(f"[ask-recipes] image backfilled onto {n} saved recipe(s) for {recipe_id}")
            except Exception as e:
                print(f"[ask-recipes] saved-recipe image backfill failed for {recipe_id}: {type(e).__name__}: {e}")

    def _run_model_and_nutrition(body: dict, messages: list, deadline: float) -> dict:
        t0 = time.monotonic()
        resp = claude_call(
            model=CLAUDE_MODEL,
            system=SYSTEM_PROMPT,
            messages=messages,
            output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
            max_tokens=_max_tokens(body["count"]),
            timeout=max(1.0, deadline - t0),
        )
        claude_ms = int((time.monotonic() - t0) * 1000)
        usage = getattr(resp, "usage", None)
        tokens = {"input": getattr(usage, "input_tokens", None), "output": getattr(usage, "output_tokens", None)}
        if getattr(resp, "stop_reason", None) == "max_tokens":
            raise ApiError(502, "model_error", "The model response was cut off.")
        out = _parse_model_json(resp)

        status = out.get("status")
        if status not in ALLOWED_STATUSES:
            raise ApiError(502, "model_error", "The model returned an invalid status.")
        message = out.get("assistantMessage")
        if not _is_str(message) or not message.strip():
            raise ApiError(502, "model_error", "The model returned an empty message.")
        result = {"status": status, "assistantMessage": message.strip()[:MAX_ASSISTANT_MESSAGE_CHARS],
                  "recipes": [], "followUps": [], "_tokens": tokens,
                  "_timings": {"claudeMs": claude_ms, "nutritionMs": 0}, "_dropped": 0}
        if status != "results":
            return result

        servings = body["preferences"]["servings"]
        allergy_terms = _allergy_terms(body["preferences"]["allergies"])
        recipes, dropped = [], 0
        for raw in out.get("recipes") or []:
            r = _normalize_recipe(raw, servings)
            if r is None:
                dropped += 1
                continue
            names = " | ".join(i["name"].lower() + " " + i["note"].lower() for i in r["ingredients"])
            if any(t in names for t in allergy_terms):
                print(f"[ask-recipes] dropped recipe containing an allergen: {r['title']!r}")
                dropped += 1
                continue
            recipes.append(r)
            if len(recipes) == body["count"]:
                break
        if not recipes:
            raise ApiError(502, "model_error", "The model returned no usable recipes.")
        result["_dropped"] = dropped

        # Step 6: nutrition for every recipe, in parallel.
        t1 = time.monotonic()
        futures = {}
        if recompute_nutrition is not None:
            for idx, r in enumerate(recipes):
                futures[idx] = nutrition_pool.submit(recompute_nutrition, r)
        for idx, r in enumerate(recipes):
            recomputed = None
            if idx in futures:
                try:
                    recomputed = futures[idx].result(timeout=max(0.5, deadline - time.monotonic()))
                except Exception as e:
                    print(f"[ask-recipes] nutrition recompute failed: {type(e).__name__}: {e}")
            r["nutrition"] = cr._final_nutrition(recomputed, r.pop("_model_nutrition"), {})
        result["_timings"]["nutritionMs"] = int((time.monotonic() - t1) * 1000)
        result["recipes"] = recipes
        result["followUps"] = _normalize_follow_ups(out.get("followUps"))
        return result

    @bp.route("/ask-recipes", methods=["POST"])
    def ask_recipes():
        started = time.monotonic()
        deadline = started + budget
        request_id = (request.headers.get("X-Request-Id") or "").strip() or str(uuid.uuid4())
        log: dict[str, Any] = {"event": "ask_recipes", "requestId": request_id, "uid": None,
                               "client": request.headers.get("X-Client"), "model": None,
                               "inputTokens": None, "outputTokens": None}
        timings: dict[str, int] = {}

        def respond(payload: dict, http: int):
            log["http"] = http
            log["latencyMs"] = int((time.monotonic() - started) * 1000)
            log["timings"] = timings
            print("[ask-recipes] " + json.dumps(log, ensure_ascii=False))
            return jsonify(payload), http

        dedupe_key = dedupe_fut = final = None
        try:
            # 1. Authenticate
            t = time.monotonic()
            uid = _authenticate()
            log["uid"] = uid
            timings["authMs"] = int((time.monotonic() - t) * 1000)

            # 2. Validate
            if request.content_length is not None and request.content_length > MAX_BODY_BYTES:
                raise ApiError(400, "invalid_request", "Request body exceeds 64 KB.")
            _validate_headers(request.headers)
            body = _validate_body(request.get_data(cache=False) or b"")
            conversation_id = body["conversationId"] or ("c_" + uuid.uuid4().hex[:12])
            log.update({"mode": body["mode"], "suggestionId": body.get("suggestionId"),
                        "conversationId": conversation_id, "count": body["count"]})

            dedupe_key = f"{uid}:{request_id}"
            fut, owner = _dedupe_claim(dedupe_key)
            if not owner:
                try:
                    prior = fut.result(timeout=max(0.0, deadline - time.monotonic()))
                except FutureTimeoutError:
                    raise ApiError(504, "timeout", "This is taking longer than expected. Please try again.")
                if prior is not None:
                    log["duplicate"] = True
                    return respond(*prior)
                fut, owner = _dedupe_claim(dedupe_key)
                if not owner:
                    raise ApiError(504, "timeout", "This is taking longer than expected. Please try again.")
            dedupe_fut = fut

            # 3. Check the limit
            t = time.monotonic()
            db = get_db()
            premium = bool(is_premium(db, uid))
            limit = PREMIUM_DAILY_LIMIT if premium else FREE_DAILY_LIMIT
            log["premium"] = premium
            if limit <= 0:
                raise ApiError(403, "premium_required", "Recipe ideas are a premium feature.")
            used = int(read_usage(db, uid))
            if used >= limit:
                msg = (f"You've used today's {limit} free recipe ideas." if not premium
                       else f"You've used today's {limit} recipe ideas. Come back tomorrow!")
                raise ApiError(429, "limit_reached", msg)

            # 4. Build the messages (titles of shown/excluded recipes from the cache)
            ids = list(body["excludeRecipeIds"]) + [i for h in body["history"] for i in h["recipeIds"]]
            try:
                titles = cache_titles(db, uid, ids) if ids else {}
            except Exception as e:
                print(f"[ask-recipes] cache title lookup failed: {type(e).__name__}: {e}")
                titles = {}
            messages = _build_messages(body, titles)
            timings["firestoreReadMs"] = int((time.monotonic() - t) * 1000)

            # Repeatable first turn already answered recently? Reuse it (no Claude call).
            cache_key = _results_cache_key(body)
            cached = None
            if cache_key:
                t = time.monotonic()
                try:
                    cached = results_cache_get(db, cache_key)
                except Exception as e:
                    print(f"[ask-recipes] results cache read failed: {type(e).__name__}: {e}")
                timings["resultsCacheMs"] = int((time.monotonic() - t) * 1000)
            log["cacheHit"] = bool(cached)

            # 5 + 6. Claude, then nutrition (within the 45 s budget)
            log["model"] = CLAUDE_MODEL
            if cached:
                job = Future()
                job.set_result({"status": "results", "assistantMessage": cached.get("assistantMessage") or "",
                                "recipes": [dict(r) for r in cached["recipes"]][:body["count"]],
                                "followUps": cached.get("followUps") or _normalize_follow_ups([]),
                                "_tokens": {}, "_timings": {}, "_dropped": 0})
            else:
                job = executor.submit(_run_model_and_nutrition, body, messages, deadline)
            try:
                result = job.result(timeout=max(0.0, deadline - time.monotonic()))
            except FutureTimeoutError:
                raise ApiError(504, "timeout", "This is taking longer than expected. Please try again.")
            except ApiError:
                raise
            except Exception as e:
                if cr._is_timeout_exception(e):
                    raise ApiError(504, "timeout", "This is taking longer than expected. Please try again.")
                print(f"[ask-recipes] Claude call failed: {type(e).__name__}: {e}")
                raise ApiError(502, "model_error", "We couldn't find recipes right now. Please try again.")
            tokens = result.pop("_tokens", {}) or {}
            timings.update(result.pop("_timings", {}) or {})
            log.update({"inputTokens": tokens.get("input"), "outputTokens": tokens.get("output"),
                        "status": result["status"], "recipes": len(result["recipes"]),
                        "dropped": result.pop("_dropped", 0)})
            if cache_key and not cached and result["status"] == "results":
                entry = {"assistantMessage": result["assistantMessage"],
                         "recipes": result["recipes"], "followUps": result["followUps"]}
                executor.submit(lambda: _safe(results_cache_put, db, cache_key, entry))

            # 7. Ids, cache and images (reuse a library image for a known dish instantly)
            library = {}
            if result["recipes"] and images_on:
                t = time.monotonic()
                try:
                    library = image_library_lookup(db, [_slug(r["title"]) for r in result["recipes"]])
                except Exception as e:
                    print(f"[ask-recipes] image library read failed: {type(e).__name__}: {e}")
                timings["imageLibraryMs"] = int((time.monotonic() - t) * 1000)
            recipes = []
            for r in result["recipes"]:
                rid = "r_" + uuid.uuid4().hex[:12]
                url = library.get(_slug(r["title"]))
                recipes.append({"id": rid, "title": r["title"], "description": r["description"],
                                "imageUrl": url,
                                "imageStatus": "pending" if (images_on and not url) else "ready",
                                **{k: v for k, v in r.items()
                                   if k not in ("id", "title", "description", "imageUrl", "imageStatus")}})
            log["imagesReused"] = sum(1 for r in recipes if r["imageUrl"])
            for r in recipes:
                issued[r["id"]] = (uid, r["imageUrl"])
            if recipes:
                t = time.monotonic()
                try:
                    cache_store(db, uid, conversation_id, recipes)
                except Exception as e:
                    print(f"[ask-recipes] cache write failed: {type(e).__name__}: {e}")
                timings["cacheWriteMs"] = int((time.monotonic() - t) * 1000)
                if images_on:
                    for r in recipes:
                        if r["imageStatus"] != "pending":
                            continue
                        slug = _slug(r["title"])
                        with inflight_lock:
                            if slug in inflight_slugs:
                                continue  # same dish already generating; poll resolves via the library
                            inflight_slugs.add(slug)
                        image_pool.submit(_generate_image_bg, r["id"], r, db, uid)

            # 8. Count the use (only for results)
            if result["status"] == "results":
                t = time.monotonic()
                try:
                    used = int(increment_usage(db, uid, request_id))
                except Exception as e:
                    print(f"[ask-recipes] usage increment failed: {type(e).__name__}: {e}")
                    used += 1
                timings["usageWriteMs"] = int((time.monotonic() - t) * 1000)

            payload = {
                "requestId": request_id,
                "conversationId": conversation_id,
                "status": result["status"],
                "assistantMessage": result["assistantMessage"],
                "recipes": recipes,
                "followUps": result["followUps"],
                "usage": {"usedToday": used, "limitToday": limit},
            }
            final = (payload, 200)
            return respond(payload, 200)
        except ApiError as err:
            log["error"] = err.code
            if err.http == 400:
                log["errorMessage"] = err.message  # which field failed validation
            return respond(_error_body(request_id, err), err.http)
        except Exception as e:
            log["error"] = "internal_error"
            print(f"[ask-recipes] unexpected error: {type(e).__name__}: {e}")
            return respond(_error_body(request_id, ApiError(
                500, "internal_error", "We ran into a problem. Please try again in a moment.")), 500)
        finally:
            if dedupe_key is not None and dedupe_fut is not None:
                _dedupe_release(dedupe_key, dedupe_fut, final)

    @bp.route("/ask-recipes/image/<recipe_id>", methods=["GET"])
    def ask_recipes_image(recipe_id: str):
        request_id = (request.headers.get("X-Request-Id") or "").strip() or str(uuid.uuid4())
        try:
            uid = _authenticate()
            # The answer is always the actual Storage URL stored for this recipe, whatever the
            # file is called (a reused library image has another recipe's file name).
            done = image_results.get(recipe_id)
            if done and done[0] == uid and done[1]:  # generated on this worker
                return jsonify({"recipeId": recipe_id, "imageStatus": "ready", "imageUrl": done[1]}), 200
            known = issued.get(recipe_id)
            if known and known[0] == uid and known[1]:  # returned with a ready (reused) image
                return jsonify({"recipeId": recipe_id, "imageStatus": "ready", "imageUrl": known[1]}), 200
            snap = get_db().collection(CACHE_COLLECTION).document(recipe_id).get()
            data = (snap.to_dict() or {}) if snap.exists else {}
            if not data or data.get("uid") != uid:
                raise ApiError(404, "not_found", "Recipe not found.")
            status, url = data.get("imageStatus") or "ready", data.get("imageUrl")
            if status == "pending":
                title = (data.get("recipe") or {}).get("title") or ""
                url = image_library_lookup(get_db(), [_slug(title)]).get(_slug(title))
                created = data.get("createdAt")
                age = (datetime.now(timezone.utc) - created).total_seconds() if isinstance(created, datetime) else 0
                if url or age > IMAGE_GIVEUP_SECONDS:
                    status = "ready"
                    db = get_db()
                    _safe(cache_image_update, db, recipe_id, url)
                    if url:
                        _safe(backfill_saved_image, db, uid, recipe_id, url)
            return jsonify({"recipeId": recipe_id, "imageStatus": status, "imageUrl": url}), 200
        except ApiError as err:
            return jsonify(_error_body(request_id, err)), err.http

    @bp.route("/ask-recipes/suggestions", methods=["GET"])
    def ask_recipes_suggestions():
        request_id = (request.headers.get("X-Request-Id") or "").strip() or str(uuid.uuid4())
        try:
            _authenticate()
            return jsonify({"suggestions": load_suggestions(get_db())}), 200
        except ApiError as err:
            return jsonify(_error_body(request_id, err)), err.http

    return bp
