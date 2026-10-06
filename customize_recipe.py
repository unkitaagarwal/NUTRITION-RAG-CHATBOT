"""POST /customize-recipe — apply a change request to a recipe with Claude.

Spec: "Customize Recipe API" (Oct 4, 2026). Powers "Customize this recipe" on
the recipe detail screen, the import-URL screen and the iOS share sheet.

The handler runs seven steps in order and stops at the first failure:

    1. Authenticate   Firebase ID token -> uid                       (401)
    2. Validate       headers, sizes, shapes                         (400)
    3. Check limit    meal_plan_chef_users/{uid}/usage/customize_{d} (403/429)
    4. Call Claude    structured JSON: status/message/edits/changes   (502)
    5. Recompute      nutrition via the same step /extract-recipe uses
    6. Count the use  only when status == "updated"
    7. Log + return   uid, requestId, model, tokens, latency         (200)

Steps 4-5 must finish within CUSTOMIZE_TIMEOUT_SECONDS (45 s) or we return 504.

This module deliberately does not import app.py (avoids a circular import);
app.py builds the blueprint and injects the nutrition step:

    app.register_blueprint(
        create_customize_recipe_blueprint(recompute_nutrition=fn)
    )
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

from flask import Blueprint, jsonify, request

# ── Config ────────────────────────────────────────────────────────────────────

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


CUSTOMIZE_TIMEOUT_SECONDS = float(os.getenv("CUSTOMIZE_TIMEOUT_SECONDS", "45"))
MAX_BODY_BYTES = _env_int("CUSTOMIZE_MAX_BODY_BYTES", 64 * 1024)
MAX_PROMPT_CHARS = 500
MAX_INGREDIENTS = 60
MAX_STEPS = 60
MAX_HISTORY = 10
MAX_HISTORY_TEXT_CHARS = 4000
MAX_ASSISTANT_MESSAGE_CHARS = 1000  # spec says "about 600"; hard cap a bit above

FREE_DAILY_LIMIT = _env_int("CUSTOMIZE_FREE_DAILY_LIMIT", 3)        # 0 => premium-only (403)
PREMIUM_DAILY_LIMIT = _env_int("CUSTOMIZE_PREMIUM_DAILY_LIMIT", 50)
USAGE_TZ = os.getenv("CUSTOMIZE_USAGE_TZ", "UTC")  # which "today" the counter uses

# Haiku 4.5 for both chips and chat (cost/latency). Set CUSTOMIZE_CLAUDE_MODEL=claude-sonnet-5-5
# to send typed chat requests to Sonnet instead.
CLAUDE_MODEL = os.getenv("CUSTOMIZE_CLAUDE_MODEL", "claude-haiku-4-5-20251001")
CLAUDE_PRESET_MODEL = os.getenv("CUSTOMIZE_CLAUDE_PRESET_MODEL", "claude-haiku-4-5-20251001")
# Preset chip taps use Haiku (faster, cheaper); typed chat uses Sonnet.
# Set CUSTOMIZE_USE_PRESET_MODEL=0 to send presets to Sonnet too.
USE_PRESET_MODEL = (os.getenv("CUSTOMIZE_USE_PRESET_MODEL") or "1").strip().lower() in ("1", "true", "yes")
CLAUDE_MAX_TOKENS = _env_int("CUSTOMIZE_CLAUDE_MAX_TOKENS", 3000)  # edits only, so much smaller

USERS_COLLECTION = os.getenv("CUSTOMIZE_USERS_COLLECTION", "meal_plan_chef_users")

ALLOWED_MODES = {"preset", "chat"}
ALLOWED_PRESETS = {"spicier", "more_protein", "healthier", "dairy_free", "under_20", "kid_friendly"}
ALLOWED_SOURCES = {"recipe_detail", "recipe_ready", "share_extension"}
ALLOWED_CLIENTS = {"ios-app", "android-app", "ios-share-extension"}
ALLOWED_STATUSES = {"updated", "clarify", "declined"}
ALLOWED_CHANGE_KINDS = {
    "spicier", "more_protein", "healthier", "dairy_free",
    "quicker", "kid_friendly", "kept", "general",
}

SYSTEM_PROMPT = (
    "You customize home-cooking recipes for the RecipeVault app. "
    "Apply only the requested change. Keep the dish recognizable. "
    "Keep existing dietary tags unless the user explicitly asks to change them. "
    "Use realistic amounts for the given servings. "
    "Never claim a recipe is safe for an allergy. "
    "If the request is unclear, set status to clarify and ask one short question. "
    "If it is not about cooking this recipe, set status to declined. "
    "Write assistantMessage in a friendly tone: say what changed in 1-2 sentences, "
    "then ask whether they'd like to see a preview."
    # Output-format guidance (not part of the product prompt above):
    "\n\nRespond only with JSON matching the required schema. "
    "assistantMessage is plain text (no markdown), at most about 600 characters. "
    "Return ONLY what changes, never the whole recipe. Ingredients and steps are numbered "
    "by their field n. In edits.ingredientOps / edits.stepOps use: "
    "{op:'update', n, ...new values} to change item n (give all of name, amount, unit for "
    "ingredients, or instruction for steps); {op:'remove', n} to delete item n; "
    "{op:'add', n, ...} to insert a new item after item n (n=0 inserts at the start). "
    "Every n refers to the ORIGINAL numbering. Include a top-level edits field "
    "(title, description, servings, prepMinutes, cookMinutes, difficulty, cuisine, "
    "dietaryTags) only if it changes; omit unchanged ones. If servings change, update "
    "every affected ingredient amount. Always include edits.nutrition: your per-serving "
    "estimate for the updated recipe. "
    "When status is updated, also return 1-5 changes; otherwise set edits and changes "
    "to null. Use a change of kind 'kept' to reassure the user about something preserved "
    "(e.g. still vegetarian). Amounts are strings."
)

_INGREDIENT_OP_SCHEMA = {
    "type": "object",
    "properties": {
        "op": {"type": "string", "enum": ["update", "remove", "add"]},
        "n": {"type": "integer"},
        "name": {"type": "string"},
        "amount": {"type": "string"},
        "unit": {"type": "string"},
    },
    "required": ["op", "n"],
    "additionalProperties": False,
}
_STEP_OP_SCHEMA = {
    "type": "object",
    "properties": {
        "op": {"type": "string", "enum": ["update", "remove", "add"]},
        "n": {"type": "integer"},
        "instruction": {"type": "string"},
    },
    "required": ["op", "n"],
    "additionalProperties": False,
}
# Claude's response format (structured outputs via output_config.format). Claude
# returns only the edits; the server merges them into the client's recipe.
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": sorted(ALLOWED_STATUSES)},
        "assistantMessage": {"type": "string"},
        "edits": {
            "type": ["object", "null"],
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "servings": {"type": "integer"},
                "prepMinutes": {"type": "integer"},
                "cookMinutes": {"type": "integer"},
                "difficulty": {"type": "string"},
                "cuisine": {"type": "string"},
                "dietaryTags": {"type": "array", "items": {"type": "string"}},
                "ingredientOps": {"type": "array", "items": _INGREDIENT_OP_SCHEMA},
                "stepOps": {"type": "array", "items": _STEP_OP_SCHEMA},
                "nutrition": {
                    "type": "object",
                    "description": "Per-serving estimate for the updated recipe (the server recomputes it).",
                    "properties": {
                        "calories": {"type": "number"},
                        "protein": {"type": "number"},
                        "carbs": {"type": "number"},
                        "fat": {"type": "number"},
                    },
                    "required": ["calories", "protein", "carbs", "fat"],
                    "additionalProperties": False,
                },
            },
            "required": ["ingredientOps", "stepOps", "nutrition"],
            "additionalProperties": False,
        },
        "changes": {
            "type": ["array", "null"],
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": sorted(ALLOWED_CHANGE_KINDS)},
                    "title": {"type": "string"},
                    "detail": {"type": "string"},
                },
                "required": ["kind", "title", "detail"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["status", "assistantMessage", "edits", "changes"],
    "additionalProperties": False,
}


# ── Errors ────────────────────────────────────────────────────────────────────

class ApiError(Exception):
    def __init__(self, http: int, code: str, message: str):
        super().__init__(message)
        self.http = http
        self.code = code
        self.message = message


def _error_body(request_id: str, err: ApiError) -> dict:
    return {"requestId": request_id, "error": {"code": err.code, "message": err.message}}


# ── Default dependencies (overridable for tests) ──────────────────────────────

def _default_verify_token(id_token: str) -> str:
    """Verify a Firebase ID token against the RecipeVault (MealMap) project."""
    import firebase_admin
    from firebase_admin import auth as fb_auth
    from firebase_utils import init_mealmap_firestore

    init_mealmap_firestore()  # ensures the named 'mealmap' app exists
    app_name = os.getenv("CUSTOMIZE_FIREBASE_APP", "mealmap")
    decoded = fb_auth.verify_id_token(id_token, app=firebase_admin.get_app(app_name))
    return decoded["uid"]


def _default_firestore():
    from firebase_utils import init_mealmap_firestore
    return init_mealmap_firestore()


_anthropic_client = None
_anthropic_lock = threading.Lock()


def _default_claude_call(*, model: str, system: str, messages: list,
                         output_config: dict, max_tokens: int, timeout: float):
    global _anthropic_client
    if _anthropic_client is None:
        with _anthropic_lock:
            if _anthropic_client is None:
                import anthropic
                # Retries would blow the 45 s budget; the app retries on "Try again".
                # Keys not scoped to a workspace must name one on every request.
                workspace_id = (os.getenv("ANTHROPIC_WORKSPACE_ID") or "").strip()
                headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
                _anthropic_client = anthropic.Anthropic(max_retries=0, default_headers=headers)
    return _anthropic_client.messages.create(
        model=model, system=system, messages=messages,
        output_config=output_config, max_tokens=max_tokens, timeout=timeout,
    )


# ── Usage / premium (Firestore) ───────────────────────────────────────────────

def _usage_doc_id(now: datetime | None = None) -> str:
    now = now or datetime.now(ZoneInfo(USAGE_TZ))
    return f"customize_{now.strftime('%Y-%m-%d')}"


def _usage_ref(db, uid: str):
    return (db.collection(USERS_COLLECTION).document(uid)
              .collection("usage").document(_usage_doc_id()))


def _is_premium(db, uid: str) -> bool:
    """Premium = `is_premium_user` is true on meal_plan_chef_users/{uid}. Nothing else."""
    snap = db.collection(USERS_COLLECTION).document(uid).get()
    data = (snap.to_dict() or {}) if snap.exists else {}
    return data.get("is_premium_user") is True


def _read_usage_count(db, uid: str) -> int:
    snap = _usage_ref(db, uid).get()
    if not snap.exists:
        return 0
    try:
        return int((snap.to_dict() or {}).get("count") or 0)
    except (TypeError, ValueError):
        return 0


def _increment_usage(db, uid: str, request_id: str) -> int:
    """Increment today's counter once per X-Request-Id. Returns the new count."""
    from google.cloud import firestore as gcf

    ref = _usage_ref(db, uid)

    @gcf.transactional
    def _txn(transaction):
        snap = ref.get(transaction=transaction)
        data = (snap.to_dict() or {}) if snap.exists else {}
        count = int(data.get("count") or 0)
        seen = list(data.get("requestIds") or [])
        if request_id in seen:
            return count  # duplicate retry already counted (possibly on another worker)
        seen = (seen + [request_id])[-(PREMIUM_DAILY_LIMIT + 10):]
        transaction.set(ref, {
            "count": count + 1,
            "requestIds": seen,
            "updatedAt": gcf.SERVER_TIMESTAMP,
        }, merge=True)
        return count + 1

    return _txn(db.transaction())


# ── Validation ────────────────────────────────────────────────────────────────

def _is_str(v) -> bool:
    return isinstance(v, str)


def _validate_headers(headers) -> None:
    ctype = (headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if ctype != "application/json":
        raise ApiError(400, "invalid_request", "Content-Type must be application/json.")
    if (headers.get("X-Client") or "").strip() not in ALLOWED_CLIENTS:
        raise ApiError(400, "invalid_request", "X-Client must be ios-app, android-app or ios-share-extension.")
    rid = (headers.get("X-Request-Id") or "").strip()
    try:
        uuid.UUID(rid)
    except (ValueError, AttributeError):
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
        bad("mode must be 'preset' or 'chat'.")
    if mode == "preset" and body.get("presetId") not in ALLOWED_PRESETS:
        bad("presetId is required for preset mode and must be one of: " + ", ".join(sorted(ALLOWED_PRESETS)) + ".")
    if body.get("presetId") is not None and not _is_str(body.get("presetId")):
        bad("presetId must be a string.")

    prompt = body.get("prompt")
    if not _is_str(prompt) or not prompt.strip():
        bad("prompt is required.")
    if len(prompt) > MAX_PROMPT_CHARS:
        bad(f"prompt must be at most {MAX_PROMPT_CHARS} characters.")

    if body.get("source") not in ALLOWED_SOURCES:
        bad("source must be recipe_detail, recipe_ready or share_extension.")
    if body.get("recipeId") is not None and not _is_str(body.get("recipeId")):
        bad("recipeId must be a string or null.")

    recipe = body.get("recipe")
    if not isinstance(recipe, dict):
        bad("recipe is required.")
    ingredients = recipe.get("ingredients")
    if not isinstance(ingredients, list):
        bad("recipe.ingredients is required.")
    if len(ingredients) > MAX_INGREDIENTS:
        bad(f"recipe.ingredients must have at most {MAX_INGREDIENTS} items.")
    for i, ing in enumerate(ingredients):
        if not isinstance(ing, dict) or not _is_str(ing.get("name")):
            bad(f"recipe.ingredients[{i}] must be {{name, amount, unit}}.")
        for k in ("amount", "unit"):
            if ing.get(k) is not None and not isinstance(ing.get(k), (str, int, float)):
                bad(f"recipe.ingredients[{i}].{k} must be a string.")
    steps = recipe.get("steps")
    if not isinstance(steps, list):
        bad("recipe.steps is required.")
    if len(steps) > MAX_STEPS:
        bad(f"recipe.steps must have at most {MAX_STEPS} items.")
    for i, st in enumerate(steps):
        if not isinstance(st, dict) or not _is_str(st.get("instruction")):
            bad(f"recipe.steps[{i}] must be {{order, instruction}}.")
    nutrition = recipe.get("nutrition")
    if not isinstance(nutrition, dict) or nutrition.get("basis") != "per_serving":
        bad("recipe.nutrition.basis must be 'per_serving'.")

    history = body.get("history")
    if history is None:
        history = []
    if not isinstance(history, list):
        bad("history must be an array.")
    if len(history) > MAX_HISTORY:
        bad(f"history must have at most {MAX_HISTORY} items.")
    for i, turn in enumerate(history):
        if (not isinstance(turn, dict) or turn.get("role") not in ("user", "assistant")
                or not _is_str(turn.get("text"))):
            bad(f"history[{i}] must be {{role: user|assistant, text}}.")
        if len(turn["text"]) > MAX_HISTORY_TEXT_CHARS:
            bad(f"history[{i}].text is too long.")
    body["history"] = history
    return body


# ── Claude ────────────────────────────────────────────────────────────────────

def _pick_model(mode: str) -> str:
    return CLAUDE_PRESET_MODEL if (mode == "preset" and USE_PRESET_MODEL) else CLAUDE_MODEL


def _numbered_recipe(recipe: dict) -> dict:
    """The recipe as Claude sees it: ingredients and steps numbered 1..n by field `n`."""
    view = {k: v for k, v in recipe.items() if k not in ("ingredients", "steps")}
    view["ingredients"] = [
        {"n": i + 1, "name": ing.get("name"), "amount": ing.get("amount"), "unit": ing.get("unit")}
        for i, ing in enumerate(recipe.get("ingredients") or [])
    ]
    view["steps"] = [
        {"n": i + 1, "instruction": st.get("instruction")}
        for i, st in enumerate(recipe.get("steps") or [])
    ]
    return view


def _build_user_content(body: dict) -> str:
    """User content: recipe as JSON, then history, then prompt (spec step 4)."""
    history_lines = [f"{t['role']}: {t['text']}" for t in body["history"]]
    history_text = "\n".join(history_lines) if history_lines else "(none)"
    return (
        "<recipe>\n" + json.dumps(_numbered_recipe(body["recipe"]), ensure_ascii=False) + "\n</recipe>\n\n"
        "<history>\n" + history_text + "\n</history>\n\n"
        "<request>\n" + body["prompt"].strip() + "\n</request>"
    )


def _tool_input_from_response(resp) -> dict:
    """Return Claude's JSON answer (structured-output text block)."""
    for block in getattr(resp, "content", None) or []:
        if getattr(block, "type", None) == "text" and (getattr(block, "text", "") or "").strip():
            try:
                data = json.loads(block.text)
            except json.JSONDecodeError:
                raise ApiError(502, "model_error", "The model returned invalid JSON.")
            if isinstance(data, dict):
                return data
    raise ApiError(502, "model_error", "The model did not return a customization.")


def _num(v, *, integer: bool = False):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f < 0:
        return None
    if integer:
        return int(round(f))
    return int(round(f)) if abs(f - round(f)) < 0.05 else round(f, 1)


def _clean_ingredient(op: dict) -> dict | None:
    name = op.get("name")
    if not _is_str(name) or not name.strip():
        return None
    return {
        "name": name.strip(),
        "amount": "" if op.get("amount") is None else str(op.get("amount")).strip(),
        "unit": "" if op.get("unit") is None else str(op.get("unit")).strip(),
    }


def _clean_step(op: dict) -> dict | None:
    text = op.get("instruction")
    if not _is_str(text) or not text.strip():
        return None
    return {"instruction": text.strip()}


def _apply_ops(items: list, ops, clean) -> tuple[list, int]:
    """Apply update/remove/add ops (n = 1-based ORIGINAL position) to a list.

    Updates merge into the original item (so extra client fields survive), adds go
    after original item n (0 = start; out of range = end). Returns (new list, applied).
    """
    total = len(items)
    updates: dict[int, dict] = {}
    removed: set[int] = set()
    adds: dict[int, list] = {}
    applied = 0
    for op in ops or []:
        if not isinstance(op, dict):
            continue
        kind, n = op.get("op"), op.get("n")
        if not isinstance(n, int) or isinstance(n, bool):
            continue
        if kind == "remove" and 1 <= n <= total:
            removed.add(n)
            applied += 1
        elif kind == "update" and 1 <= n <= total:
            new = clean(op)
            if new:
                updates[n] = new
                applied += 1
        elif kind == "add":
            new = clean(op)
            if new:
                adds.setdefault(n if 0 <= n <= total else total, []).append(new)
                applied += 1
    out = list(adds.get(0, []))
    for i, item in enumerate(items, start=1):
        if i not in removed:
            merged = dict(item) if isinstance(item, dict) else {}
            merged.update(updates.get(i, {}))
            out.append(merged)
        out.extend(adds.get(i, []))
    return out, applied


def _normalize_model_output(out: dict, original_recipe: dict) -> dict:
    """Validate Claude's JSON and merge its edits into a copy of the client's recipe."""
    status = out.get("status")
    if status not in ALLOWED_STATUSES:
        raise ApiError(502, "model_error", "The model returned an invalid status.")
    message = out.get("assistantMessage")
    if not _is_str(message) or not message.strip():
        raise ApiError(502, "model_error", "The model returned an empty message.")
    message = message.strip()[:MAX_ASSISTANT_MESSAGE_CHARS]

    if status != "updated":
        return {"status": status, "assistantMessage": message, "recipe": None, "changes": None,
                "_model_nutrition": None}

    edits = out.get("edits")
    if not isinstance(edits, dict):
        raise ApiError(502, "model_error", "The model returned no edits.")

    # Start from the client's recipe so everything not edited survives untouched.
    recipe = copy.deepcopy(original_recipe)
    applied = 0
    for key in ("title", "description", "difficulty", "cuisine"):
        if _is_str(edits.get(key)) and edits[key].strip() and edits[key].strip() != recipe.get(key):
            recipe[key] = edits[key].strip()
            applied += 1
    for key in ("servings", "prepMinutes", "cookMinutes"):
        n = _num(edits.get(key), integer=True)
        if n is not None and n != recipe.get(key):
            recipe[key] = n
            applied += 1
    if isinstance(edits.get("dietaryTags"), list):
        recipe["dietaryTags"] = [t for t in edits["dietaryTags"] if _is_str(t) and t.strip()]
        applied += 1

    ingredients, n_ing = _apply_ops(recipe.get("ingredients") or [], edits.get("ingredientOps"), _clean_ingredient)
    steps, n_steps = _apply_ops(recipe.get("steps") or [], edits.get("stepOps"), _clean_step)
    applied += n_ing + n_steps
    if not ingredients or not steps:
        raise ApiError(502, "model_error", "The model removed every ingredient or step.")
    if applied == 0:
        raise ApiError(502, "model_error", "The model reported an update but changed nothing.")
    recipe["ingredients"] = ingredients[:MAX_INGREDIENTS]
    recipe["steps"] = [{**st, "order": i + 1} for i, st in enumerate(steps[:MAX_STEPS])]

    changes = []
    for ch in out.get("changes") or []:
        if not isinstance(ch, dict) or not _is_str(ch.get("title")) or not ch["title"].strip():
            continue
        kind = ch.get("kind") if ch.get("kind") in ALLOWED_CHANGE_KINDS else "general"
        changes.append({
            "kind": kind,
            "title": ch["title"].strip(),
            "detail": (ch.get("detail") or "").strip() if _is_str(ch.get("detail")) else "",
        })
    changes = changes[:5]
    if not changes:
        changes = [{"kind": "general", "title": "Updated recipe", "detail": "Applied your requested change"}]

    model_nutrition = edits.get("nutrition") if isinstance(edits.get("nutrition"), dict) else None
    return {"status": status, "assistantMessage": message, "recipe": recipe, "changes": changes,
            "_model_nutrition": model_nutrition}


def _final_nutrition(recomputed: dict | None, model_nutrition: dict | None, original: dict) -> dict:
    """Prefer the server recompute; fall back to Claude's estimate, then the original."""
    for candidate in (recomputed, model_nutrition):
        if not isinstance(candidate, dict):
            continue
        vals = {
            "calories": _num(candidate.get("calories"), integer=True),
            "protein": _num(candidate.get("protein")),
            "carbs": _num(candidate.get("carbs")),
            "fat": _num(candidate.get("fat")),
        }
        if all(v is not None for v in vals.values()) and vals["calories"] > 0:
            return {"basis": "per_serving", **vals}
    out = dict(original or {})
    out["basis"] = "per_serving"
    return out


# ── Blueprint ─────────────────────────────────────────────────────────────────

def create_customize_recipe_blueprint(
    *,
    recompute_nutrition: Callable[[dict], dict | None] | None = None,
    verify_token: Callable[[str], str] = _default_verify_token,
    get_db: Callable[[], Any] = _default_firestore,
    claude_call: Callable[..., Any] = _default_claude_call,
    is_premium: Callable[[Any, str], bool] = _is_premium,
    read_usage: Callable[[Any, str], int] = _read_usage_count,
    increment_usage: Callable[[Any, str, str], int] = _increment_usage,
    timeout_seconds: float | None = None,
) -> Blueprint:
    bp = Blueprint("customize_recipe", __name__)
    budget = CUSTOMIZE_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds

    # Steps 4-5 run here so we can enforce the 45 s budget. A thread that misses
    # the deadline finishes in the background; its result is dropped and not counted.
    executor = ThreadPoolExecutor(
        max_workers=_env_int("CUSTOMIZE_WORKERS", 8), thread_name_prefix="customize",
    )

    # Duplicate-retry handling, keyed by uid + X-Request-Id: a retry that arrives
    # while the first call is running waits for it; one that arrives afterwards
    # gets the same response replayed. Only final 200s are kept (errors are
    # retryable). Per-process; the Firestore counter is also idempotent per id.
    dedupe_ttl = _env_int("CUSTOMIZE_DEDUPE_TTL_SECONDS", 900)
    dedupe: dict[str, tuple[float, Future]] = {}
    dedupe_lock = threading.Lock()

    def _dedupe_claim(key: str) -> tuple[Future, bool]:
        now = time.monotonic()
        with dedupe_lock:
            for k in [k for k, (ts, f) in dedupe.items() if f.done() and now - ts > dedupe_ttl]:
                dedupe.pop(k, None)
            existing = dedupe.get(key)
            if existing:
                return existing[1], False
            fut: Future = Future()
            dedupe[key] = (now, fut)
            return fut, True

    def _dedupe_release(key: str, fut: Future, result: tuple[dict, int] | None) -> None:
        with dedupe_lock:
            if result is not None and result[1] == 200:
                dedupe[key] = (time.monotonic(), fut)
                fut.set_result(result)
            else:
                dedupe.pop(key, None)
                fut.set_result(None)  # waiters fall through and run their own attempt

    def _run_model_and_nutrition(body: dict, model: str, deadline: float) -> dict:
        t0 = time.monotonic()
        remaining = max(1.0, deadline - t0)
        resp = claude_call(
            model=model,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_content(body)}],
            output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
            max_tokens=CLAUDE_MAX_TOKENS,
            timeout=remaining,
        )
        usage = getattr(resp, "usage", None)
        tokens = {
            "input": getattr(usage, "input_tokens", None),
            "output": getattr(usage, "output_tokens", None),
        }
        if getattr(resp, "stop_reason", None) == "max_tokens":
            raise ApiError(502, "model_error", "The model response was cut off.")
        claude_ms = int((time.monotonic() - t0) * 1000)
        result = _normalize_model_output(_tool_input_from_response(resp), body["recipe"])
        nutrition_ms = 0

        if result["status"] == "updated":
            recomputed = None
            t1 = time.monotonic()
            if recompute_nutrition is not None:
                try:
                    recomputed = recompute_nutrition(result["recipe"])
                except Exception as e:  # nutrition is best-effort; fall back below
                    print(f"[customize-recipe] nutrition recompute failed: {e}")
            result["recipe"]["nutrition"] = _final_nutrition(
                recomputed, result["_model_nutrition"], body["recipe"].get("nutrition"),
            )
            nutrition_ms = int((time.monotonic() - t1) * 1000)
            result["_nutrition_source"] = "recomputed" if recomputed else "model_or_original"
        result.pop("_model_nutrition", None)
        result["_tokens"] = tokens
        result["_timings"] = {"claudeMs": claude_ms, "nutritionMs": nutrition_ms}
        return result

    @bp.route("/customize-recipe", methods=["POST"])
    def customize_recipe():
        started = time.monotonic()
        deadline = started + budget
        request_id = (request.headers.get("X-Request-Id") or "").strip() or str(uuid.uuid4())
        log: dict[str, Any] = {
            "event": "customize_recipe", "requestId": request_id, "uid": None,
            "client": request.headers.get("X-Client"), "model": None,
            "inputTokens": None, "outputTokens": None,
        }

        def respond(payload: dict, http: int):
            log["http"] = http
            log["latencyMs"] = int((time.monotonic() - started) * 1000)
            print("[customize-recipe] " + json.dumps(log, ensure_ascii=False))
            return jsonify(payload), http

        dedupe_key = None
        dedupe_fut = None
        final: tuple[dict, int] | None = None
        try:
            # 1. Authenticate
            auth_header = request.headers.get("Authorization") or ""
            if not auth_header.lower().startswith("bearer ") or not auth_header[7:].strip():
                raise ApiError(401, "unauthenticated", "Missing Firebase ID token.")
            t_auth = time.monotonic()
            try:
                uid = verify_token(auth_header[7:].strip())
            except Exception as e:
                print(f"[customize-recipe] token verification failed: {type(e).__name__}: {e}")
                raise ApiError(401, "unauthenticated", "Your session has expired. Please sign in again.")
            if not uid:
                raise ApiError(401, "unauthenticated", "Invalid Firebase ID token.")
            log["uid"] = uid
            timings: dict[str, int] = {"authMs": int((time.monotonic() - t_auth) * 1000)}
            log["timings"] = timings

            # 2. Validate
            if request.content_length is not None and request.content_length > MAX_BODY_BYTES:
                raise ApiError(400, "invalid_request", "Request body exceeds 64 KB.")
            _validate_headers(request.headers)
            body = _validate_body(request.get_data(cache=False) or b"")
            log.update({"mode": body["mode"], "presetId": body.get("presetId"),
                        "source": body["source"], "recipeId": body.get("recipeId")})

            # Duplicate retry? Wait for / replay the original.
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
                fut, owner = _dedupe_claim(dedupe_key)  # original failed; take over
                if not owner:
                    raise ApiError(504, "timeout", "This is taking longer than expected. Please try again.")
            dedupe_fut = fut

            # 3. Check the limit
            t_fs = time.monotonic()
            db = get_db()
            premium = bool(is_premium(db, uid))
            limit = PREMIUM_DAILY_LIMIT if premium else FREE_DAILY_LIMIT
            log["premium"] = premium
            if limit <= 0:
                raise ApiError(403, "premium_required", "Recipe customization is a premium feature.")
            used = int(read_usage(db, uid))
            timings["firestoreReadMs"] = int((time.monotonic() - t_fs) * 1000)
            if used >= limit:
                msg = (f"You've used today's {limit} free customizations." if not premium
                       else f"You've used today's {limit} customizations. Come back tomorrow!")
                raise ApiError(429, "limit_reached", msg)

            # 4 + 5. Call Claude, then recompute nutrition (within the 45 s budget)
            model = _pick_model(body["mode"])
            log["model"] = model
            job = executor.submit(_run_model_and_nutrition, body, model, deadline)
            try:
                result = job.result(timeout=max(0.0, deadline - time.monotonic()))
            except FutureTimeoutError:
                raise ApiError(504, "timeout", "This is taking longer than expected. Please try again.")
            except ApiError:
                raise
            except Exception as e:
                if _is_timeout_exception(e):
                    raise ApiError(504, "timeout", "This is taking longer than expected. Please try again.")
                print(f"[customize-recipe] Claude call failed: {type(e).__name__}: {e}")
                raise ApiError(502, "model_error", "We couldn't customize this recipe. Please try again.")
            tokens = result.pop("_tokens", {}) or {}
            timings.update(result.pop("_timings", {}) or {})
            if "_nutrition_source" in result:
                log["nutritionSource"] = result.pop("_nutrition_source")
            log["inputTokens"] = tokens.get("input")
            log["outputTokens"] = tokens.get("output")
            log["status"] = result["status"]

            # 6. Count the use (only when updated)
            if result["status"] == "updated":
                t_inc = time.monotonic()
                try:
                    used = int(increment_usage(db, uid, request_id))
                except Exception as e:
                    # The user already paid the latency; don't fail the response.
                    print(f"[customize-recipe] usage increment failed: {type(e).__name__}: {e}")
                    used += 1
                timings["usageWriteMs"] = int((time.monotonic() - t_inc) * 1000)

            # 7. Log and return
            payload = {
                "requestId": request_id,
                "status": result["status"],
                "assistantMessage": result["assistantMessage"],
                "recipe": result["recipe"],
                "changes": result["changes"],
                "usage": {"usedToday": used, "limitToday": limit},
            }
            final = (payload, 200)
            return respond(payload, 200)

        except ApiError as err:
            log["error"] = err.code
            return respond(_error_body(request_id, err), err.http)
        except Exception as e:
            log["error"] = "internal_error"
            print(f"[customize-recipe] unexpected error: {type(e).__name__}: {e}")
            return respond(_error_body(request_id, ApiError(
                500, "internal_error", "We ran into a problem. Please try again in a moment.")), 500)
        finally:
            if dedupe_key is not None and dedupe_fut is not None:
                _dedupe_release(dedupe_key, dedupe_fut, final)

    return bp


def _is_timeout_exception(e: Exception) -> bool:
    name = type(e).__name__.lower()
    return "timeout" in name or isinstance(e, TimeoutError)
