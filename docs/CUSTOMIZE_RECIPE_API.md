# `/customize-recipe` API — implementation notes

Spec: "Customize Recipe API" (Oct 4, 2026). Code: `customize_recipe.py` (Flask blueprint),
registered at the bottom of `app.py`, which injects the nutrition step `/extract-recipe` uses
(`_customize_recompute_nutrition` → `_ensure_recipe_nutrition_macros`).

- `POST /customize-recipe`, JSON in/out, Firebase ID token required (verified against the
  MealMap/RecipeVault Firebase app; override with `CUSTOMIZE_FIREBASE_APP`).
- Headers: `Content-Type: application/json`, `Authorization: Bearer <token>`,
  `X-Client` (`ios-app` | `android-app` | `ios-share-extension`), `X-Request-Id` (UUID).
- Claude, structured outputs (`output_config.format` = JSON schema; Sonnet 5.5 rejects forced
  `tool_choice`). Default model is Haiku 4.5 (`claude-haiku-4-5-20251001`) for both preset chips
  and chat; set `CUSTOMIZE_CLAUDE_MODEL=claude-sonnet-5-5` to send chat to Sonnet.
- Claude returns **only the edits**, not the whole recipe: ingredients and steps are sent numbered
  (`n`), and Claude replies with `ingredientOps` / `stepOps` (`update` / `remove` / `add` after `n`,
  all against the original numbering) plus any changed top-level fields. The server merges them
  into the client's recipe, renumbers steps and returns the full recipe as before — the
  response shape for the app is unchanged.
- Steps 4–5 (Claude + nutrition) must finish in 45 s, else `504 timeout`.
- Usage counter: `meal_plan_chef_users/{uid}/usage/customize_{yyyy-mm-dd}` (`count`,
  `requestIds`), incremented only for `status = updated`, once per `X-Request-Id`.
  Date uses `CUSTOMIZE_USAGE_TZ` (default UTC). Free 3/day, premium 50/day;
  `CUSTOMIZE_FREE_DAILY_LIMIT=0` makes the feature premium-only (`403 premium_required`).
- Premium = `is_premium_user == true` on `meal_plan_chef_users/{uid}` only (no household logic).
- Duplicate retries (same uid + `X-Request-Id`): an in-flight retry waits for the original;
  a later one gets the same 200 replayed (per worker, 15 min). Errors are not cached.
- Nutrition: recomputed from the new ingredient list; falls back to Claude's estimate, then
  the original numbers, if the estimator fails.
- Errors: `{"requestId", "error": {"code", "message"}}` — 400 `invalid_request`,
  401 `unauthenticated`, 403 `premium_required`, 429 `limit_reached`, 502 `model_error`,
  504 `timeout` (plus 500 `internal_error` for unexpected failures, e.g. Firestore down).
- Each request logs one `[customize-recipe] {...}` JSON line: uid, requestId, model,
  input/output tokens, latency, status/error.

Env: `ANTHROPIC_API_KEY` (required), see `.env.example` for the optional `CUSTOMIZE_*` knobs.
Tests: `pytest tests/test_customize_recipe.py` (mocks Firebase/Firestore/Claude).
