"""Two-tier cache for /extract-recipe results.

L1: per-worker in-memory dict. Hot path, zero latency, dies with the worker
    (deploys, gunicorn max_requests recycling) — that's fine, L2 survives.
L2: Firestore collection (default: "recipe_cache"), doc ID = SHA256(url).
    Survives deploys and is shared across all workers/instances.

Read:  L1 hit -> return. L1 miss -> L2 lookup -> backfill L1 -> return.
Write: L1 synchronously, L2 fire-and-forget on a background thread so it
       never adds latency or failure risk to the response.

Invalidation: RECIPE_CACHE_SCHEMA_VERSION is stamped on every L2 doc.
Bump it (same PR) whenever extraction/nutrition output changes; mismatched
docs are treated as misses and lazily overwritten. Optionally configure a
Firestore TTL policy on the `expires_at` field to auto-prune old docs.

The recipe payload is stored as a JSON string (`payload` field) rather than
a Firestore map, sidestepping Firestore type restrictions (nested arrays,
non-string keys, etc.).

Env:
  RECIPE_CACHE_COLLECTION   Firestore collection name (default "recipe_cache")
  RECIPE_CACHE_DISABLE_L2   "1"/"true" = emergency kill switch, L1-only
  RECIPE_CACHE_TTL_DAYS     expires_at horizon (default 90)
"""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

# Bump when extraction/nutrition logic changes what results look like,
# so stale pre-deploy cache entries are re-extracted instead of served.
RECIPE_CACHE_SCHEMA_VERSION = 1

_COLLECTION_NAME = os.getenv("RECIPE_CACHE_COLLECTION", "recipe_cache")
_L2_DISABLED = str(os.getenv("RECIPE_CACHE_DISABLE_L2", "")).strip().lower() in ("1", "true", "yes")
_TTL_DAYS = int(os.getenv("RECIPE_CACHE_TTL_DAYS", "90"))
# Firestore documents cap at ~1 MiB; leave headroom for metadata fields.
_MAX_PAYLOAD_BYTES = 900_000


class RecipeCache:
    """Thread-safe two-tier (in-memory + Firestore) cache. Best-effort L2:
    any Firestore error degrades to L1-only behavior, never breaks a request."""

    def __init__(self):
        self._l1: dict = {}
        self._lock = threading.Lock()
        self._l2_writer = ThreadPoolExecutor(max_workers=2, thread_name_prefix="recipe-cache-l2")

    # ── internal ──────────────────────────────────────────────────────────
    def _collection(self):
        # Lazy import/init: Firebase clients must not be created at module
        # import time (see gunicorn.conf.py note on preload_app/fork safety).
        # MealMap project: it owns all extract-recipe data (see firebase_utils).
        from firebase_utils import init_mealmap_firestore
        return init_mealmap_firestore().collection(_COLLECTION_NAME)

    def _l2_get(self, key: str) -> dict | None:
        try:
            doc = self._collection().document(key).get()
        except Exception as e:
            print(f"⚠️ recipe_cache L2 read failed (serving as miss): {e}")
            return None
        if not doc.exists:
            return None
        data = doc.to_dict() or {}
        if data.get("schema_version") != RECIPE_CACHE_SCHEMA_VERSION:
            return None  # produced by older code — re-extract and overwrite
        expires_at = data.get("expires_at")
        if expires_at is not None:
            try:
                if expires_at < datetime.now(timezone.utc):
                    return None
            except TypeError:
                pass
        payload = data.get("payload")
        if not isinstance(payload, str):
            return None
        try:
            result = json.loads(payload)
        except (ValueError, TypeError):
            return None
        return result if isinstance(result, dict) else None

    def _l2_set(self, key: str, result: dict) -> None:
        try:
            payload = json.dumps(result, ensure_ascii=False, default=str)
            if len(payload.encode("utf-8")) > _MAX_PAYLOAD_BYTES:
                print(f"⚠️ recipe_cache L2 write skipped (payload too large): {key}")
                return
            now = datetime.now(timezone.utc)
            self._collection().document(key).set({
                "payload": payload,
                "schema_version": RECIPE_CACHE_SCHEMA_VERSION,
                "created_at": now,
                "expires_at": now + timedelta(days=_TTL_DAYS),
            })
        except Exception as e:
            print(f"⚠️ recipe_cache L2 write failed (ignored): {e}")

    # ── public API ────────────────────────────────────────────────────────
    def get(self, key: str) -> dict | None:
        """Return a shallow copy of the cached result, or None on miss."""
        with self._lock:
            hit = self._l1.get(key)
        if hit is not None:
            return dict(hit)
        if _L2_DISABLED:
            return None
        result = self._l2_get(key)
        if result is None:
            return None
        with self._lock:
            self._l1[key] = result  # backfill hot layer
        return dict(result)

    def set(self, key: str, result: dict) -> None:
        """Write-through: L1 now, L2 asynchronously (fire-and-forget)."""
        with self._lock:
            self._l1[key] = result
        if not _L2_DISABLED:
            self._l2_writer.submit(self._l2_set, key, result)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None
