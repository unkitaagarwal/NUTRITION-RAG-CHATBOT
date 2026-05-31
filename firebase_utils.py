import firebase_admin
from firebase_admin import credentials, firestore, storage
from datetime import datetime
import json
import os
import re
import tempfile
import threading
import uuid

import requests

# ── NutriLens app (nutrilensai-77be2) ─────────────────────────────────────────
# Used by: /recommend-meals, /food-logging, and all meal image storage.
# Env vars: FIREBASE_SERVICE_ACCOUNT_JSON, FIREBASE_PROJECT_ID,
#           FIREBASE_STORAGE_BUCKET
# ── MealMap app ───────────────────────────────────────────────────────────────
# Used by: /extract-recipe, /bulk-import-recipes (recipe-log collection).
# Env vars: MEALMAP_SERVICE_ACCOUNT_JSON, MEALMAP_PROJECT_ID,
#           MEALMAP_STORAGE_BUCKET
# ─────────────────────────────────────────────────────────────────────────────
# Both env vars accept EITHER a file path OR raw JSON content (for Render /
# cloud deployments where you paste the service-account JSON directly into the
# env var value).

_NUTRILENS_APP_NAME = firebase_admin.DEFAULT_APP_NAME   # "[DEFAULT]"
_MEALMAP_APP_NAME   = "mealmap"

_firestore_client = None
_firestore_lock = threading.Lock()

_mealmap_firestore_client = None
_mealmap_lock = threading.Lock()

_NUTRILENS_PROJECT_ID = "nutrilensai-77be2"
_NUTRILENS_CRED_CANDIDATES = (
    "nutrilens_firebase_service_account.json",
    "firebase_service_account.json",
)
_MEALMAP_CRED_CANDIDATES = (
    "mealmap_firebase_service_account.json",
    "firebase_service_account.json",
)
_MEAL_IMAGE_MAX_BYTES = int(os.getenv("MEAL_IMAGE_MAX_DOWNLOAD_BYTES", str(12 * 1024 * 1024)))


def _load_credentials(env_var: str, file_candidates: tuple) -> credentials.Certificate:
    """
    Build a firebase_admin Certificate from either:
      1. env_var containing raw JSON  (paste JSON directly → works on Render)
      2. env_var containing a file path
      3. first matching file from file_candidates
    """
    raw = (os.getenv(env_var) or "").strip()

    # Case 1: env var holds JSON content directly
    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
            # Write to a temp file because Certificate() needs a file or dict
            return credentials.Certificate(parsed)
        except Exception as e:
            raise ValueError(f"[firebase] {env_var} looks like JSON but failed to parse: {e}")

    # Case 2: env var holds a file path
    if raw and os.path.isfile(raw):
        return credentials.Certificate(raw)

    # Case 3: search candidate file names
    for name in file_candidates:
        if os.path.isfile(name):
            return credentials.Certificate(name)

    raise FileNotFoundError(
        f"Firebase credentials not found. Set {env_var} to the service-account "
        f"JSON content (on Render: paste the full JSON) or a valid file path. "
        f"Tried candidates: {file_candidates}"
    )


def _project_id_from_env_or_cred(env_var: str, cred: credentials.Certificate) -> str:
    explicit = (os.getenv(env_var) or "").strip()
    if explicit:
        return explicit
    try:
        return (cred.service_account_email or "").split("@")[1].split(".")[0]
    except Exception:
        pass
    return ""


def _sanitize_storage_segment(value: str, *, max_len: int = 180) -> str:
    cleaned = re.sub(r"[^\w\-.]", "_", (value or "").strip())
    return cleaned[:max_len] or "unknown"


# ── NutriLens (default) app ───────────────────────────────────────────────────

def init_firestore():
    """Return Firestore client for the NutriLens project (nutrilensai-77be2)."""
    global _firestore_client
    if _firestore_client:
        return _firestore_client

    with _firestore_lock:
        if _firestore_client:
            return _firestore_client

        if firebase_admin.DEFAULT_APP_NAME not in firebase_admin._apps:
            cred = _load_credentials("FIREBASE_SERVICE_ACCOUNT_JSON", _NUTRILENS_CRED_CANDIDATES)
            bucket_name = (os.getenv("FIREBASE_STORAGE_BUCKET") or "").strip()
            if not bucket_name:
                pid = (os.getenv("FIREBASE_PROJECT_ID") or _NUTRILENS_PROJECT_ID).strip()
                bucket_name = f"{pid}.appspot.com"
            print(f"[firebase/nutrilens] initializing default app, bucket={bucket_name}")
            firebase_admin.initialize_app(cred, {"storageBucket": bucket_name})

        _firestore_client = firestore.client()
    return _firestore_client


# ── MealMap app ───────────────────────────────────────────────────────────────

def init_mealmap_firestore():
    """Return Firestore client for the MealMap project (extract-recipe / recipe-log)."""
    global _mealmap_firestore_client
    if _mealmap_firestore_client:
        return _mealmap_firestore_client

    with _mealmap_lock:
        if _mealmap_firestore_client:
            return _mealmap_firestore_client

        if _MEALMAP_APP_NAME not in firebase_admin._apps:
            cred = _load_credentials("MEALMAP_SERVICE_ACCOUNT_JSON", _MEALMAP_CRED_CANDIDATES)
            bucket_name = (os.getenv("MEALMAP_STORAGE_BUCKET") or "").strip()
            if not bucket_name:
                try:
                    pid = (os.getenv("MEALMAP_PROJECT_ID") or "").strip()
                    if not pid:
                        # derive from service account email: name@project-id.iam...
                        pid = cred.service_account_email.split("@")[1].split(".iam")[0]
                    bucket_name = f"{pid}.appspot.com"
                except Exception:
                    bucket_name = ""
            opts = {"storageBucket": bucket_name} if bucket_name else {}
            print(f"[firebase/mealmap] initializing named app '{_MEALMAP_APP_NAME}', bucket={bucket_name or '(none)'}")
            firebase_admin.initialize_app(cred, opts, name=_MEALMAP_APP_NAME)

        mealmap_app = firebase_admin.get_app(_MEALMAP_APP_NAME)
        _mealmap_firestore_client = firestore.client(app=mealmap_app)
    return _mealmap_firestore_client


def recommend_meal_image_storage_path(plan_id: str, day_index: int, meal_key: str) -> str:
    """gs://nutrilensai-77be2.appspot.com/generated-meals/{plan_id}/day-{n}/{meal}.png"""
    prefix = (os.getenv("FIREBASE_MEAL_IMAGE_PREFIX") or "generated-meals").strip().strip("/")
    safe_plan = _sanitize_storage_segment(plan_id)
    safe_meal = _sanitize_storage_segment(meal_key, max_len=80)
    return f"{prefix}/{safe_plan}/day-{day_index}/{safe_meal}.png"


def upload_meal_image_bytes_to_storage(
    data: bytes,
    content_type: str = "image/png",
    *,
    storage_path: str | None = None,
) -> str | None:
    """
    Upload raw image bytes to Firebase Storage and return a public HTTPS URL, or None on failure/skip.
    """
    if (os.getenv("DISABLE_MEAL_IMAGE_PERSISTENCE") or "").strip().lower() in ("1", "true", "yes"):
        return None
    if not data or not isinstance(data, (bytes, bytearray)):
        return None
    data = bytes(data)
    if len(data) > _MEAL_IMAGE_MAX_BYTES:
        print("[meal-image] persist skipped: image too large")
        return None

    ct = (content_type or "image/png").split(";")[0].strip()
    ext = ".png"
    if "jpeg" in ct or "jpg" in ct:
        ext = ".jpg"
    elif "webp" in ct:
        ext = ".webp"

    init_firestore()
    try:
        bucket = storage.bucket()
    except Exception as e:
        print(f"[meal-image] storage bucket unavailable: {e}")
        return None

    object_name = storage_path or f"generated-meals/{uuid.uuid4().hex}{ext}"
    blob = bucket.blob(object_name)
    try:
        blob.upload_from_string(data, content_type=ct)
        print(f"[meal-image] uploaded to Firebase Storage: {object_name}")
    except Exception as e:
        print(f"[meal-image] upload failed: {e}")
        return None

    # Always return the canonical firebasestorage.googleapis.com URL.
    # blob.public_url returns a storage.googleapis.com URL which is a different domain
    # and won't work with Flutter's Firebase SDK or cached network image widgets.
    # make_public() is attempted as a best-effort ACL grant (no-op if uniform bucket-level
    # access is enabled — Storage Rules handle public access in that case).
    try:
        blob.make_public()
    except Exception as e:
        print(f"[meal-image] make_public skipped (ok if Storage Rules grant public read): {e}")

    bucket_name = bucket.name
    encoded_name = object_name.replace("/", "%2F")
    firebase_url = (
        f"https://firebasestorage.googleapis.com/v0/b/{bucket_name}/o/{encoded_name}?alt=media"
    )
    print(f"[meal-image] returning Firebase URL: {firebase_url}")
    return firebase_url


def save_recommend_meal_image_record(
    plan_id: str,
    day_index: int,
    meal_key: str,
    image_url: str,
) -> None:
    """
    Store a permanent meal image URL for /recommend-meals so clients (e.g. Flutter) can sync from Firestore.
    Document id: {plan_id}_d{day_index}_{meal_key}
    """
    if not plan_id or not image_url:
        return
    try:
        db = init_firestore()
        doc_id = f"{plan_id}_d{day_index}_{meal_key}"
        db.collection("recommend_meal_images").document(doc_id).set(
            {
                "plan_id": plan_id,
                "day_index": day_index,
                "meal_key": meal_key,
                "image_url": image_url,
                "updated_at": datetime.utcnow(),
            },
            merge=True,
        )
    except Exception as e:
        print(f"[recommend-meal-image] firestore write failed: {e}")


def _trusted_openai_generated_image_url(url: str) -> bool:
    """Only fetch URLs we expect from OpenAI image APIs (SSRF-safe)."""
    from urllib.parse import urlparse

    u = urlparse(url)
    if u.scheme != "https" or not u.hostname:
        return False
    h = u.hostname.lower()
    return h.endswith(".blob.core.windows.net")


def try_persist_meal_image_from_openai_url(url: str) -> str | None:
    """
    Download a short-lived OpenAI/DALL·E image URL and upload to Firebase Storage.
    Returns a stable HTTPS URL (GCS public URL), or None on skip/failure.
    """
    if (os.getenv("DISABLE_MEAL_IMAGE_PERSISTENCE") or "").strip().lower() in ("1", "true", "yes"):
        return None
    if not url or not isinstance(url, str) or not _trusted_openai_generated_image_url(url):
        return None
    try:
        r = requests.get(url, timeout=45, stream=True)
        r.raise_for_status()
        ct = (r.headers.get("Content-Type") or "image/png").split(";")[0].strip()
        buf = bytearray()
        for chunk in r.iter_content(65536):
            if not chunk:
                continue
            buf.extend(chunk)
            if len(buf) > _MEAL_IMAGE_MAX_BYTES:
                print("[meal-image] persist skipped: image too large")
                return None
        data = bytes(buf)
    except Exception as e:
        print(f"[meal-image] download failed: {e}")
        return None

    return upload_meal_image_bytes_to_storage(data, ct)


def get_user_context(email, max_logs=5):
    db = init_firestore()
    
    # Load meal logs directly from log_entry collection using user_email
    # Temporarily removed order_by to avoid needing composite index
    logs = db.collection("log_entry").where("user_email", "==", email).limit(max_logs).stream()
    
    meal_history = []
    snacks_only = []
    breakfast_only = []
    lunch_only = []
    dinner_only = []
    
    for log in logs:
        log_data = log.to_dict()
        
        # Debug: Print what we're getting from Firebase
        print(f"DEBUG: Retrieved log data: {log_data}")
        
        # Handle item_name - it might be a string or array
        item_name = log_data.get('item_name', '')
        if isinstance(item_name, list):
            items_str = ', '.join(item_name)
        else:
            items_str = str(item_name) if item_name else 'Unknown item'
        
        meal_type = log_data.get('meal_type', 'Unknown meal')
        date_time = log_data.get('date_time')
        date_str = date_time.date() if date_time else 'Unknown date'
        calories = log_data.get('total_calories', 0)
        carbs = log_data.get('total_carbs', 0)
        protein = log_data.get('total_protein', 0)
        fat = log_data.get('total_fat', 0)
        
        # Enhanced meal entry with nutritional info
        meal_entry = f"{meal_type} on {date_str}: {items_str} - {calories} kcal (Carbs: {carbs}g, Protein: {protein}g, Fat: {fat}g)"
        meal_history.append(meal_entry)
        
        # Categorize by meal type for filtered responses
        if meal_type.lower() == 'snacks':
            snacks_only.append(meal_entry)
        elif meal_type.lower() == 'breakfast':
            breakfast_only.append(meal_entry)
        elif meal_type.lower() == 'lunch':
            lunch_only.append(meal_entry)
        elif meal_type.lower() == 'dinner':
            dinner_only.append(meal_entry)
    
    print(f"DEBUG: Final meal_history: {meal_history}")
    print(f"DEBUG: Snacks only: {snacks_only}")

    # Fetch user preferences/goals (stored in subcollection user_preferences)
    user_preferences = {}
    try:
        # Find the user document by email in nutrilensai-77be2 collection
        user_doc_ref = None
        user_query = (
            db.collection("nutrilensai-77be2")
            .where("email", "==", email)
            .limit(1)
            .stream()
        )
        for user_doc in user_query:
            user_doc_ref = user_doc.reference
            print(f"DEBUG: Found user document for {email} with ID: {user_doc_ref.id}")
            break

        if user_doc_ref:
            # Access user_preferences subcollection under the user document
            # Get the first document (assuming one document per user, or get any if multiple exist)
            prefs_query = (
                user_doc_ref.collection("user_preferences")
                .limit(1)
                .stream()
            )
            
            pref_count = 0
            for pref_doc in prefs_query:
                user_preferences = pref_doc.to_dict() or {}
                pref_count += 1
                print(f"DEBUG: Found user_preferences document with ID: {pref_doc.id}")
                print(f"DEBUG: User preferences keys: {list(user_preferences.keys())}")
                break
            
            if pref_count == 0:
                print(f"DEBUG: No user_preferences subcollection found under user document {user_doc_ref.id}")
                # List all subcollections to help debug
                print(f"DEBUG: Checking if user_preferences subcollection exists...")
        else:
            print(f"DEBUG: No user document found for email: {email} in nutrilensai-77be2 collection")

        print(f"DEBUG: User preferences: {user_preferences}")
    except Exception as e:
        print(f"DEBUG: Error fetching user preferences for {email}: {e}")
        import traceback
        traceback.print_exc()
        user_preferences = {}
    
    # goal kept for backwards compatibility (now mirrors preferences)
    goal = user_preferences.copy()
    
    # Return both all meals and categorized meals
    categorized_meals = {
        'all': meal_history,
        'snacks': snacks_only,
        'breakfast': breakfast_only,
        'lunch': lunch_only,
        'dinner': dinner_only
    }
    
    return goal, categorized_meals, user_preferences

def save_user_chat(email, question, answer):
    db = init_firestore()
    
    # Ensure the user document exists first
    user_ref = db.collection("users").document(email)
    user_doc = user_ref.get()
    
    if not user_doc.exists:
        # Create user document if it doesn't exist
        user_ref.set({
            "email": email,
            "created_at": datetime.utcnow()
        })
        print(f"DEBUG: Created new user document for {email}")
    
    # Now add the chat to the chats subcollection
    chat_ref = user_ref.collection("chats")
    chat_doc = chat_ref.add({
        "question": question,
        "answer": answer,
        "timestamp": datetime.utcnow()
    })
    
    print(f"DEBUG: Successfully saved chat for {email} with ID: {chat_doc[1].id}")

def get_user_chat_history(email, max_chats=2):  # Reduced from 3 to 2 for faster queries
    db = init_firestore()
    print(f"DEBUG: Looking for chat history for email: {email}")
    
    chat_ref = db.collection("users").document(email).collection("chats")
    print(f"DEBUG: Chat reference path: users/{email}/chats")
    
    docs = chat_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(max_chats).stream()
    
    history = []
    for doc in docs:
        data = doc.to_dict()
        question = data.get("question", "")
        answer = data.get("answer", "")
        history.append((question, answer))
        print(f"DEBUG: Found chat entry - Q: {question[:50]}..., A: {answer[:50]}...")
    
    print(f"DEBUG: Total chat history entries found: {len(history)}")
    
    # Reverse to maintain chronological order (oldest → newest)
    return history[::-1]

