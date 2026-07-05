from flask import Flask, request, jsonify, send_file
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from firebase_utils import (
    init_mealmap_firestore,
    get_user_context,
    get_user_chat_history,
    save_user_chat,
    init_firestore,
    try_persist_meal_image_from_openai_url,
    upload_meal_image_bytes_to_storage,
    save_recommend_meal_image_record,
    recommend_meal_image_storage_path,
    persist_extract_recipe_image,
)
from dotenv import load_dotenv
import os
import csv
import io
import json
import re
import base64
import tempfile
from datetime import datetime, timedelta
import uuid
from openai import OpenAI
from threading import Thread, Lock
from collections import defaultdict
import yt_dlp
import instaloader
import shutil
import time
import socket
import ipaddress
from urllib.parse import urlparse, parse_qs, urlunparse
import http.cookiejar
import requests
from bs4 import BeautifulSoup
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib


# Load environment variables first
load_dotenv()

# Initialize OpenAI client for voice functionality
client = OpenAI()
app = Flask(__name__)

# In-memory de-dup memory for /recommend-meals/day calls per plan_id.
# This keeps day plans distinct for sequential calls using the same plan_id.
_plan_meal_name_history = defaultdict(set)
_plan_meal_history_lock = Lock()

# In-memory cache for /extract-recipe-from-video results, keyed by SHA256 of the video URL.
# Same video URL always produces the same recipe, so we never re-run the full pipeline.
_recipe_cache: dict = {}
_recipe_cache_lock = Lock()
_lang_detect_cache: dict[str, str] = {}

# Bounded pool + registry for uploading the recipe thumbnail to Storage
# concurrently with recipe extraction (the upload is kicked off as soon as the
# thumbnail URL is known and resolved just before the response is built, so it
# overlaps the LLM call instead of adding to it). Keyed by url_key.
_image_persist_executor = ThreadPoolExecutor(
    max_workers=int(os.getenv("EXTRACT_IMAGE_PERSIST_WORKERS", "4")),
    thread_name_prefix="persist-img",
)
_img_persist_futures: dict = {}

# Initialize once
# ---------- Config ----------
MAX_VIDEO_SECONDS = int(os.getenv("MAX_VIDEO_SECONDS", "900"))  # 15 min default
# Cookies can be provided as: 1) file path, or 2) base64-encoded content in YTDLP_COOKIES_B64 env var
YTDLP_COOKIES_FILE = os.path.expanduser(os.path.expandvars(os.getenv("YTDLP_COOKIES_FILE", ""))) or None  # optional, helps IG/TikTok/Facebook
YTDLP_COOKIES_B64 = os.getenv("YTDLP_COOKIES_B64")  # alternative: base64-encoded cookies content (for Render/cloud)

# Startup diagnostic: confirm whether yt-dlp cookies are actually available on this instance.
# Facebook/IG/TikTok return "302 redirect loop" when cookies are missing/expired (bounced to login).
try:
    _cookie_status = "NONE"
    if YTDLP_COOKIES_FILE and os.path.exists(YTDLP_COOKIES_FILE):
        _cookie_status = f"FILE ({YTDLP_COOKIES_FILE}, {os.path.getsize(YTDLP_COOKIES_FILE)} bytes)"
    elif YTDLP_COOKIES_FILE:
        _cookie_status = f"FILE SET BUT MISSING ON DISK ({YTDLP_COOKIES_FILE})"
    elif YTDLP_COOKIES_B64:
        _cookie_status = "B64"
    print(f"[startup] yt-dlp cookies: {_cookie_status}")
except Exception as _e:
    print(f"[startup] cookie check failed: {_e}")


def _prepare_cookiefile(temp_dir: str | None = None, video_url: str | None = None) -> str | None:
    """Return a WRITABLE cookies.txt path for yt-dlp, or None if no cookies configured.

    NEVER passes cookies for YouTube URLs: when yt-dlp is given cookies, it
    SKIPS player clients that don't support cookie auth (incl. android_vr) and
    falls back to the web client, which triggers "Sign in to confirm you're
    not a bot" from datacenter IPs. Cookie-free android_vr is what makes
    proxy-free YouTube extraction work, so cookies stay IG/TikTok/FB-only.

    IMPORTANT: yt-dlp writes the (refreshed) cookie jar BACK to `cookiefile`
    after a download. Render secret files are mounted read-only at /etc/secrets,
    so pointing yt-dlp directly at YTDLP_COOKIES_FILE raises
    "[Errno 30] Read-only file system". We therefore always copy the cookies
    into a writable location and hand yt-dlp the copy.

    Pass `temp_dir` (the per-download temp dir) when available so the copy is
    cleaned up automatically; otherwise a standalone temp file is created.
    """
    if not (YTDLP_COOKIES_FILE or YTDLP_COOKIES_B64):
        return None
    if video_url and is_youtube_url(video_url):
        return None
    try:
        if temp_dir:
            dest = os.path.join(temp_dir, "cookies.txt")
        else:
            fd, dest = tempfile.mkstemp(suffix="_cookies.txt")
            os.close(fd)
        if YTDLP_COOKIES_FILE and os.path.exists(YTDLP_COOKIES_FILE):
            shutil.copyfile(YTDLP_COOKIES_FILE, dest)
            return dest
        if YTDLP_COOKIES_B64:
            with open(dest, "w") as f:
                f.write(base64.b64decode(YTDLP_COOKIES_B64).decode("utf-8"))
            return dest
    except Exception as e:
        print(f"[cookies] failed to prepare writable cookie file: {e}")
    return None


# Hosts whose share/short links must be resolved to a canonical URL before yt-dlp.
# yt-dlp's generic extractor loops on fb.watch redirects ("302 redirect loop"),
# but the canonical facebook.com/watch?v=... / /reel/<id>/ URL extracts fine.
_SHARE_LINK_HOSTS = {"fb.watch", "www.fb.watch", "fb.com", "www.fb.com"}


def _resolve_share_url(url: str) -> str:
    """Resolve a short/share link (e.g. fb.watch) to its canonical URL.

    Returns the original URL unchanged on any failure or for non-share hosts.
    """
    try:
        host = (urlparse(url).hostname or "").lower()
        if host not in _SHARE_LINK_HOSTS:
            return url

        sess = requests.Session()
        # Attach cookies so Facebook resolves the video instead of bouncing to
        # a login/consent interstitial (which is what causes the redirect loop).
        cf = _prepare_cookiefile()
        if cf:
            try:
                cj = http.cookiejar.MozillaCookieJar(cf)
                cj.load(ignore_discard=True, ignore_expires=True)
                sess.cookies = cj
            except Exception:
                pass

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        proxies = None
        if SOCIAL_PROXY:
            proxies = {"http": SOCIAL_PROXY, "https": SOCIAL_PROXY}
        resp = sess.get(url, headers=headers, allow_redirects=True, timeout=15, proxies=proxies)
        final = resp.url or url
        p = urlparse(final)
        fhost = (p.hostname or "").lower()
        if "facebook.com" not in fhost:
            # Landed somewhere unexpected (e.g. login). Keep original for yt-dlp.
            return url

        # Build a clean canonical URL, dropping tracking params (mibextid, etc.)
        qs = parse_qs(p.query)
        vid = (qs.get("v") or [None])[0]
        if vid:
            return f"https://www.facebook.com/watch/?v={vid}"
        return f"https://www.facebook.com{p.path}"
    except Exception as e:
        print(f"[resolve] share-url resolution failed for {url}: {e}")
        return url
LLM_MODEL = os.getenv("RECIPE_LLM_MODEL", "gpt-4o-mini")  # change if needed
RECIPE_LLM_MODEL = os.getenv("RECIPE_LLM_MODEL", "gpt-4o-mini")
# Model for TEXT-based recipe extraction (caption + webpage HTML). Defaults to
# RECIPE_LLM_MODEL so behavior is unchanged unless set. Point this at a faster
# text model (e.g. gpt-4.1-mini) to speed up caption/webpage extraction without
# affecting the vision model used for on-screen video-frame text.
RECIPE_TEXT_MODEL = os.getenv("RECIPE_TEXT_MODEL") or RECIPE_LLM_MODEL
print(f"[startup] recipe models: text={RECIPE_TEXT_MODEL}, base/vision={RECIPE_LLM_MODEL}")
# LLM/ffmpeg timeout budget for /extract-recipe (seconds)
EXTRACT_RECIPE_TIMEOUT = int(os.getenv("EXTRACT_RECIPE_TIMEOUT", "50"))
DEFAULT_RECIPE_SERVINGS = int(os.getenv("DEFAULT_RECIPE_SERVINGS", "4"))

# YouTube proxy is now OPT-IN only (set YT_USE_PROXY=1 to re-enable).
# YouTube is downloaded directly — no 3rd-party proxy — by impersonating the
# ANDROID_VR (Oculus Quest) Innertube client, the same approach YoutubeExplode
# uses. That client doesn't require Proof-of-Origin (PO) tokens or signature
# deciphering, so direct connections from datacenter IPs (e.g. Render) work.
# See https://github.com/Tyrrrz/YoutubeExplode/issues/933
YT_PROXY = os.getenv("YT_PROXY")
YT_USE_PROXY = os.getenv("YT_USE_PROXY", "0") == "1"  # opt-in escape hatch if direct ever breaks
SOCIAL_USE_PROXY = os.getenv("SOCIAL_USE_PROXY", "0") == "1"  # same opt-in for FB/IG/TikTok

# Player clients to impersonate for YouTube, in priority order.
# android_vr: PO-token-free, full format access (primary).
# web_safari / web: fallbacks if android_vr is missing a format.
YT_PLAYER_CLIENTS = ["android_vr", "web_safari", "web"]


def _yt_extractor_args() -> dict:
    """yt-dlp extractor_args for proxy-free YouTube extraction."""
    return {"player_client": list(YT_PLAYER_CLIENTS)}
# Proxy for Facebook/Instagram/TikTok. Cloud/datacenter IPs (e.g. Render) get
# challenged/redirect-looped by these sites even with valid cookies, so a
# residential/mobile proxy is usually required to extract from a deployed server.
# Falls back to YT_PROXY if SOCIAL_PROXY is not set, so a single proxy can serve both.
# Falls back to YT_PROXY so a single Decodo residential proxy serves all sites.
SOCIAL_PROXY = os.getenv("SOCIAL_PROXY") or os.getenv("YTDLP_PROXY") or YT_PROXY
print(f"[startup] yt-dlp version: {getattr(yt_dlp.version, '__version__', 'unknown')}")
print(
    f"[startup] proxies: YT_PROXY={'set' if YT_PROXY else 'none'} "
    f"({'ENABLED via YT_USE_PROXY' if YT_USE_PROXY else 'NOT USED — YouTube goes direct via android_vr client'}), "
    f"SOCIAL_PROXY={'set' if SOCIAL_PROXY else 'none'}"
)
PROFILE_VIDEO_DOWNLOADS_DIR = os.getenv("PROFILE_VIDEO_DOWNLOADS_DIR", "./downloads/profile_videos")
PROFILE_VIDEO_PARALLEL_DOWNLOADS = max(1, min(int(os.getenv("PROFILE_VIDEO_PARALLEL_DOWNLOADS", "4")), 10))

# Note: Global ydl_opts is not used - proxy is conditionally applied in individual functions
# Proxy: YT_PROXY for YouTube; SOCIAL_PROXY for Facebook/Instagram/TikTok.
# See _ytdlp_proxy(), ytdlp_base_opts(), _yt_meta(), _download_audio_mp3(), _download_video_to_file().


def _ytdlp_proxy(url: str) -> str | None:
    """Return the proxy to use for a given URL, or None.

    Direct-first strategy: the 1st attempt is proxy-free, and _ydl_extract()
    retries through the matching proxy only if the direct attempt is blocked.
    Proxy fallback applies to YouTube and Facebook ONLY — Instagram/TikTok
    work fine directly (with cookies) and never use a proxy.
    Forcing the proxy on every request is opt-in:
    YT_USE_PROXY=1 (YouTube) / SOCIAL_USE_PROXY=1 (Facebook).
    """
    try:
        if is_youtube_url(url):
            return (YT_PROXY or None) if YT_USE_PROXY else None
        if _is_facebook_url(url):
            return (SOCIAL_PROXY or None) if SOCIAL_USE_PROXY else None
    except Exception:
        pass
    return None


def _is_facebook_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(s in host for s in ("facebook.com", "fb.watch", "fb.com"))


def _fallback_proxy(url: str) -> str | None:
    """Proxy to retry through when a direct attempt is blocked, or None.

    YouTube → YT_PROXY; Facebook → SOCIAL_PROXY (defaults to YT_PROXY).
    Instagram/TikTok → no proxy ever (direct + cookies is sufficient).
    """
    try:
        if is_youtube_url(url):
            return YT_PROXY or None
        if _is_facebook_url(url):
            return SOCIAL_PROXY or None
    except Exception:
        pass
    return None


def _blocked_error(exc: Exception) -> bool:
    """True if the yt-dlp error looks like IP-reputation blocking / anti-bot.

    Covers YouTube (bot-check, 429), and Facebook/Instagram/TikTok
    (login walls, rate limits, redirect loops on datacenter IPs).
    """
    s = str(exc)
    return any(
        marker in s
        for marker in (
            "Sign in to confirm",       # YouTube bot-check
            "Too Many Requests",
            "HTTP Error 429",
            "HTTP Error 403",
            "LOGIN_REQUIRED",
            "login required",           # FB/IG login wall
            "log in",
            "Login Required",
            "checkpoint",               # FB security checkpoint
            "rate-limit",
            "rate limit",
            "redirect loop",
            "Cannot parse data",        # FB serving an interstitial page
            "Restricted Video",
            "unable to extract",        # generic: site served a block page
        )
    )


def _ydl_extract(ydl_opts: dict, video_url: str, *, download: bool):
    """Run yt-dlp extract_info with direct-first / proxy-fallback.

    1st attempt: direct connection (proxy-free; YouTube uses the android_vr
    client). If the site blocks the host IP (common on cloud egress IPs) and a
    fallback proxy is configured for that site (YT_PROXY for YouTube,
    SOCIAL_PROXY for FB/IG/TikTok), retry ONCE through the proxy. This keeps
    proxy bandwidth (and cost) at zero unless the direct path is blocked.
    """
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(video_url, download=download)
    except Exception as e:
        proxy = _fallback_proxy(video_url)
        if proxy and not ydl_opts.get("proxy") and _blocked_error(e):
            print(f"⚠️ Direct attempt blocked ({str(e)[:80]}...); retrying via proxy")
            opts = dict(ydl_opts)
            opts["proxy"] = proxy
            with yt_dlp.YoutubeDL(opts) as ydl:
                return ydl.extract_info(video_url, download=download)
        raise

vector_db = Chroma(persist_directory="./vector_store", embedding_function=OpenAIEmbeddings())
retriever = vector_db.as_retriever(search_kwargs={"k": 3})  # Reduced from 5 to 3 for faster retrieval
llm = ChatOpenAI(
    model="gpt-3.5-turbo", 
    temperature=0.3,  # Lower temperature for faster, more deterministic responses
    max_tokens=4000  # Increased to ensure complete recipes for 3-5 meal recommendations with detailed instructions (each meal ~600-800 tokens, so 3-5 meals need ~3000-4000 tokens)
)

# Note: OpenAI client is initialized once above and reused for all endpoints
# The client.chat.completions.create() calls are just API requests, not re-initializations

# User-friendly message for any 500 (client can show this when status is 500)
DEFAULT_500_USER_MESSAGE = "We ran into a problem. Please try again in a moment."


@app.errorhandler(500)
def handle_500(err):
    """Ensure unhandled exceptions return a consistent JSON body with user_message."""
    return jsonify({
        "error": "Internal Server Error",
        "user_message": DEFAULT_500_USER_MESSAGE,
        "details": str(err) if app.debug else None,
    }), 500


@app.after_request
def ensure_500_user_message(response):
    """Ensure every 500 response includes user_message so clients can show a message."""
    if response.status_code != 500:
        return response
    if not response.is_json:
        return response
    try:
        data = response.get_json()
        if data is None:
            return response
        if "user_message" not in data or not data["user_message"]:
            data = dict(data) if data else {}
            data["user_message"] = data.get("user_message") or DEFAULT_500_USER_MESSAGE
            response.set_data(json.dumps(data))
    except Exception:
        pass
    return response


# Initialize RAG chain once at startup (not on every request)
# Note: system_context is included in the query string, not as a separate prompt variable
# Create RAG chain without custom prompt (system_context is included in query)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    retriever=retriever
)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    email = data["email"]
    user_question = data["question"]

    # Parallelize Firestore calls
    context_result = {}
    history_result = {}

    def fetch_context():
        try:
            context_result["data"] = get_user_context(email)
        except Exception as e:
            print(f"ERROR fetching user context: {e}")
            context_result["data"] = ({}, {'all': []}, {},)

    def fetch_history():
        try:
            history_result["data"] = get_user_chat_history(email)
        except Exception as e:
            print(f"ERROR fetching chat history: {e}")
            history_result["data"] = []

    t1 = Thread(target=fetch_context)
    t2 = Thread(target=fetch_history)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    goal, categorized_meals, user_preferences = context_result.get("data", ({}, {'all': []}, {}))

    # Get recent chat history (limit to last 2 for performance)
    chat_history = history_result.get("data", [])[-2:]  # Only last 2 messages
    # Truncate long bot responses to keep context manageable
    formatted_history = "\n".join([
        f"User: {q}\nBot: {a[:150] + '...' if len(a) > 150 else a}" 
        for q, a in chat_history
    ])

    # Build profile details from user preferences
    profile_parts = []
    if user_preferences:
        if user_preferences.get("age"):
            profile_parts.append(f"Age: {user_preferences.get('age')}")
        if user_preferences.get("gender"):
            profile_parts.append(f"Gender: {user_preferences.get('gender')}")
        if user_preferences.get("height"):
            unit = user_preferences.get("height_unit", "cm")
            profile_parts.append(f"Height: {user_preferences.get('height')} {unit}")
        if user_preferences.get("target_weight"):
            profile_parts.append(f"Target Weight: {user_preferences.get('target_weight')} {user_preferences.get('weigh_unit', 'kg')}")
        if user_preferences.get("weight_goal"):
            profile_parts.append(f"Weight Goal: {user_preferences.get('weight_goal')}")
        if user_preferences.get("lifestyle"):
            profile_parts.append(f"Lifestyle: {user_preferences.get('lifestyle')}")
        if user_preferences.get("meal_per_day"):
            profile_parts.append(f"Meals per day: {user_preferences.get('meal_per_day')}")
        if user_preferences.get("meal_type_list"):
            cuisines = ", ".join(user_preferences.get("meal_type_list"))
            profile_parts.append(f"Preferred cuisines: {cuisines}")
        if user_preferences.get("calorie_goal") is not None:
            profile_parts.append(f"Calorie goal: {user_preferences.get('calorie_goal')} kcal")
        if user_preferences.get("protein_goal") is not None:
            profile_parts.append(f"Protein goal: {user_preferences.get('protein_goal')} g")
        if user_preferences.get("carbs_goal") is not None:
            profile_parts.append(f"Carbs goal: {user_preferences.get('carbs_goal')} g")
        # Check for fat_goal (handle both snake_case and camelCase variants)
        if "fat_goal" in user_preferences:
            profile_parts.append(f"Fat goal: {user_preferences.get('fat_goal')} g")
        elif "fatGoal" in user_preferences:
            profile_parts.append(f"Fat goal: {user_preferences.get('fatGoal')} g")

    if profile_parts:
        goal_summary = "User Profile:\n" + "\n".join(profile_parts)
    elif goal:
        goal_summary = f"User goal: {goal.get('goalType', 'not set')}, Current: {goal.get('currentWeight')}kg, Target: {goal.get('targetWeight')}kg by {goal.get('targetDate')}"
    else:
        goal_summary = "User goal: No specific goals set"
    
    # Debug: Print goal_summary to verify it contains weight_goal
    print(f"DEBUG: Goal summary for {email}: {goal_summary}")
    print(f"DEBUG: User preferences keys: {list(user_preferences.keys()) if user_preferences else 'None'}")
    print(f"DEBUG: Fat goal value (fat_goal): {user_preferences.get('fat_goal')}")
    print(f"DEBUG: Fat goal value (fatGoal): {user_preferences.get('fatGoal')}")
    
    # Get recent meals (limit to last 2 for performance)
    all_meals = categorized_meals.get('all', [])[:2]  # Only last 2 meals
    meals_summary = "\n".join(all_meals) if all_meals else "No recent meals found"
    
    # Don't truncate goal_summary - user profile data is critical and must be complete
    # The performance impact of a few hundred extra characters is negligible compared to LLM processing time
    system_context = f"""You are a nutrition assistant. User: {goal_summary}. Recent: {formatted_history[:200] if formatted_history else 'New conversation'}. Meals: {meals_summary[:200] if meals_summary else 'None'}.

CRITICAL: Meal history (shown above as "Meals:") is COMPLETELY OPTIONAL and used ONLY for understanding patterns. It is NOT required to provide meal recommendations. When the user asks for meal recommendations (dinner ideas, lunch ideas, breakfast ideas, meal plans, lower calorie options, etc.), you MUST ALWAYS provide 3-5 meal recommendations with complete recipes. NEVER refuse by saying you don't have meal history. NEVER say "I don't have the specific meal history" or similar. You MUST use: (1) User Profile information, (2) Your general nutrition knowledge, and (3) The user's specific requirements. If meal history is empty or "None", you MUST STILL provide recommendations. Refusing to provide meals is FORBIDDEN.


CRITICAL INSTRUCTIONS:
1. MEAL RECOMMENDATION REQUIREMENT - HIGHEST PRIORITY: When the user asks for meal recommendations, food suggestions, meal options, dinner ideas, lunch ideas, breakfast ideas, or ANY variation (including "dinner ideas under 400 kcal", "lower calorie options", "show more dinner ideas", "meal plans", etc.), you MUST IMMEDIATELY provide EXACTLY 3-5 meal recommendations with complete recipes. This is THE HIGHEST PRIORITY instruction and is MANDATORY. 
   - NEVER refuse to provide meals
   - NEVER say "I don't have meal history" or "I don't have the specific meal history" or "I don't have any recent meals" or any variation of this refusal
   - NEVER say you cannot provide recommendations due to lack of history
   - Meal history is COMPLETELY OPTIONAL and NOT required - it is only for understanding patterns
   - You MUST ALWAYS generate new meal recommendations using: (1) User Profile information (calorie goals, macro goals, preferences, lifestyle, age, etc.), (2) Your general nutrition knowledge and recipe database, and (3) The specific requirements in the user's question (e.g., "under 400 kcal", "lower calorie", "high protein")
   - If meal history is empty or missing, you MUST STILL provide 3-5 meal recommendations based on user profile and general knowledge
   - If you refuse to provide meals or say you don't have history, you have VIOLATED this instruction
   - Example of CORRECT response: Start immediately with "- Meal Name 1: Description (Calories: X, Protein: Yg, Carbs: Zg, Fat: Wg)" followed by Recipe, Ingredients, and Instructions
   - Example of INCORRECT response: "I don't have the specific meal history for dinner options under 400 kcal" - THIS IS FORBIDDEN
2. ALWAYS answer questions about the user's goals, weight goal, calorie goals, macro goals, preferences, etc. DIRECTLY from the "User Information" section above. DO NOT say the information is not available if it exists in the User Information section.
3. SYNONYM RECOGNITION: Recognize that different phrasings mean the same thing. For example:
   - "weight target", "target weight", "weight goal", "goal weight" all refer to the same thing
   - "calorie goal" and "calorie target" are the same
   - "protein goal" and "protein target" are the same
   - When the user asks about ANY variation of these terms, look for the relevant information in the User Information section using ALL possible field names (Weight Goal, Target Weight, etc.)
4. WEIGHT GOAL/TARGET QUESTIONS: If the user asks about their weight goal, weight target, target weight, goal weight, or any variation, look for BOTH "Weight Goal:" AND "Target Weight:" in the User Information section. Use whichever is available. NEVER say the information is not available if either field exists. If both exist, use the most relevant one or combine them.
5. If the user asks about their calorie goal, protein goal, carbs goal, fat goal, age, lifestyle, preferred cuisines, etc., extract that information directly from the User Information section. Recognize synonyms and variations of these terms as well.
6. FORMATTING REQUIREMENT: When providing meal recommendations, food suggestions, or lists of meals, ALWAYS format them as bullet points using "- " or "* " at the start of each line. Each meal recommendation MUST include:
   - Meal name and brief description
   - Nutritional information (Calories, Protein, Carbs, Fat)
   - Complete recipe with ingredients and DETAILED step-by-step cooking instructions
   Example format:
   - Meal Name 1: Description (Calories: X, Protein: Yg, Carbs: Zg, Fat: Wg)
     Recipe: 
     Ingredients: 
     - Ingredient 1: exact quantity (e.g., "1 cup", "200g", "2 tablespoons")
     - Ingredient 2: exact quantity
     - Ingredient 3: exact quantity
     Instructions:
     1. Detailed step 1 with specific actions, temperatures, and times (e.g., "Heat 1 tablespoon olive oil in a large pan over medium-high heat for 2 minutes")
     2. Detailed step 2 with specific techniques and measurements
     3. Detailed step 3 with cooking times and temperatures
     4. Continue with numbered steps until the meal is complete
   - Meal Name 2: Description (Calories: X, Protein: Yg, Carbs: Zg, Fat: Wg)
     Recipe: [same detailed format]
   - Meal Name 3: Description (Calories: X, Protein: Yg, Carbs: Zg, Fat: Wg)
     Recipe: [same detailed format]
   ALWAYS include a complete recipe for every meal you recommend, even if the recipe is not in the retrieved context. Use your knowledge to provide accurate recipes.
7. ALWAYS maintain conversation context - remember everything the user has asked and your previous responses
8. Use the complete conversation history to provide contextual and personalized responses
9. When the user asks "what was my last question", refer to the question they asked BEFORE their current question (not the current one)
10. Build upon previous conversations - if they ask follow-up questions, reference what you've already discussed
11. Be conversational and remember what you've told them before
12. When asked about specific meal types (breakfast, lunch, dinner, snacks), use only the data from that category when analyzing history or patterns.
13. The meal data includes detailed nutritional information (calories, carbs, protein, fat) for each meal.
14. If the user asks about trends or patterns, analyze their meal history across multiple entries.
15. Provide personalized insights based on their eating patterns and previous questions.
16. Maintain a helpful, friendly tone throughout the conversation.
17. Use the "User Profile" section (age, lifestyle, calorie/macro goals, preferred cuisines, etc.) to tailor every recommendation. Respect their macros, calorie targets, and cuisine preferences when possible.
18. DETAILED RECIPE REQUIREMENT: ALWAYS provide a complete, DETAILED recipe for EVERY meal you recommend. REMEMBER: You must provide 3-5 meals (never just 1), and each meal needs a full recipe. The recipe MUST include:
    - Ingredients list with EXACT quantities (e.g., "1 cup", "200g", "2 tablespoons", "1 medium onion", "3 cloves garlic")
    - Numbered step-by-step instructions that are SPECIFIC and ACTIONABLE:
      * Include exact cooking temperatures (e.g., "375°F", "medium-high heat")
      * Include exact cooking times (e.g., "cook for 5-7 minutes", "bake for 25 minutes")
      * Include specific techniques (e.g., "sauté until golden brown", "whisk until smooth", "simmer uncovered")
      * Include preparation details (e.g., "dice into 1-inch cubes", "chop finely", "slice thinly")
      * Include when to add ingredients (e.g., "add after 2 minutes", "stir in at the end")
      * Include visual/textural cues (e.g., "until tender", "until golden", "until sauce thickens")
    Never skip the recipe or provide vague instructions. The recipe should be detailed enough that someone with basic cooking knowledge can successfully prepare the meal without additional research. You have 4000 tokens available, so use them to provide 3-5 complete meal recommendations with full recipes.
19. NO CROSS-QUESTIONING OR REFUSAL FOR MEAL REQUESTS - ABSOLUTELY FORBIDDEN: If the user asks for meal ideas, meal plans, or specific meal suggestions (for example, "Show dinner ideas under 400 kcal", "lower calorie options", "show more dinner ideas", "dinner ideas", etc.), you MUST directly provide the requested meal recommendations. 
   FORBIDDEN RESPONSES (DO NOT USE THESE):
   - "I don't have the specific meal history" or "I don't have the specific meal history for dinner options under 400 kcal"
   - "I don't have meal history" or "I don't have any recent meals"
   - "I don't have the necessary meal history" or any variation
   - "Would you like me to suggest..." or any follow-up questions
   - Any response that refuses to provide meals
   REQUIRED RESPONSE: You MUST ALWAYS provide 3-5 meal recommendations with complete recipes using: (1) User Profile (calorie/macro goals, preferences, lifestyle), (2) Your general nutrition knowledge, and (3) The user's specific requirements. Meal history is completely optional and NOT required. If meal history is missing or empty, you MUST still provide recommendations based on user profile and general knowledge. Start your response immediately with the first meal recommendation in the required format.
20. FRESH MEAL RECOMMENDATIONS: When the user asks for meal recommendations, DO NOT simply repeat or select meals from their past meal history. Always generate NEW meal ideas and recipes that fit their goals and preferences. You may use history only to understand patterns and preferences, but the recommended meals themselves should be fresh suggestions, not just a recap of what they already ate.
"""

    # Use invoke() instead of run() for better performance
    # Format the query with system context
    query = f"{system_context}\n\nUser question: {user_question}"
    result = rag_chain.invoke({"query": query})
    response = result.get("result", str(result))

    # Save chat interaction for future context
    Thread(target=save_user_chat, args=(email, user_question, response)).start()
    return jsonify({"reply": response})


@app.route("/detect-ingredients", methods=["POST"])
def detect_ingredients():
    try:
        image_url_for_vision = None
        
        # Priority: file upload > base64 JSON
        # Check for file upload (multipart/form-data)
        if 'image' in request.files:
            uploaded_file = request.files['image']
            if uploaded_file.filename:
                print("🔍 Received image file upload:", uploaded_file.filename)
                
                # Read image bytes
                image_bytes = uploaded_file.read()
                
                # Determine content type from file extension
                filename = uploaded_file.filename.lower()
                if filename.endswith(('.jpg', '.jpeg')):
                    content_type = "image/jpeg"
                elif filename.endswith('.png'):
                    content_type = "image/png"
                elif filename.endswith('.gif'):
                    content_type = "image/gif"
                elif filename.endswith('.webp'):
                    content_type = "image/webp"
                else:
                    content_type = "image/jpeg"  # Default
                
                # Convert to base64
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                image_url_for_vision = f"data:{content_type};base64,{image_base64}"
                print("✅ Image file converted to base64")
        
        # Check for base64 in JSON (application/json)
        elif request.is_json:
            data = request.get_json()
            image_base64 = data.get("imageBase64")
            image_format = data.get("imageFormat", "jpg")
            
            if image_base64:
                print("🔍 Received base64 image")
                # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
                if "," in image_base64:
                    image_base64 = image_base64.split(",")[-1]
                
                # Determine content type from format
                format_to_mime = {
                    "jpg": "image/jpeg",
                    "jpeg": "image/jpeg",
                    "png": "image/png",
                    "gif": "image/gif",
                    "webp": "image/webp"
                }
                content_type = format_to_mime.get(image_format.lower(), "image/jpeg")
                
                # Create data URL for GPT-Vision
                image_url_for_vision = f"data:{content_type};base64,{image_base64}"
                print("✅ Using base64 image directly")
        
        # If no image was provided through any method
        if not image_url_for_vision:
            return jsonify({
                "error": "Image is required",
                "details": "Provide image in one of these formats: 1) File upload (multipart/form-data with 'image' field), or 2) Base64 JSON (imageBase64 + imageFormat)"
            }), 400

        print("🤖 Calling GPT Vision API...")

        # 2. Call GPT-Vision API using pre-initialized OpenAI client
        # Note: 'client' is initialized once at startup (line 20), so this is just an API call, not a re-initialization
        try:
            vision_response = client.chat.completions.create(
                model="gpt-4o-mini",  # Cost-effective: ~10x cheaper than gpt-4o, still supports vision
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "List visible food ingredients. Comma-separated only, no explanations.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url_for_vision
                                },
                            },
                        ],
                    }
                ],
                max_tokens=200,  # Reduced from 300 - ingredients list doesn't need that many tokens
                temperature=0.2  # Lower temperature for more deterministic, cheaper responses
            )

            # 3. Extract content
            content = vision_response.choices[0].message.content
            print("📝 Raw AI content:", content)

            # 4. Parse comma-separated ingredients
            ingredients = [
                item.strip().lower()
                for item in content.replace("\n", ",").split(",")
                if item.strip() and ":" not in item.lower() and "sorry" not in item.lower()
            ][:15]

            print("🥬 Final ingredients:", ingredients)

            if not ingredients:
                return jsonify({
                    "ingredients": [],
                    "message": "No ingredients detected in this image",
                })

            return jsonify({
                "ingredients": ingredients,
                "message": f"Found {len(ingredients)} ingredients",
            })

        except Exception as vision_error:
            print("💥 GPT Vision error:", str(vision_error))
            return jsonify({
                "error": "Failed to analyze image with GPT Vision",
                "user_message": "We couldn't read ingredients from this image. Please try another photo or type them in.",
                "details": str(vision_error),
            }), 500

    except Exception as e:
        print("💥 Critical error:", str(e))
        return jsonify({
            "error": "Something went wrong analyzing the image",
            "user_message": "We couldn't analyze this image. Please try again or type your ingredients instead.",
            "details": str(e),
        }), 500


@app.route("/generate-meals", methods=["POST"])
def generate_meals():
    data = request.get_json()
    
    # Extract parameters
    ingredients = data.get("ingredients", [])
    cuisine = data.get("cuisine", "")
    cooking_time = data.get("cookingTime", "")
    diet = data.get("diet", "")
    macro_targets = data.get("macroTargets", {})
    meal_count = data.get("mealCount", 3)
    include_images = data.get("includeImages", True)
    
    # Validate ingredients list is not empty
    if not ingredients or len(ingredients) == 0:
        return jsonify({"error": "Ingredients list cannot be empty"}), 400
    
    # Compute macro-per-meal values if macroTargets exist
    calories_per_meal = None
    protein_per_meal = None
    carbs_per_meal = None
    fats_per_meal = None
    
    if macro_targets:
        total_calories = macro_targets.get("calories", 0)
        total_protein = macro_targets.get("protein", 0)
        total_carbs = macro_targets.get("carbs", 0)
        total_fats = macro_targets.get("fats", 0)
        
        if meal_count > 0:
            calories_per_meal = total_calories / meal_count if total_calories > 0 else None
            protein_per_meal = total_protein / meal_count if total_protein > 0 else None
            carbs_per_meal = total_carbs / meal_count if total_carbs > 0 else None
            fats_per_meal = total_fats / meal_count if total_fats > 0 else None
    
    # Build the GPT prompt
    prompt_parts = []
    prompt_parts.append(f"Generate exactly {meal_count} meal recommendations.")
    prompt_parts.append(f"\nRequired ingredients: {', '.join(ingredients)}")
    
    if cuisine:
        prompt_parts.append(f"Cuisine preference: {cuisine}")
    if cooking_time:
        prompt_parts.append(f"Cooking time preference: {cooking_time}")
    if diet:
        prompt_parts.append(f"Diet preference: {diet}")
    
    if macro_targets and (calories_per_meal or protein_per_meal or carbs_per_meal or fats_per_meal):
        prompt_parts.append("\nMacro target requirements per meal:")
        if calories_per_meal:
            prompt_parts.append(f"- Calories: approximately {calories_per_meal:.0f} kcal")
        if protein_per_meal:
            prompt_parts.append(f"- Protein: approximately {protein_per_meal:.1f}g")
        if carbs_per_meal:
            prompt_parts.append(f"- Carbs: approximately {carbs_per_meal:.1f}g")
        if fats_per_meal:
            prompt_parts.append(f"- Fats: approximately {fats_per_meal:.1f}g")
    
    # JSON schema for meal object (optimized - more concise)
    macro_summary = ""
    if macro_targets:
        totals = []
        if macro_targets.get('calories'):
            totals.append(f"{macro_targets.get('calories', 0):.0f} kcal")
        if macro_targets.get('protein'):
            totals.append(f"{macro_targets.get('protein', 0):.1f}g protein")
        if macro_targets.get('carbs'):
            totals.append(f"{macro_targets.get('carbs', 0):.1f}g carbs")
        if macro_targets.get('fats'):
            totals.append(f"{macro_targets.get('fats', 0):.1f}g fats")
        if totals:
            macro_summary = f" Total targets: {', '.join(totals)}."
    
    prompt_parts.append(f"""
JSON array with {meal_count} meals. Schema: {{"name":"string","description":"string","calories":number,"protein":number,"carbs":number,"fats":number,"ingredients":["string"],"instructions":["step"],"cookingTime":"string","servings":number}}
Rules: Use required ingredients. Step-by-step instructions.{macro_summary} Valid JSON only, no markdown, no trailing commas.
""")
    
    user_prompt = "\n".join(prompt_parts)
    
    # System message (cost-optimized - minimal tokens)
    system_message = """Nutrition expert. Return JSON array only. No markdown. No trailing commas. Accurate nutrition."""
    
    # Calculate optimal max_tokens based on meal count (cost-optimized)
    # Reduced estimate: each meal ~500-600 tokens (optimized prompts)
    estimated_tokens = meal_count * 550 + 150
    max_tokens = min(max(estimated_tokens, 1200), 3500)  # Reduced range: 1200-3500 tokens
    
    # Call OpenAI Chat Completion API (optimized for speed)
    response_text = None
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower temperature for faster, more deterministic responses
            max_tokens=max_tokens
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*', '', response_text)
        response_text = response_text.strip()
        
        # Clean trailing commas from JSON (common GPT issue)
        # Remove trailing commas before closing brackets/braces (handle nested structures)
        # This regex handles whitespace and newlines before closing brackets/braces
        response_text = re.sub(r',(\s*})', r'\1', response_text)  # Remove trailing comma before }
        response_text = re.sub(r',(\s*])', r'\1', response_text)  # Remove trailing comma before ]
        # Also handle cases with newlines
        response_text = re.sub(r',\s*\n\s*}', '\n}', response_text)
        response_text = re.sub(r',\s*\n\s*]', '\n]', response_text)
        
        # Parse JSON with retry logic for trailing commas
        meals = None
        parse_attempts = 0
        while parse_attempts < 3:
            try:
                meals = json.loads(response_text)
                break
            except json.JSONDecodeError as parse_error:
                parse_attempts += 1
                if parse_attempts >= 3:
                    raise  # Re-raise if all attempts failed
                # Try more aggressive cleaning
                # Remove trailing commas more aggressively line by line
                lines = response_text.split('\n')
                cleaned_lines = []
                for i, line in enumerate(lines):
                    # Remove trailing comma if next non-empty line starts with } or ]
                    if i < len(lines) - 1:
                        next_line = lines[i + 1].strip()
                        if next_line in ['}', ']'] and line.rstrip().endswith(','):
                            cleaned_lines.append(line.rstrip().rstrip(','))
                        else:
                            cleaned_lines.append(line)
                    else:
                        cleaned_lines.append(line)
                response_text = '\n'.join(cleaned_lines)
                # Try one more cleanup pass
                response_text = re.sub(r',(\s*})', r'\1', response_text)
                response_text = re.sub(r',(\s*])', r'\1', response_text)
        
        # Validate and fill missing fields with defaults
        default_values = {
            "cookingTime": "30 minutes",
            "servings": 1
        }
        
        validated_meals = []
        for meal in meals:
            # Ensure all required fields exist
            validated_meal = {
                "name": meal.get("name", "Unnamed Meal"),
                "description": meal.get("description", ""),
                "calories": meal.get("calories", 0),
                "protein": meal.get("protein", 0),
                "carbs": meal.get("carbs", 0),
                "fats": meal.get("fats", 0),
                "ingredients": meal.get("ingredients", []),
                "instructions": meal.get("instructions", []),
                "cookingTime": meal.get("cookingTime", default_values["cookingTime"]),
                "servings": meal.get("servings", default_values["servings"]),
                "imageUrl": None  # Will be populated if include_images is True
            }
            validated_meals.append(validated_meal)
        
        # Generate images for meals if requested (b64_json → Firebase Storage, same as /food-logging)
        if include_images:
            default_image_url = os.getenv("DEFAULT_MEAL_IMAGE_URL", "")
            workers = _food_logging_image_pool_size(len(validated_meals))

            def _generate_and_persist(m: dict) -> str:
                b64 = _generate_food_log_meal_image_b64(m)
                return _food_logging_finalize_image_from_b64(b64, default_image_url)

            with ThreadPoolExecutor(max_workers=workers) as ex:
                finalized = list(ex.map(_generate_and_persist, validated_meals))
            for meal, url in zip(validated_meals, finalized):
                meal["imageUrl"] = url
        
        # Compute totals
        totals = {
            "calories": sum(meal["calories"] for meal in validated_meals),
            "protein": sum(meal["protein"] for meal in validated_meals),
            "carbs": sum(meal["carbs"] for meal in validated_meals),
            "fats": sum(meal["fats"] for meal in validated_meals)
        }
        
        return jsonify({
            "meals": validated_meals,
            "totals": totals,
            "mealCount": len(validated_meals)
        })
        
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse JSON response: {str(e)}"
        if response_text:
            error_msg += f"\nRaw response: {response_text[:500]}"
        return jsonify({
            "error": error_msg,
            "user_message": "We ran into a problem generating your meals. Please try again in a moment.",
        }), 500
    except Exception as e:
        # Handle OpenAI API errors specifically
        error_str = str(e)
        error_dict = {}
        
        # Try to extract OpenAI error details from the exception
        # OpenAI errors often contain nested error information
        try:
            # Check if error has response attribute (OpenAI SDK errors)
            if hasattr(e, 'response') and hasattr(e.response, 'json'):
                error_data = e.response.json()
                if 'error' in error_data:
                    openai_error = error_data['error']
                    error_code = openai_error.get('code', '')
                    error_type = openai_error.get('type', '')
                    error_message = openai_error.get('message', str(e))
                    
                    if error_code == 'insufficient_quota' or error_type == 'insufficient_quota' or '429' in error_str:
                        error_dict = {
                            "error": "OpenAI API quota exceeded",
                            "message": error_message,
                            "type": "quota_exceeded",
                            "status_code": 429,
                            "details": "Please check your OpenAI plan and billing details at https://platform.openai.com/account/billing"
                        }
                        return jsonify(error_dict), 429
                    elif error_code == 'invalid_api_key' or '401' in error_str:
                        error_dict = {
                            "error": "OpenAI API authentication failed",
                            "message": error_message,
                            "type": "authentication_error",
                            "status_code": 401
                        }
                        return jsonify(error_dict), 401
                    elif 'rate_limit' in error_code.lower() or 'rate_limit' in error_type.lower():
                        error_dict = {
                            "error": "OpenAI API rate limit exceeded",
                            "message": error_message,
                            "type": "rate_limit_exceeded",
                            "status_code": 429
                        }
                        return jsonify(error_dict), 429
        except:
            pass  # Fall through to string-based detection
        
        # String-based error detection (fallback)
        if "429" in error_str or "quota" in error_str.lower() or "insufficient_quota" in error_str.lower():
            error_dict = {
                "error": "OpenAI API quota exceeded",
                "message": "You have exceeded your OpenAI API quota. Please check your plan and billing details.",
                "type": "quota_exceeded",
                "status_code": 429,
                "details": "For more information, visit: https://platform.openai.com/docs/guides/error-codes/api-errors"
            }
            return jsonify(error_dict), 429
        elif "401" in error_str or "unauthorized" in error_str.lower():
            error_dict = {
                "error": "OpenAI API authentication failed",
                "message": "Invalid API key or authentication error",
                "type": "authentication_error",
                "status_code": 401
            }
            return jsonify(error_dict), 401
        elif "rate_limit" in error_str.lower() or "rate limit" in error_str.lower():
            error_dict = {
                "error": "OpenAI API rate limit exceeded",
                "message": "Too many requests. Please try again later.",
                "type": "rate_limit_exceeded",
                "status_code": 429
            }
            return jsonify(error_dict), 429
        else:
            # Generic error
            error_dict = {
                "error": "Error generating meals",
                "user_message": "We ran into a problem generating your meals. Please try again in a moment.",
                "message": str(e),
                "type": "unknown_error",
                "status_code": 500
            }
            print(f"Error generating meals: {error_str}")
            return jsonify(error_dict), 500

@app.route("/speak", methods=["POST"])
def speak():
    text = request.json["text"]
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text
    )
    audio_stream = io.BytesIO(response.read())
    return send_file(audio_stream, mimetype="audio/mpeg")


def _get_meal_keys(num_meals: int) -> list[str]:
    """
    Return canonical meal keys for dynamic meal count.
    First four keys are breakfast/lunch/dinner/snacks, extras are snacks_2, snacks_3, ...
    """
    base = ["breakfast", "lunch", "dinner", "snacks"]
    if num_meals <= 4:
        return base[:num_meals]
    extra = [f"snacks_{i}" for i in range(2, num_meals - 4 + 2)]
    return base + extra


def _meal_type_from_key(meal_key: str) -> str:
    if meal_key == "breakfast":
        return "Breakfast"
    if meal_key == "lunch":
        return "Lunch"
    if meal_key == "dinner":
        return "Dinner"
    return "Snack"


def _dietary_restrictions_plant_based_prompt_block(restrictions: list) -> str:
    """
    When restrictions include vegan or vegetarian, return strict LLM instructions so
    recommendations exclude meat (vegan: no animal products at all).
    """
    if not isinstance(restrictions, list):
        return ""
    lowered = [str(x).lower() for x in restrictions if x is not None]
    if any("vegan" in r for r in lowered):
        return (
            "CRITICAL — VEGAN: Every meal must be fully plant-based. "
            "Do NOT include meat, poultry, fish, shellfish, eggs, dairy, honey, gelatin, or other animal-derived ingredients. "
            "Use only vegan proteins and ingredients (legumes, tofu, tempeh, seitan, nuts, seeds, plant milks, etc.).\n"
        )
    if any("vegetarian" in r for r in lowered):
        return (
            "CRITICAL — VEGETARIAN: Every meal must be vegetarian. "
            "Do NOT include meat, poultry, fish, or shellfish. "
            "Eggs and dairy are allowed unless other restrictions forbid them.\n"
        )
    return ""


def _recommend_meals_image_model() -> str:
    """Image model: set RECOMMEND_MEALS_IMAGE_MODEL env var to override. Defaults to dall-e-3."""
    return (os.getenv("RECOMMEND_MEALS_IMAGE_MODEL") or "dall-e-3").strip()


def _recommend_meals_dalle_size(*, image_quality: str | None = None) -> str:
    """
    Size selection based on model:
      dall-e-3  → 1024x1024 (low/medium) | 1792x1024 (high)
      gpt-image-1 → 1024x1024 (low/medium) | 1536x1024 (high)
      dall-e-2  → 256x256 (low) | 512x512 (medium) | 1024x1024 (high)
    Override with RECOMMEND_MEALS_DALLE_SIZE env var.
    """
    model = _recommend_meals_image_model()
    q = (image_quality or "low").strip().lower()
    if "dall-e-3" in model:
        tier = {"low": "1024x1024", "medium": "1024x1024", "high": "1792x1024"}
        allowed = {"1024x1024", "1792x1024", "1024x1792"}
        default = "1024x1024"
    elif "gpt-image-1" in model:
        tier = {"low": "1024x1024", "medium": "1024x1024", "high": "1536x1024"}
        allowed = {"1024x1024", "1536x1024", "1024x1536"}
        default = "1024x1024"
    else:  # dall-e-2 or unknown
        tier = {"low": "256x256", "medium": "512x512", "high": "1024x1024"}
        allowed = {"256x256", "512x512", "1024x1024"}
        default = "256x256"
    override = (os.getenv("RECOMMEND_MEALS_DALLE_SIZE") or "").strip()
    if override in allowed:
        return override
    return tier.get(q, default)


def _recommend_meals_image_pool_size(num_images: int) -> int:
    """Parallel DALL·E calls; higher = lower wall-clock time (capped)."""
    try:
        w = int(os.getenv("RECOMMEND_MEALS_IMAGE_MAX_WORKERS", "16"))
    except ValueError:
        w = 8
    w = max(1, min(w, 16))
    return min(w, max(1, num_images))


def _meal_image_prompt_for_recommend(name: str, desc: str, *, fast_mode: bool) -> str:
    """Short prompts = slightly faster API handling; truncate description for lower token load."""
    n = (name or "Meal").strip()[:100]
    d = (desc or "").strip()
    d = d[:40] if fast_mode else d[:70]
    if d:
        return f"Food photo: {n}. {d}"
    return f"Food photo: {n}"


def _food_logging_image_pool_size(num_images: int) -> int:
    try:
        w = int(
            os.getenv("FOOD_LOGGING_IMAGE_MAX_WORKERS")
            or os.getenv("RECOMMEND_MEALS_IMAGE_MAX_WORKERS", "8")
        )
    except ValueError:
        w = 8
    w = max(1, min(w, 16))
    return min(w, max(1, num_images))


def _generate_meal_image_b64(
    name: str,
    description: str,
    *,
    fast_mode: bool = True,
    log_prefix: str = "meal-image",
    log_context: str = "",
) -> str | None:
    """
    Shared meal image generation (same as /recommend-meals):
    gpt-image-1, 1024x1024, quality=low; returns base64 for Firebase upload.
    """
    ctx = f" {log_context}" if log_context else ""
    try:
        prompt = _meal_image_prompt_for_recommend(name, description, fast_mode=fast_mode)
        print(f"[{log_prefix}] generating image{ctx} prompt={prompt!r}")
        img = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
            quality="low",
            n=1,
        )
        if not (img and img.data):
            print(f"[{log_prefix}] empty response{ctx}")
            return None
        b64 = getattr(img.data[0], "b64_json", None)
        if b64:
            print(f"[{log_prefix}] success (b64){ctx}")
            return b64
        img_url = getattr(img.data[0], "url", None)
        if img_url:
            print(f"[{log_prefix}] success (url){ctx}, downloading")
            r = requests.get(img_url, timeout=30)
            r.raise_for_status()
            return base64.b64encode(r.content).decode("utf-8")
        print(f"[{log_prefix}] no b64 or url{ctx}")
        return None
    except Exception as e:
        print(f"[{log_prefix}] image error{ctx}: {type(e).__name__}: {e}")
        return None


def _generate_food_log_meal_image_b64(meal: dict, *, fast_mode: bool = True) -> str | None:
    """Food-logging wrapper around shared recommend-meals image generation."""
    return _generate_meal_image_b64(
        meal.get("name", "Meal"),
        meal.get("description", ""),
        fast_mode=fast_mode,
        log_prefix="food-logging",
    )


def _food_log_photo_to_storage_url(photo_data_url: str, default_image_url: str) -> str:
    """Upload submitted photo bytes to Firebase; avoid returning huge data: URLs in JSON."""
    if not photo_data_url or not str(photo_data_url).startswith("data:"):
        return default_image_url
    try:
        header, b64_part = str(photo_data_url).split(",", 1)
        content_type = "image/jpeg"
        if "image/png" in header:
            content_type = "image/png"
        elif "image/webp" in header:
            content_type = "image/webp"
        raw = base64.b64decode(b64_part)
        path = f"food-log-photos/{uuid.uuid4().hex}.jpg"
        url = upload_meal_image_bytes_to_storage(raw, content_type, storage_path=path)
        return url if url else default_image_url
    except Exception as e:
        print(f"[food-logging] photo upload failed: {e}")
        return default_image_url


def _food_logging_finalize_image_from_b64(b64_json: str | None, default_image_url: str) -> str:
    """Decode DALL·E b64_json and upload directly to Firebase Storage.
    Returns a stable firebasestorage.googleapis.com URL, or default_image_url on failure.
    """
    if not b64_json:
        return default_image_url
    try:
        raw = base64.b64decode(b64_json)
    except Exception as e:
        print(f"[food-logging] image b64 decode failed: {e}")
        return default_image_url
    url = upload_meal_image_bytes_to_storage(raw, "image/png")
    return url if url else default_image_url


def _finalize_meal_image_url(openai_url: str | None, default_image_url: str) -> str:
    """Upload DALL·E CDN URL to Firebase Storage when possible; else return default URL."""
    if not openai_url:
        return default_image_url
    permanent = try_persist_meal_image_from_openai_url(openai_url)
    return permanent or default_image_url or openai_url


def _recommend_meals_finalize_image_from_b64(
    b64_json: str | None,
    default_image_url: str,
    plan_id: str,
    day_index: int,
    meal_key: str,
) -> str:
    """
    Decode DALL·E b64_json, upload to Firebase Storage, record URL in Firestore (recommend_meal_images).
    """
    if not b64_json:
        return default_image_url
    try:
        raw = base64.b64decode(b64_json)
    except Exception as e:
        print(f"[recommend-meals] image b64 decode failed: {e}")
        return default_image_url
    storage_path = recommend_meal_image_storage_path(plan_id, day_index, meal_key)
    url = upload_meal_image_bytes_to_storage(raw, "image/png", storage_path=storage_path)
    if url:
        save_recommend_meal_image_record(plan_id, day_index, meal_key, url)
        return url
    print(
        f"[recommend-meals] Firebase upload failed for plan={plan_id} "
        f"day={day_index} meal={meal_key} path={storage_path}"
    )
    return default_image_url


def _chat_completion_limit_kw(model: str, max_tokens: int) -> dict:
    """
    Some newer OpenAI chat models reject max_tokens and require max_completion_tokens.
    """
    m = (model or "").lower()
    if m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3"):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}


def _recommend_llm_completion_budget(model: str, base_max: int) -> int:
    """
    GPT-5 / o-series may use much of max_completion_tokens for internal reasoning before
    emitting visible JSON; small budgets can yield empty message.content.
    """
    m = (model or "").lower()
    if m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3"):
        return max(base_max, 4096)
    return base_max


def _model_prefers_completion_token_param(model: str) -> bool:
    m = (model or "").lower()
    return m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3")


def _chat_response_message_text(completion) -> str:
    try:
        msg = completion.choices[0].message
        if msg and msg.content:
            return str(msg.content).strip()
    except (IndexError, AttributeError, TypeError):
        pass
    return ""


def _meal_has_name(meal: object) -> bool:
    return isinstance(meal, dict) and bool(str(meal.get("name", "")).strip())


def _unwrap_recommend_day_payload(day_obj: dict, meal_keys: list[str], day_number: int) -> dict:
    """
    Models sometimes return weekly-shaped JSON:
      { "days": [ { "breakfast": {...}, ... } ], "daily_macro_targets": ... }
    with empty top-level meal slots. Prefer the inner day object when it has the real meals.
    """
    if not isinstance(day_obj, dict):
        return day_obj
    days_arr = day_obj.get("days")
    if not isinstance(days_arr, list) or not days_arr:
        return day_obj
    first = days_arr[0]
    if not isinstance(first, dict):
        return day_obj

    def score(d: dict) -> int:
        return sum(1 for k in meal_keys if _meal_has_name(d.get(k)))

    if score(first) > score(day_obj):
        out = dict(first)
        out["day"] = out.get("day", day_number)
        return out
    return day_obj


def _log_recommend_api_request(route_label: str, data: dict) -> None:
    """
    Print request URL, method, and JSON body to server stdout (terminal).
    Set LOG_API_REQUESTS=1 (or true) in the environment to enable.
    """
    if os.getenv("LOG_API_REQUESTS", "").lower() not in ("1", "true", "yes"):
        return
    try:
        print(f"\n[{route_label}] ========== incoming request ==========")
        print(f"[{route_label}] {request.method} {request.url}")
        print(f"[{route_label}] remote: {request.remote_addr}")
        ct = request.headers.get("Content-Type", "")
        print(f"[{route_label}] Content-Type: {ct}")
        body = json.dumps(data, ensure_ascii=False, indent=2)
        if len(body) > 16000:
            body = body[:16000] + "\n... [truncated]"
        print(f"[{route_label}] JSON body:\n{body}")
        print(f"[{route_label}] ========== end request ==========\n")
    except Exception as e:
        print(f"[{route_label}] logging error: {e}")


@app.route("/recommend-meals", methods=["POST"])
def recommend_meals():
    """
    Generate a multi-day meal plan with dynamic num_days and num_meals.
    When include_images is true, image_quality may be low|medium|high (default low = 256px DALL·E).
    Images use response_format=b64_json; the API decodes, uploads to Firebase Storage, stores URLs in
    Firestore (collection recommend_meal_images), and sets each meal's imageUrl to the public URL.
    """
    data = request.get_json(silent=True) or {}
    _log_recommend_api_request("recommend-meals", data)

    # Accept both spellings to be resilient with client payloads.
    dietary_preferences = data.get("dietary_preferences", data.get("dietry_pref", []))
    dietary_restrictions = data.get("dietary_restrictions", data.get("dietry_restr", []))
    # Performance-first defaults for mobile UX:
    # - include_images defaults to False (image generation is the slowest part)
    # - fast_mode defaults to True (single attempt/day, lower token budget; also shorter image prompts)
    # - image_quality: low|medium|high → DALL·E 2 256/512/1024 (default low for fastest images)
    # - RECOMMEND_MEALS_IMAGE_MAX_WORKERS (default 16) caps parallel DALL·E calls
    include_images = data.get("include_images", False)
    include_steps = data.get("include_steps", True)
    include_macros = data.get("include_macros", True)
    if include_steps is None:
        include_steps = True
    if include_macros is None:
        include_macros = True
    include_steps = bool(include_steps)
    include_macros = bool(include_macros)
    fast_mode = data.get("fast_mode", True)
    strict_calorie_alignment = data.get("strict_calorie_alignment", True)
    if strict_calorie_alignment is None:
        strict_calorie_alignment = True
    strict_calorie_alignment = bool(strict_calorie_alignment)
    _img_quality = str(data.get("image_quality") or "low").strip().lower()
    if _img_quality not in ("low", "medium", "high"):
        _img_quality = "low"
    recommend_model = os.getenv("RECOMMEND_MEALS_MODEL", "gpt-5.4-mini")
    num_meals = int(data.get("num_meals", 4))
    if num_meals < 1 or num_meals > 8:
        return jsonify({"error": "num_meals must be between 1 and 8"}), 400
    num_days = int(data.get("num_days", 7))
    if num_days < 1 or num_days > 30:
        return jsonify({"error": "num_days must be between 1 and 30"}), 400
    meal_keys = _get_meal_keys(num_meals)
    plan_id = data.get("plan_id") or str(uuid.uuid4())
    default_image_url = data.get("default_image_url") or os.getenv("DEFAULT_MEAL_IMAGE_URL", "")

    required_fields = [
        "goal", "gender", "age", "height_cm", "weight_kg",
        "target_weight_kg", "activity_level"
    ]
    missing = [f for f in required_fields if data.get(f) in (None, "")]
    if missing:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing,
        }), 400

    if not isinstance(dietary_preferences, list):
        return jsonify({"error": "dietary_preferences (or dietry_pref) must be an array"}), 400
    if not isinstance(dietary_restrictions, list):
        return jsonify({"error": "dietary_restrictions (or dietry_restr) must be an array"}), 400

    plant_diet_prompt = _dietary_restrictions_plant_based_prompt_block(dietary_restrictions)

    meal_split_percent = data.get("meal_split_percent")
    if not isinstance(meal_split_percent, dict):
        even = round(1.0 / len(meal_keys), 4)
        meal_split_percent = {k: even for k in meal_keys}
    for k in meal_keys:
        if k not in meal_split_percent:
            meal_split_percent[k] = 0.0

    # Estimate daily calorie/macronutrient targets before generation.
    def _calculate_daily_targets(payload: dict) -> dict:
        goal = str(payload.get("goal", "maintain")).lower()
        gender = str(payload.get("gender", "other")).lower()
        age = float(payload.get("age", 25))
        height_cm = float(payload.get("height_cm", 170))
        weight_kg = float(payload.get("weight_kg", 70))
        target_weight_kg = float(payload.get("target_weight_kg", weight_kg))
        activity_level = str(payload.get("activity_level", "lightly_active")).lower()

        # Mifflin-St Jeor BMR constants
        if gender == "male":
            s = 5
        elif gender == "female":
            s = -161
        else:
            s = -78  # midpoint approximation for non-binary/other

        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + s

        activity_multipliers = {
            "sedentary": 1.2,
            "not_active": 1.2,
            "lightly_active": 1.375,
            "moderately_active": 1.55,
            "active": 1.55,
            "very_active": 1.725,
        }
        tdee = bmr * activity_multipliers.get(activity_level, 1.375)

        # Goal adjustment
        if goal == "fat_loss":
            calories = tdee - 450
        elif goal == "muscle_gain":
            calories = tdee + 300
        else:
            calories = tdee

        # Sensible floor
        min_cals = 1500 if gender == "male" else 1200
        calories = max(calories, min_cals)

        # Macro split
        # Protein scaled by goal (g/kg), fat baseline 0.8 g/kg, carbs fill remainder.
        protein_per_kg = 1.8 if goal == "fat_loss" else (2.0 if goal == "muscle_gain" else 1.6)
        protein_g = protein_per_kg * target_weight_kg
        fat_g = 0.8 * target_weight_kg

        protein_kcal = protein_g * 4
        fat_kcal = fat_g * 9
        carbs_kcal = max(calories - protein_kcal - fat_kcal, calories * 0.25)
        carbs_g = carbs_kcal / 4

        return {
            "calories": round(calories),
            "protein_g": round(protein_g),
            "carbs_g": round(carbs_g),
            "fat_g": round(fat_g),
            "bmr": round(bmr),
            "tdee": round(tdee),
        }

    target_macros = data.get("target_daily_intake")
    if target_macros is None:
        target_macros = _calculate_daily_targets(data)
    elif not isinstance(target_macros, dict):
        return jsonify({"error": "target_daily_intake must be an object when provided"}), 400

    def _to_float(v):
        try:
            return float(v)
        except Exception:
            return 0.0

    target_macros = {
        "calories": round(_to_float(target_macros.get("calories", 0)), 0),
        "protein_g": round(_to_float(target_macros.get("protein_g", 0)), 1),
        "carbs_g": round(_to_float(target_macros.get("carbs_g", 0)), 1),
        "fat_g": round(_to_float(target_macros.get("fat_g", 0)), 1),
        "bmr": round(_to_float(target_macros.get("bmr", 0)), 0) if "bmr" in target_macros else None,
        "tdee": round(_to_float(target_macros.get("tdee", 0)), 0) if "tdee" in target_macros else None,
    }

    meal_schema_keys = ", ".join([f"\"{k}\": {{...}}" for k in meal_keys])
    meal_names = ", ".join(meal_keys)
    meal_fields_parts = ["name", "description", "ingredients (array)", "cook_time_min (number)", "cuisine", "meal_type"]
    if include_macros:
        meal_fields_parts = [
            "name",
            "description",
            "calories",
            "protein_g",
            "carbs_g",
            "fat_g",
            "ingredients (array)",
            "cook_time_min (number)",
            "cuisine",
            "meal_type",
        ]
    if include_steps:
        meal_fields_parts.append("steps (array)")
    meal_fields_line = ", ".join(meal_fields_parts)
    steps_rule = (
        "Include steps array per meal (max 4 short steps each)."
        if include_steps
        else "Do NOT include a steps field for any meal."
    )
    macros_rule = (
        "Include per-meal calories, protein_g, carbs_g, fat_g (numeric)."
        if include_macros
        else "Do NOT include calories, protein_g, carbs_g, or fat_g on any meal."
    )

    system_prompt = (
        "You are an expert nutritionist and meal planner.\n"
        "Return ONLY valid JSON (no markdown, no commentary).\n"
        f"Build a {num_days}-day plan with {num_meals} meals per day using these keys: {meal_names}.\n"
        "Meals must align with the user's goal, demographics, body stats, activity level, dietary preferences, and dietary restrictions.\n"
        f"{plant_diet_prompt}"
        "Keep meals practical and realistic.\n"
        f"For each meal include ONLY these fields: {meal_fields_line}.\n"
        f"{macros_rule}\n"
        f"{steps_rule}\n\n"
        "Output schema:\n"
        "{\n"
        '  "days": [\n'
        "    {\n"
        '      "day": 1,\n'
        f"      {meal_schema_keys}\n"
        "    }\n"
        "  ],\n"
        '  "daily_macro_targets": {\n'
        '    "calories": 0,\n'
        '    "protein_g": 0,\n'
        '    "carbs_g": 0,\n'
        '    "fat_g": 0\n'
        "  },\n"
        '  "notes": []\n'
        "}\n\n"
        "Rules:\n"
        f'- Exactly {num_days} day objects in "days".\n'
        f"- Day values must be 1..{num_days}.\n"
        "- Use numeric values for cook_time_min, and for calories/protein_g/carbs_g/fat_g when included.\n"
        "- meal_type must be one of: Breakfast, Lunch, Dinner, Snack (mapped from meal key).\n"
        "- Keep output compact: max 8 ingredients and max 4 short steps per meal (when steps are included).\n"
        "- Keep description to a single short line.\n"
        "- Ensure unique meal names across different days in this same weekly plan.\n"
        "- When asked for a single day, return ONLY that day as one object; never return a top-level \"days\" array.\n"
        "- Return valid JSON only.\n"
    )

    user_payload = {
        "plan_id": plan_id,
        "goal": data.get("goal"),
        "gender": data.get("gender"),
        "age": data.get("age"),
        "height_cm": data.get("height_cm"),
        "weight_kg": data.get("weight_kg"),
        "target_weight_kg": data.get("target_weight_kg"),
        "activity_level": data.get("activity_level"),
        "dietary_preferences": dietary_preferences,
        "dietary_restrictions": dietary_restrictions,
        "target_daily_intake": target_macros,
        "meal_split_percent": meal_split_percent,
        "num_days": num_days,
        "num_meals": num_meals,
        "meal_keys": meal_keys,
        "include_steps": include_steps,
        "include_macros": include_macros,
    }

    response_text = None
    try:
        def _generate_day_plan(day_number: int, avoid_names: list[str]) -> tuple[dict | None, str | None]:
            """
            Generate one day's plan. Returns (day_obj, raw_text_on_failure).
            """
            max_attempts = 1 if fast_mode else 2
            if not include_steps and not include_macros:
                max_tokens = 700 if fast_mode else 1000
            elif not include_steps or not include_macros:
                max_tokens = 900 if fast_mode else 1300
            else:
                max_tokens = 1100 if fast_mode else 1600
            for attempt in range(max_attempts):
                no_repeat_instruction = ""
                if avoid_names:
                    no_repeat_instruction = (
                        "Avoid reusing these meal names from prior days in this same weekly plan: "
                        + ", ".join(avoid_names[:120])
                        + ". Use distinct/new meal names for this day."
                    )
                day_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                    {
                        "role": "user",
                        "content": (
                            f"Generate ONLY day {day_number} as ONE JSON object with keys: "
                            f"\"day\" (number {day_number}), {meal_schema_keys}. "
                            "Put breakfast/lunch/dinner/etc. at the ROOT of that object. "
                            "Do NOT wrap the day inside a \"days\" array or add weekly-plan wrappers."
                        ),
                    },
                    {"role": "user", "content": no_repeat_instruction},
                ]
                if attempt == 1:
                    day_messages.append({
                        "role": "user",
                        "content": "Regenerate compactly with very short steps and ensure fully valid JSON.",
                    })

                budget = _recommend_llm_completion_budget(recommend_model, max_tokens)
                completion = client.chat.completions.create(
                    model=recommend_model,
                    messages=day_messages,
                    temperature=0.1 if fast_mode else 0.2,
                    **_chat_completion_limit_kw(recommend_model, budget),
                    timeout=25 if fast_mode else 40,
                    response_format={"type": "json_object"},
                )
                raw = _chat_response_message_text(completion)
                if not raw and _model_prefers_completion_token_param(recommend_model):
                    completion = client.chat.completions.create(
                        model=recommend_model,
                        messages=day_messages,
                        temperature=0.1 if fast_mode else 0.2,
                        **_chat_completion_limit_kw(recommend_model, max(budget, 8192)),
                        timeout=60,
                    )
                    raw = _chat_response_message_text(completion)
                try:
                    day_obj = _force_json(raw)
                    if not isinstance(day_obj, dict):
                        continue
                    day_obj = _unwrap_recommend_day_payload(day_obj, meal_keys, day_number)
                    day_obj["day"] = day_obj.get("day", day_number)
                    return day_obj, None
                except Exception:
                    if attempt == (max_attempts - 1):
                        return None, raw
            return None, None

        # Sequential generation so each day can avoid meals from prior generated days.
        day_results = {}
        day_failures = []
        used_meal_names = set()
        for day_number in range(1, num_days + 1):
            try:
                day_obj, failed_raw = _generate_day_plan(day_number, sorted(used_meal_names))
                if day_obj is None:
                    day_failures.append({
                        "day": day_number,
                        "error": "Invalid JSON for day plan",
                        "raw_response": (failed_raw or "")[:500],
                    })
                    continue
                day_results[day_number] = day_obj
                for mk in meal_keys:
                    name = str(day_obj.get(mk, {}).get("name", "")).strip().lower()
                    if name:
                        used_meal_names.add(name)
            except Exception as e:
                day_failures.append({
                    "day": day_number,
                    "error": str(e),
                })

        if day_failures:
            return jsonify({
                "error": "Failed to generate weekly meal plan",
                "user_message": "We couldn't generate your weekly meal plan right now. Please try again.",
                "details": "One or more day plans failed to generate.",
                "failed_days": day_failures,
            }), 500

        days = [day_results[d] for d in range(1, num_days + 1)]
        plan = {"days": {}}

        required_meal_keys = meal_keys
        for i, day_obj in enumerate(days, start=1):
            if not isinstance(day_obj, dict):
                return jsonify({"error": f"Invalid day object at index {i-1}"}), 500
            day_obj["day"] = day_obj.get("day", i)
            for mk in required_meal_keys:
                if mk not in day_obj or not isinstance(day_obj.get(mk), dict):
                    day_obj[mk] = {}
                meal = day_obj[mk]
                meal.setdefault("name", "")
                meal.setdefault("description", "")
                if include_macros:
                    meal.setdefault("calories", 0)
                    meal.setdefault("protein_g", 0)
                    meal.setdefault("carbs_g", 0)
                    meal.setdefault("fat_g", 0)
                else:
                    meal.pop("calories", None)
                    meal.pop("protein_g", None)
                    meal.pop("carbs_g", None)
                    meal.pop("fat_g", None)
                meal.setdefault("ingredients", [])
                if include_steps:
                    meal.setdefault("steps", [])
                else:
                    meal.pop("steps", None)
                meal.setdefault("cook_time_min", 0)
                meal.setdefault("cuisine", "")
                meal.setdefault("imageUrl", default_image_url if include_images else None)
                meal["meal_type"] = _meal_type_from_key(mk)

            # Response shape requested: day number as key, details as value.
            day_payload = dict(day_obj)
            day_payload.pop("day", None)
            # Drop plan-level keys the model sometimes nests inside a "day" by mistake.
            day_payload.pop("days", None)
            day_payload.pop("daily_macro_targets", None)
            plan["days"][str(i)] = day_payload

        if include_images:
            meal_jobs: list[tuple[int, str, dict]] = []
            for day_i, day_obj in enumerate(days, start=1):
                for mk in required_meal_keys:
                    meal_jobs.append((day_i, mk, day_obj.get(mk, {})))

            _img_workers = _recommend_meals_image_pool_size(len(meal_jobs))
            _img_prompt_fast = bool(fast_mode) or (_img_quality == "low")

            _image_errors: list[str] = []

            def _generate_job_image_b64(job: tuple[int, str, dict]) -> str | None:
                day_i, mk, meal_obj = job
                b64 = _generate_meal_image_b64(
                    meal_obj.get("name", "Meal"),
                    meal_obj.get("description", ""),
                    fast_mode=_img_prompt_fast,
                    log_prefix="recommend-meals",
                    log_context=f"day={day_i} meal={mk}",
                )
                if b64 is None:
                    _image_errors.append(f"day={day_i} meal={mk}: generation failed")
                return b64

            with ThreadPoolExecutor(max_workers=_img_workers) as img_executor:
                futures = {img_executor.submit(_generate_job_image_b64, job): job for job in meal_jobs}
                raw_quads: list[tuple[int, str, dict, str | None]] = []
                for fut, job in futures.items():
                    day_i, mk, meal_obj = job
                    try:
                        raw_quads.append((day_i, mk, meal_obj, fut.result()))
                    except Exception as e:
                        print(f"[recommend-meals] future error day={day_i} meal={mk}: {e}")
                        raw_quads.append((day_i, mk, meal_obj, None))
            persist_workers = min(max(1, len(raw_quads)), _img_workers)

            def _persist_recommend_image(
                quad: tuple[int, str, dict, str | None],
            ) -> str:
                day_i, mk, _meal_obj, b64 = quad
                return _recommend_meals_finalize_image_from_b64(
                    b64, default_image_url, plan_id, day_i, mk
                )

            with ThreadPoolExecutor(max_workers=persist_workers) as ex_persist:
                finals = list(ex_persist.map(_persist_recommend_image, raw_quads))
            image_success_count = 0
            for (_day_i, _mk, meal_obj, _b64), final_u in zip(raw_quads, finals):
                meal_obj["imageUrl"] = final_u
                if final_u and final_u != default_image_url:
                    image_success_count += 1
            plan["image_generation_summary"] = {
                "requested": len(meal_jobs),
                "succeeded": image_success_count,
                "failed": len(meal_jobs) - image_success_count,
                "errors": _image_errors[:10],  # cap to avoid huge responses
            }

        def _sum_plan_macros():
            total_cal = total_pro = total_carb = total_fat = 0.0
            for day_obj in days:
                for mk in required_meal_keys:
                    meal = day_obj.get(mk, {})
                    total_cal += float(meal.get("calories", 0) or 0)
                    total_pro += float(meal.get("protein_g", 0) or 0)
                    total_carb += float(meal.get("carbs_g", 0) or 0)
                    total_fat += float(meal.get("fat_g", 0) or 0)
            return total_cal, total_pro, total_carb, total_fat

        # Derive average per-day macros from generated meals.
        total_cal, total_pro, total_carb, total_fat = _sum_plan_macros() if include_macros else (0.0, 0.0, 0.0, 0.0)
        target_daily_cal = float(target_macros["calories"])
        estimated_daily_cal = (total_cal / float(num_days)) if total_cal else 0.0
        lower_bound = target_daily_cal * 0.9
        upper_bound = target_daily_cal * 1.1
        correction_applied = False
        correction_scale = 1.0

        # Correction pass: keep generated daily calories within +/-10% of target
        # by scaling meal portions/macros proportionally.
        if include_macros and strict_calorie_alignment and estimated_daily_cal > 0 and (estimated_daily_cal < lower_bound or estimated_daily_cal > upper_bound):
            correction_applied = True
            correction_scale = target_daily_cal / estimated_daily_cal
            # Avoid extreme scaling.
            correction_scale = max(0.7, min(correction_scale, 1.5))

            for day_obj in days:
                for mk in required_meal_keys:
                    meal = day_obj.get(mk, {})
                    meal["calories"] = round(float(meal.get("calories", 0) or 0) * correction_scale, 1)
                    meal["protein_g"] = round(float(meal.get("protein_g", 0) or 0) * correction_scale, 1)
                    meal["carbs_g"] = round(float(meal.get("carbs_g", 0) or 0) * correction_scale, 1)
                    meal["fat_g"] = round(float(meal.get("fat_g", 0) or 0) * correction_scale, 1)
                    meal["portion_scale"] = round(correction_scale, 3)

            total_cal, total_pro, total_carb, total_fat = _sum_plan_macros()

        plan["plan_id"] = plan_id
        plan["num_days"] = num_days
        plan["num_meals"] = num_meals
        plan["include_steps"] = include_steps
        plan["include_macros"] = include_macros
        plan["meal_split_percent"] = meal_split_percent
        plan["daily_macro_targets"] = {
            "calories": target_macros["calories"],
            "protein_g": target_macros["protein_g"],
            "carbs_g": target_macros["carbs_g"],
            "fat_g": target_macros["fat_g"],
            "estimated_from_generated_plan": {
                "calories": round(total_cal / float(num_days), 1),
                "protein_g": round(total_pro / float(num_days), 1),
                "carbs_g": round(total_carb / float(num_days), 1),
                "fat_g": round(total_fat / float(num_days), 1),
            },
            "calorie_alignment": {
                "target_window": {
                    "min": round(lower_bound, 1),
                    "max": round(upper_bound, 1),
                },
                "correction_applied": correction_applied,
                "portion_scale": round(correction_scale, 3),
            },
        }
        plan["target_calories_by_day"] = [
            {"day": day_num, "target_calories": target_macros["calories"]}
            for day_num in range(1, num_days + 1)
        ]
        plan["metabolism"] = {
            "bmr": target_macros["bmr"],
            "tdee": target_macros["tdee"],
        }
        plan.setdefault("notes", [])

        return jsonify(plan)
    except Exception as e:
        return jsonify({
            "error": "Failed to generate weekly meal plan",
            "user_message": "We couldn't generate your weekly meal plan right now. Please try again.",
            "details": str(e),
            "raw_response": response_text[:500] if response_text else None,
        }), 500


@app.route("/recommend-meals/day", methods=["POST"])
def recommend_meals_day():
    """
    Generate one day meal plan (breakfast/lunch/dinner/snacks).
    Designed for client-side parallel calls (day 1..7) with merge on client.
    When include_images is true, image_quality may be low|medium|high (default low = 256px DALL·E).
    Images use response_format=b64_json; the API decodes and uploads to Firebase Storage, stores URLs in
    Firestore (collection recommend_meal_images), and sets each meal's imageUrl to the public URL.
    """
    data = request.get_json(silent=True) or {}
    _log_recommend_api_request("recommend-meals/day", data)

    dietary_preferences = data.get("dietary_preferences", data.get("dietry_pref", []))
    dietary_restrictions = data.get("dietary_restrictions", data.get("dietry_restr", []))
    include_images = data.get("include_images", False)
    include_steps = data.get("include_steps", True)
    include_macros = data.get("include_macros", True)
    if include_steps is None:
        include_steps = True
    if include_macros is None:
        include_macros = True
    include_steps = bool(include_steps)
    include_macros = bool(include_macros)
    fast_mode = data.get("fast_mode", True)
    strict_calorie_alignment = data.get("strict_calorie_alignment", True)
    _img_quality = str(data.get("image_quality") or "low").strip().lower()
    if _img_quality not in ("low", "medium", "high"):
        _img_quality = "low"
    recommend_model = os.getenv("RECOMMEND_MEALS_MODEL", "gpt-5.4-mini")
    num_meals = int(data.get("num_meals", 4))
    if num_meals < 1 or num_meals > 8:
        return jsonify({"error": "num_meals must be between 1 and 8"}), 400
    meal_keys = _get_meal_keys(num_meals)
    plan_id = data.get("plan_id") or str(uuid.uuid4())
    day_number = int(data.get("day_number", 1))
    if day_number < 1 or day_number > 7:
        return jsonify({"error": "day_number must be between 1 and 7"}), 400

    default_image_url = data.get("default_image_url") or os.getenv("DEFAULT_MEAL_IMAGE_URL", "")

    required_fields = [
        "goal", "gender", "age", "height_cm", "weight_kg",
        "target_weight_kg", "activity_level"
    ]
    missing = [f for f in required_fields if data.get(f) in (None, "")]
    if missing:
        return jsonify({"error": "Missing required fields", "missing_fields": missing}), 400
    if not isinstance(dietary_preferences, list):
        return jsonify({"error": "dietary_preferences (or dietry_pref) must be an array"}), 400
    if not isinstance(dietary_restrictions, list):
        return jsonify({"error": "dietary_restrictions (or dietry_restr) must be an array"}), 400

    plant_diet_prompt = _dietary_restrictions_plant_based_prompt_block(dietary_restrictions)

    # Optional split to track calorie distribution per meal
    meal_split_percent = data.get("meal_split_percent")
    if not isinstance(meal_split_percent, dict):
        # default even split across requested meal count
        even = round(1.0 / len(meal_keys), 4)
        meal_split_percent = {k: even for k in meal_keys}
    for k in meal_keys:
        if k not in meal_split_percent:
            meal_split_percent[k] = 0.0

    # Reuse same target estimation logic as weekly endpoint when target is not provided.
    def _calculate_daily_targets(payload: dict) -> dict:
        goal = str(payload.get("goal", "maintain")).lower()
        gender = str(payload.get("gender", "other")).lower()
        age = float(payload.get("age", 25))
        height_cm = float(payload.get("height_cm", 170))
        weight_kg = float(payload.get("weight_kg", 70))
        target_weight_kg = float(payload.get("target_weight_kg", weight_kg))
        activity_level = str(payload.get("activity_level", "lightly_active")).lower()

        s = 5 if gender == "male" else (-161 if gender == "female" else -78)
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + s
        activity_multipliers = {
            "sedentary": 1.2,
            "not_active": 1.2,
            "lightly_active": 1.375,
            "moderately_active": 1.55,
            "active": 1.55,
            "very_active": 1.725,
        }
        tdee = bmr * activity_multipliers.get(activity_level, 1.375)
        if goal == "fat_loss":
            calories = tdee - 450
        elif goal == "muscle_gain":
            calories = tdee + 300
        else:
            calories = tdee
        min_cals = 1500 if gender == "male" else 1200
        calories = max(calories, min_cals)
        protein_per_kg = 1.8 if goal == "fat_loss" else (2.0 if goal == "muscle_gain" else 1.6)
        protein_g = protein_per_kg * target_weight_kg
        fat_g = 0.8 * target_weight_kg
        protein_kcal = protein_g * 4
        fat_kcal = fat_g * 9
        carbs_kcal = max(calories - protein_kcal - fat_kcal, calories * 0.25)
        carbs_g = carbs_kcal / 4
        return {
            "calories": round(calories),
            "protein_g": round(protein_g),
            "carbs_g": round(carbs_g),
            "fat_g": round(fat_g),
            "bmr": round(bmr),
            "tdee": round(tdee),
        }

    # Important: if client passes target_daily_intake, do NOT recalculate it.
    # Only calculate when it is missing.
    target_daily_intake = data.get("target_daily_intake")
    if target_daily_intake is None:
        target_daily_intake = _calculate_daily_targets(data)
    elif not isinstance(target_daily_intake, dict):
        return jsonify({"error": "target_daily_intake must be an object when provided"}), 400

    # Safe numeric casting (handles strings coming from client).
    def _to_float(v):
        try:
            return float(v)
        except Exception:
            return 0.0

    target_daily_intake = {
        "calories": round(_to_float(target_daily_intake.get("calories", 0)), 0),
        "protein_g": round(_to_float(target_daily_intake.get("protein_g", 0)), 1),
        "carbs_g": round(_to_float(target_daily_intake.get("carbs_g", 0)), 1),
        "fat_g": round(_to_float(target_daily_intake.get("fat_g", 0)), 1),
        "bmr": round(_to_float(target_daily_intake.get("bmr", 0)), 0) if "bmr" in target_daily_intake else None,
        "tdee": round(_to_float(target_daily_intake.get("tdee", 0)), 0) if "tdee" in target_daily_intake else None,
    }

    target_daily_cal = float(target_daily_intake.get("calories", 0) or 0)

    day_meal_schema_keys = ", ".join([f"\"{k}\": {{...}}" for k in meal_keys])
    day_meal_names = ", ".join(meal_keys)

    with _plan_meal_history_lock:
        prior_meal_names = sorted(_plan_meal_name_history.get(plan_id, set()))

    meal_fields_parts = ["name", "description", "ingredients (array)", "cook_time_min (number)", "cuisine", "meal_type"]
    if include_macros:
        meal_fields_parts = [
            "name",
            "description",
            "calories",
            "protein_g",
            "carbs_g",
            "fat_g",
            "ingredients (array)",
            "cook_time_min (number)",
            "cuisine",
            "meal_type",
        ]
    if include_steps:
        meal_fields_parts.append("steps (array)")
    meal_fields_line = ", ".join(meal_fields_parts)

    steps_rule = (
        "Include steps array per meal (max 4 short steps each)."
        if include_steps
        else "Do NOT include a steps field for any meal."
    )
    macros_rule = (
        "Include per-meal calories, protein_g, carbs_g, fat_g (numeric)."
        if include_macros
        else "Do NOT include calories, protein_g, carbs_g, or fat_g on any meal."
    )

    no_repeat_rule = ""
    if prior_meal_names:
        no_repeat_rule = (
            "For this plan_id, avoid reusing meal names from previous days: "
            + ", ".join(prior_meal_names[:120])
            + ". Use distinct/new meal names for this day."
        )

    system_prompt = f"""You are an expert nutritionist and meal planner.
Return ONLY valid JSON.
Generate ONE day meal plan with exactly {num_meals} meals using these keys: {day_meal_names}.
Meals must align with the user's goal, demographics, body stats, activity level, dietary preferences, and dietary restrictions.
{plant_diet_prompt}For each meal include ONLY these fields: {meal_fields_line}.
{macros_rule}
{steps_rule}
{no_repeat_rule}
Keep compact: max 8 ingredients per meal.
"""

    user_payload = {
        "plan_id": plan_id,
        "day_number": day_number,
        "goal": data.get("goal"),
        "gender": data.get("gender"),
        "age": data.get("age"),
        "height_cm": data.get("height_cm"),
        "weight_kg": data.get("weight_kg"),
        "target_weight_kg": data.get("target_weight_kg"),
        "activity_level": data.get("activity_level"),
        "dietary_preferences": dietary_preferences,
        "dietary_restrictions": dietary_restrictions,
        "target_daily_intake": target_daily_intake,
        "meal_split_percent": meal_split_percent,
        "num_meals": num_meals,
        "meal_keys": meal_keys,
        "include_steps": include_steps,
        "include_macros": include_macros,
        "avoid_meal_names": prior_meal_names,
    }

    response_text = None
    try:
        max_attempts = 1 if fast_mode else 2
        if not include_steps and not include_macros:
            max_tokens = 700 if fast_mode else 1000
        elif not include_steps or not include_macros:
            max_tokens = 900 if fast_mode else 1300
        else:
            max_tokens = 1100 if fast_mode else 1600
        day_obj = None
        for attempt in range(max_attempts):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                {"role": "user", "content": f"Return JSON schema: {{\"day\": {day_number}, {day_meal_schema_keys}}}"},
            ]
            if attempt == 1:
                messages.append({"role": "user", "content": "Regenerate with stricter compactness and valid JSON."})
            budget = _recommend_llm_completion_budget(recommend_model, max_tokens)
            completion = client.chat.completions.create(
                model=recommend_model,
                messages=messages,
                temperature=0.1 if fast_mode else 0.2,
                **_chat_completion_limit_kw(recommend_model, budget),
                timeout=25 if fast_mode else 40,
                response_format={"type": "json_object"},
            )
            response_text = _chat_response_message_text(completion)
            if not response_text and _model_prefers_completion_token_param(recommend_model):
                completion = client.chat.completions.create(
                    model=recommend_model,
                    messages=messages,
                    temperature=0.1 if fast_mode else 0.2,
                    **_chat_completion_limit_kw(recommend_model, max(budget, 8192)),
                    timeout=60,
                )
                response_text = _chat_response_message_text(completion)
            try:
                parsed = _force_json(response_text)
                if isinstance(parsed, dict):
                    day_obj = parsed
                    break
            except Exception:
                day_obj = None

        if day_obj is None:
            raise ValueError("Could not parse one-day JSON output")

        day_obj["day"] = day_number
        required_meal_keys = meal_keys
        for mk in required_meal_keys:
            if mk not in day_obj or not isinstance(day_obj.get(mk), dict):
                day_obj[mk] = {}
            meal = day_obj[mk]
            meal.setdefault("name", "")
            meal.setdefault("description", "")
            if include_macros:
                meal.setdefault("calories", 0)
                meal.setdefault("protein_g", 0)
                meal.setdefault("carbs_g", 0)
                meal.setdefault("fat_g", 0)
            else:
                meal.pop("calories", None)
                meal.pop("protein_g", None)
                meal.pop("carbs_g", None)
                meal.pop("fat_g", None)
            meal.setdefault("ingredients", [])
            if include_steps:
                meal.setdefault("steps", [])
            else:
                meal.pop("steps", None)
            meal.setdefault("cook_time_min", 0)
            meal.setdefault("cuisine", "")
            meal.setdefault("imageUrl", default_image_url if include_images else None)
            meal["meal_type"] = _meal_type_from_key(mk)

        # Update per-plan de-dup history with generated meal names.
        generated_names = {
            str(day_obj.get(k, {}).get("name", "")).strip().lower()
            for k in required_meal_keys
            if str(day_obj.get(k, {}).get("name", "")).strip()
        }
        if generated_names:
            with _plan_meal_history_lock:
                _plan_meal_name_history[plan_id].update(generated_names)

        if include_images:
            _img_workers = _recommend_meals_image_pool_size(len(required_meal_keys))
            _img_prompt_fast = bool(fast_mode) or (_img_quality == "low")

            def _generate_day_meal_image_b64(meal_obj: dict) -> str | None:
                return _generate_meal_image_b64(
                    meal_obj.get("name", "Meal"),
                    meal_obj.get("description", ""),
                    fast_mode=_img_prompt_fast,
                    log_prefix="recommend-meals/day",
                )

            with ThreadPoolExecutor(max_workers=_img_workers) as ex:
                futs = {
                    k: ex.submit(_generate_day_meal_image_b64, day_obj.get(k, {}))
                    for k in required_meal_keys
                }
                raws: list[tuple[str, str | None]] = []
                for k in required_meal_keys:
                    try:
                        raws.append((k, futs[k].result()))
                    except Exception:
                        raws.append((k, None))
            persist_workers = min(max(1, len(raws)), _img_workers)

            def _persist_day_image(pair: tuple[str, str | None]) -> str:
                meal_key, b64 = pair
                return _recommend_meals_finalize_image_from_b64(
                    b64, default_image_url, plan_id, day_number, meal_key
                )

            with ThreadPoolExecutor(max_workers=persist_workers) as ex_persist:
                finals = list(ex_persist.map(_persist_day_image, raws))
            for k, u in zip([x[0] for x in raws], finals):
                day_obj[k]["imageUrl"] = u

        # Day totals + optional correction alignment to target calories (requires per-meal macros)
        totals = {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}
        if include_macros:
            for k in required_meal_keys:
                m = day_obj.get(k, {})
                totals["calories"] += float(m.get("calories", 0) or 0)
                totals["protein_g"] += float(m.get("protein_g", 0) or 0)
                totals["carbs_g"] += float(m.get("carbs_g", 0) or 0)
                totals["fat_g"] += float(m.get("fat_g", 0) or 0)

        lower = target_daily_cal * 0.9
        upper = target_daily_cal * 1.1
        correction_applied = False
        portion_scale = 1.0
        if (
            include_macros
            and strict_calorie_alignment
            and totals["calories"] > 0
            and (totals["calories"] < lower or totals["calories"] > upper)
        ):
            correction_applied = True
            portion_scale = max(0.7, min(target_daily_cal / totals["calories"], 1.5))
            for k in required_meal_keys:
                m = day_obj.get(k, {})
                m["calories"] = round(float(m.get("calories", 0) or 0) * portion_scale, 1)
                m["protein_g"] = round(float(m.get("protein_g", 0) or 0) * portion_scale, 1)
                m["carbs_g"] = round(float(m.get("carbs_g", 0) or 0) * portion_scale, 1)
                m["fat_g"] = round(float(m.get("fat_g", 0) or 0) * portion_scale, 1)
                m["portion_scale"] = round(portion_scale, 3)
            totals = {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}
            for k in required_meal_keys:
                m = day_obj.get(k, {})
                totals["calories"] += float(m.get("calories", 0) or 0)
                totals["protein_g"] += float(m.get("protein_g", 0) or 0)
                totals["carbs_g"] += float(m.get("carbs_g", 0) or 0)
                totals["fat_g"] += float(m.get("fat_g", 0) or 0)

        response = {
            "plan_id": plan_id,
            "day_number": day_number,
            "include_steps": include_steps,
            "include_macros": include_macros,
            "day": day_obj,
            "target_daily_intake": {
                "calories": target_daily_intake.get("calories", 0),
                "protein_g": target_daily_intake.get("protein_g", 0),
                "carbs_g": target_daily_intake.get("carbs_g", 0),
                "fat_g": target_daily_intake.get("fat_g", 0),
            },
            "daily_totals": {
                "calories": round(totals["calories"], 1),
                "protein_g": round(totals["protein_g"], 1),
                "carbs_g": round(totals["carbs_g"], 1),
                "fat_g": round(totals["fat_g"], 1),
            },
            "variance_from_target": {
                "calories": round(totals["calories"] - target_daily_cal, 1),
                "calories_pct": round(((totals["calories"] - target_daily_cal) / target_daily_cal) * 100, 1) if target_daily_cal else 0,
                "target_window": {"min": round(lower, 1), "max": round(upper, 1)},
                "correction_applied": correction_applied,
                "portion_scale": round(portion_scale, 3),
            },
            "meal_split_percent": meal_split_percent,
            "metabolism": {
                "bmr": target_daily_intake.get("bmr"),
                "tdee": target_daily_intake.get("tdee"),
            },
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({
            "error": "Failed to generate day meal plan",
            "user_message": "We couldn't generate your day meal plan right now. Please try again.",
            "details": str(e),
            "raw_response": response_text[:500] if response_text else None,
        }), 500

#####video only code starts here#####

# ---------------------------
# Helper: Transcribe Audio
# ---------------------------
def transcribe_audio_file(
    audio_bytes: bytes,
    file_name: str,
    *,
    language: str | None = None,
) -> str:
    """
    Whisper transcription. Optional `language` (ISO-639-1, e.g. en) skips language detection and can reduce latency.
    """
    try:
        # Create file-like object
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = file_name

        t_kwargs: dict = {
            "model": "whisper-1",  # Cost-effective: Much cheaper than gpt-4o-transcribe, same quality
            "file": audio_file,
            "response_format": "text",
        }
        if language:
            t_kwargs["language"] = language
        transcription = client.audio.transcriptions.create(**t_kwargs)

        transcript_text = (
            transcription if isinstance(transcription, str) else str(transcription)
        ).strip()

        if not transcript_text:
            raise ValueError("Transcription returned empty text")

        return transcript_text

    except Exception as e:
        raise Exception(f"Transcription failed: {e}")


# ---------------------------
# Helper: Extract Ingredients
# ---------------------------
def extract_ingredients(transcript: str):
    try:
        system_prompt = (
            "You are a food ingredient extraction assistant.\n"
            "Given a sentence, extract ONLY the food ingredients listed.\n"
            "Return them as a comma-separated list with NO extra explanations.\n"
            "Example:\n"
            "Input: 'I have chicken breast, broccoli and garlic.'\n"
            "Output: 'chicken breast, broccoli, garlic'\n"
        )

        user_prompt = f"Extract ingredients from: \"{transcript}\""

        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # small + fast + cheap
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        content = completion.choices[0].message.content.strip()

        # Comma-split + cleanup
        raw_items = content.replace("\n", ",").split(",")
        ingredients = [
            item.strip()
            for item in raw_items
            if item.strip()
            and ":" not in item.lower()
            and "sorry" not in item.lower()
        ]

        # Remove duplicates while keeping order
        seen = set()
        unique = []
        for ing in ingredients:
            low = ing.lower()
            if low not in seen:
                seen.add(low)
                unique.append(ing)

        return unique

    except Exception as e:
        raise Exception(f"Ingredient extraction failed: {e}")


def _voice_logging_completion_budget(*, fast: bool) -> int:
    """Shared max output tokens for voice + photo food logging LLM calls."""
    default_cap = 500 if fast else 800
    try:
        max_out = int(os.getenv("VOICE_LOGGING_MAX_COMPLETION_TOKENS", str(default_cap)))
    except ValueError:
        max_out = default_cap
    if fast:
        max_out = min(max_out, 600)
    return max(256, min(max_out, 4000))


def _normalize_food_log_meals_from_obj(obj: dict) -> list[dict]:
    """Normalize LLM JSON `meals` array into the /food-logging meal shape."""
    meals = obj.get("meals", [])
    if not isinstance(meals, list):
        return []
    out: list[dict] = []
    for m in meals:
        if not isinstance(m, dict):
            continue
        try:
            cals = float(m.get("calories", 0) or 0)
        except (TypeError, ValueError):
            cals = 0.0

        def _nf(key: str) -> float:
            try:
                return float(m.get(key, 0) or 0)
            except (TypeError, ValueError):
                return 0.0

        out.append({
            "name": str(m.get("name", "Meal")).strip() or "Meal",
            "description": str(m.get("description", "")).strip(),
            "meal_type": str(m.get("meal_type", "Unknown")).strip() or "Unknown",
            "calories": int(round(cals)),
            "protein_g": round(_nf("protein_g"), 1),
            "carbs_g": round(_nf("carbs_g"), 1),
            "fat_g": round(_nf("fat_g"), 1),
        })
    return out


def extract_meals_from_voice_log_transcript(transcript: str, *, fast: bool = False) -> list[dict]:
    """
    Parse spoken food log into structured meals with estimated calories and macros.
    Tuned for latency: small completion budget, compact prompt.
    """
    model = os.getenv("VOICE_LOGGING_MODEL", "gpt-4o-mini")
    max_out = _voice_logging_completion_budget(fast=fast)

    system_prompt = (
        "Extract food log items from the transcript. Return ONLY JSON: "
        '{"meals":[{"name":"","description":"","meal_type":"Breakfast|Lunch|Dinner|Snack|Unknown",'
        '"calories":0,"protein_g":0,"carbs_g":0,"fat_g":0}]}. '
        "One entry per distinct meal/snack; estimate kcal and macros for typical portions. "
        "Non-food or empty transcript → {\"meals\":[]}. No markdown."
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript[:12000]},
        ],
        temperature=0.1,
        **_chat_completion_limit_kw(model, max_out),
        response_format={"type": "json_object"},
    )
    raw = completion.choices[0].message.content
    if not raw:
        return []
    obj = _force_json(raw)
    return _normalize_food_log_meals_from_obj(obj)


def extract_meals_from_food_photo(image_data_url: str, *, fast: bool = False) -> tuple[list[dict], str]:
    """
    Vision: identify food in a meal photo and return the same meal objects as voice/text logging.
    Second return value is a short summary for the `transcript` field (what was visible).
    """
    model = os.getenv("FOOD_LOGGING_PHOTO_MODEL") or os.getenv("VOICE_LOGGING_MODEL", "gpt-4o-mini")
    max_out = _voice_logging_completion_budget(fast=fast)

    system_prompt = (
        "You analyze photos of food for a meal logging app. Identify dishes and estimate portions. "
        "Return ONLY JSON: "
        '{"summary":"1–2 sentences describing what food is visible (or empty if none).","meals":['
        '{"name":"","description":"","meal_type":"Breakfast|Lunch|Dinner|Snack|Unknown",'
        '"calories":0,"protein_g":0,"carbs_g":0,"fat_g":0}]}. '
        "One meal entry per distinct dish; if one plate has several items, you may use one combined entry with a clear description. "
        "Estimate kcal and macros for likely portion sizes from the image. "
        "If the image shows no edible food, is too blurry, or is not food → {\"summary\":\"\",\"meals\":[]}. "
        "No markdown."
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this food photo and return the JSON."},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        temperature=0.1,
        **_chat_completion_limit_kw(model, max_out),
        response_format={"type": "json_object"},
    )
    raw = completion.choices[0].message.content
    if not raw:
        return [], ""
    obj = _force_json(raw)
    summary = str(obj.get("summary", "")).strip()
    meals = _normalize_food_log_meals_from_obj(obj)
    return meals, summary


def _multipart_upload_mimetype(uploaded_file) -> str | None:
    for attr in ("mimetype", "content_type"):
        v = getattr(uploaded_file, attr, None)
        if v:
            return str(v).strip()
    return None


def _looks_like_image_multipart_file(filename: str | None, mimetype: str | None) -> bool:
    """True if upload is likely an image (extension or Content-Type)."""
    mt = (mimetype or "").strip().lower()
    if mt.startswith("image/"):
        return True
    fn = (filename or "").lower()
    return fn.endswith(
        (".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".heif", ".bmp", ".tif", ".tiff"),
    )


def _content_type_for_image_filename(filename: str) -> str:
    fn = filename.lower()
    if fn.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if fn.endswith(".png"):
        return "image/png"
    if fn.endswith(".gif"):
        return "image/gif"
    if fn.endswith(".webp"):
        return "image/webp"
    if fn.endswith((".heic", ".heif")):
        return "image/heic"
    if fn.endswith((".bmp",)):
        return "image/bmp"
    if fn.endswith((".tif", ".tiff")):
        return "image/tiff"
    return "image/jpeg"


def _food_logging_image_data_url_from_request() -> tuple[str | None, str | None]:
    """
    Build a data: URL for vision from multipart or JSON (imageBase64 + imageFormat).

    Multipart field names tried in order: image, photo, file, audio.
    The audio key is accepted only when the part looks like an image (some clients reuse the voice field).

    Returns (data_url, None) or (None, error_message for 400 responses).
    """
    for key in ("image", "photo", "file", "audio"):
        if key not in request.files:
            continue
        uploaded_file = request.files.get(key)
        if not uploaded_file or not uploaded_file.filename:
            continue
        mt_part = _multipart_upload_mimetype(uploaded_file)
        if not _looks_like_image_multipart_file(uploaded_file.filename, mt_part):
            continue
        image_bytes = uploaded_file.read()
        if not image_bytes:
            return None, "Uploaded image file is empty"
        content_type = _content_type_for_image_filename(uploaded_file.filename)
        mt = (mt_part or "").strip().lower()
        if mt.startswith("image/"):
            content_type = mt
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{content_type};base64,{image_base64}", None

    if request.is_json:
        data = request.get_json(silent=True) or {}
        image_base64 = data.get("imageBase64") or data.get("image")
        if image_base64:
            if "," in str(image_base64):
                image_base64 = str(image_base64).split(",")[-1]
            image_format = (data.get("imageFormat") or "jpg").lower()
            format_to_mime = {
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "png": "image/png",
                "gif": "image/gif",
                "webp": "image/webp",
            }
            content_type = format_to_mime.get(image_format, "image/jpeg")
            return f"data:{content_type};base64,{image_base64}", None

    return None, (
        "Image is required when action=photo. "
        "Use multipart fields image, photo, file, or audio (if the file is an image), or JSON imageBase64."
    )


# ---------------------------
# MAIN ENDPOINT (File Upload Only)
# ---------------------------
@app.route("/voice-ingredients", methods=["POST"])
def voice_ingredients():
    """
    Accepts audio file only.
    Returns list of ingredients + transcript.
    """
    try:
        # Check if audio file is in request
        if 'audio' not in request.files:
            return jsonify({"error": "Audio file is required"}), 400
        
        audio_file = request.files['audio']
        if not audio_file.filename:
            return jsonify({"error": "Uploaded audio file is empty"}), 400

        # Read audio file
        audio_bytes = audio_file.read()
        if not audio_bytes:
            return jsonify({"error": "Uploaded audio file is empty"}), 400

        print(f"Received audio file: {audio_file.filename}")

        # 1. Transcribe voice → text
        transcript = transcribe_audio_file(audio_bytes, audio_file.filename)

        # 2. Extract ingredients from transcript
        ingredients = extract_ingredients(transcript)

        response = {
            "ingredients": ingredients,
            "transcript": transcript,
            "message": f"Found {len(ingredients)} ingredient(s).",
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error in voice_ingredients: {str(e)}")
        return jsonify({
            "error": "Unexpected server error",
            "user_message": "We couldn't process the audio. Please try again or type your ingredients instead.",
            "details": str(e),
        }), 500


@app.route("/food-logging", methods=["POST"])
def food_logging():
    """
    Form fields:
    - action: "voice" (default) — requires multipart audio; runs transcribe_audio_file (Whisper).
      "text" — provide transcript via form field transcript or text; no audio.
      "photo" — multipart image under `image`, `photo`, `file`, or `audio` (if the upload is an image),
        or JSON with imageBase64 (+ imageFormat). Uses vision to identify food and estimate macros;
        `transcript` is a short summary of what was seen.
    - audio: file (required when action=voice)
    - transcript / text: food log text (required when action=text)
    - fast (optional): 1/true — smaller LLM completion budget for meal extraction only.
    - include_image (optional): true/false across form fields, query, or JSON (default true).
      When false, skip all meal image generation/attachment.
    Each meal in the response includes imageUrl: for action=photo the submitted image is uploaded to Firebase
    (short HTTPS URL); for voice/text, gpt-image-1 then Firebase when configured (stable URL),
    otherwise the temporary OpenAI CDN URL or DEFAULT_MEAL_IMAGE_URL fallback.
    Whisper: optional env FOOD_LOGGING_WHISPER_LANGUAGE (e.g. en) can reduce latency.
    Photo: optional FOOD_LOGGING_PHOTO_MODEL (defaults to VOICE_LOGGING_MODEL / gpt-4o-mini).
    """
    try:
        json_body = request.get_json(silent=True) or {}
        action = (
            request.form.get("action")
            or request.args.get("action")
            or json_body.get("action")
            or "voice"
        )
        action = str(action).strip().lower()
        if action not in ("voice", "text", "photo"):
            return jsonify({
                "error": "Invalid action",
                "allowed": ["voice", "text", "photo"],
            }), 400

        default_image_url = os.getenv("DEFAULT_MEAL_IMAGE_URL", "")
        fast_flag = (
            (request.form.get("fast") or request.args.get("fast") or "").strip().lower() in ("1", "true", "yes")
            or str(json_body.get("fast", "")).strip().lower() in ("1", "true", "yes")
        )
        include_image_raw = (
            request.form.get("include_image")
            or request.args.get("include_image")
            or json_body.get("include_image")
        )
        include_image = True if include_image_raw is None else str(include_image_raw).strip().lower() in ("1", "true", "yes")

        if action == "voice":
            if "audio" not in request.files:
                return jsonify({"error": "Audio file is required when action=voice"}), 400

            audio_file = request.files["audio"]
            if not audio_file.filename:
                return jsonify({"error": "Uploaded audio file is empty"}), 400

            audio_bytes = audio_file.read()
            if not audio_bytes:
                return jsonify({"error": "Uploaded audio file is empty"}), 400

            print(f"[food-logging] action=voice audio: {audio_file.filename}, {len(audio_bytes)} bytes")

            whisper_lang = (os.getenv("FOOD_LOGGING_WHISPER_LANGUAGE") or "").strip() or None
            transcript = transcribe_audio_file(
                audio_bytes,
                audio_file.filename,
                language=whisper_lang,
            )
        elif action == "text":
            transcript = (request.form.get("transcript") or request.form.get("text") or "").strip()
            if not transcript:
                return jsonify({
                    "error": "transcript or text is required when action=text",
                }), 400
            print(f"[food-logging] action=text, len={len(transcript)} chars")
        else:
            # action == "photo"
            photo_data_url, img_err = _food_logging_image_data_url_from_request()
            if img_err or not photo_data_url:
                return jsonify({"error": img_err or "Image required"}), 400
            print(f"[food-logging] action=photo, data_url_len={len(photo_data_url)}")
            meals, transcript = extract_meals_from_food_photo(photo_data_url, fast=fast_flag)
            if not meals and not (transcript or "").strip():
                return jsonify({
                    "error": "No food detected",
                    "transcript": "",
                    "meals": [],
                    "user_message": "We couldn't identify food in this photo. Try a clearer picture, better lighting, or use voice or text logging.",
                }), 400
            # Skip the shared voice/text path below
            if not meals:
                return jsonify({
                    "transcript": transcript,
                    "meals": [],
                    "message": "No structured meals extracted from this photo.",
                    "action": "photo",
                })

            photo_image_url = (
                _food_log_photo_to_storage_url(photo_data_url, default_image_url)
                if include_image
                else None
            )
            for m in meals:
                m["imageUrl"] = photo_image_url

            return jsonify({
                "transcript": transcript,
                "meals": meals,
                "message": f"Logged {len(meals)} meal(s) from photo.",
                "action": "photo",
            })

        if not transcript or not transcript.strip():
            return jsonify({
                "error": "No speech detected",
                "transcript": "",
                "meals": [],
                "user_message": "We couldn't detect speech in this recording. Try again or speak more clearly.",
            }), 400

        meals = extract_meals_from_voice_log_transcript(transcript, fast=fast_flag)

        if meals and include_image:
            workers = _food_logging_image_pool_size(len(meals))

            def _generate_and_persist(m: dict) -> str:
                b64 = _generate_food_log_meal_image_b64(m, fast_mode=fast_flag)
                return _food_logging_finalize_image_from_b64(b64, default_image_url)

            with ThreadPoolExecutor(max_workers=workers) as ex:
                finalized = list(ex.map(_generate_and_persist, meals))
            for meal, url in zip(meals, finalized):
                meal["imageUrl"] = url
        elif meals:
            for meal in meals:
                meal["imageUrl"] = None

        return jsonify({
            "transcript": transcript,
            "meals": meals,
            "message": f"Logged {len(meals)} meal(s).",
        })

    except Exception as e:
        print(f"Error in food_logging: {str(e)}")
        return jsonify({
            "error": "Unexpected server error",
            "user_message": "We couldn't process your food log. Please try again or type what you ate.",
            "details": str(e),
        }), 500


# ---------- Security: SSRF protection ----------
def _is_private_host(hostname: str) -> bool:
    """
    Resolve hostname and block private / local / link-local ranges.
    Prevents SSRF attacks like http://127.0.0.1:... or cloud metadata IPs.
    """
    if not hostname:
        return True

    # Block obvious localhost names
    lowered = hostname.lower()
    if lowered in {"localhost", "localhost.localdomain"}:
        return True

    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return True

    for family, _, _, _, sockaddr in infos:
        ip_str = sockaddr[0]
        ip = ipaddress.ip_address(ip_str)

        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return True

        # Block AWS/GCP/Azure metadata IP (most important)
        if ip_str == "169.254.169.254":
            return True

    return False


def validate_video_url(video_url: str):
    if not video_url or not isinstance(video_url, str):
        return False, "videoUrl is required"

    parsed = urlparse(video_url)
    if parsed.scheme not in {"http", "https"}:
        return False, "Only http/https URLs are allowed"

    if not parsed.netloc:
        return False, "Invalid URL"

    hostname = parsed.hostname
    if _is_private_host(hostname):
        return False, "URL host is not allowed"

    return True, ""


# ---------- yt-dlp helpers ----------
def ytdlp_base_opts(temp_dir: str, video_url: str = None):
    """
    Base yt-dlp options for audio extraction.
    Returns a dictionary of options that can be further customized.
    
    Args:
        temp_dir: Temporary directory for output files
        video_url: Optional video URL to determine if proxy should be used (YouTube only)
    """
    opts = {
        # IMPORTANT: robust selector with fallbacks (handles many edge cases)
        # Prefer low-bitrate audio — Whisper doesn't need high quality, so this reduces download size/time
        "format": "bestaudio[abr<=64]/bestaudio/best",
        "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
        "restrictfilenames": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 20,
        "retries": 2,
        "fragment_retries": 2,
        # Helps for YouTube signature issues & format availability
        "extractor_args": {
            "youtube": _yt_extractor_args()
        },
        # Convert to mp3 at low bitrate — Whisper transcription doesn't need high quality audio
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "64",
        }],
    }

    # Cookies improve TikTok/Instagram/Facebook reliability (never used for YouTube)
    _cf = _prepare_cookiefile(temp_dir, video_url)
    if _cf:
        opts["cookiefile"] = _cf

    # Proxy: YouTube via YT_PROXY, social (FB/IG/TikTok) via SOCIAL_PROXY
    _proxy = _ytdlp_proxy(video_url) if video_url else None
    if _proxy:
        opts["proxy"] = _proxy
        print(f"🌐 Using proxy: {_proxy}")

    return opts


def get_video_metadata(video_url: str):
    """
    Uses yt-dlp to fetch metadata without downloading.
    """
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "extractor_args": {"youtube": _yt_extractor_args()},
    }
    _proxy = _ytdlp_proxy(video_url)
    if _proxy:
        opts["proxy"] = _proxy
        print(f"🌐 Using proxy for metadata: {_proxy}")
    info = _ydl_extract(opts, video_url, download=False)
    return info


def download_audio_mp3(video_url: str):
    """
    Downloads video audio and returns mp3 bytes + basic metadata.
    Uses a single yt-dlp call to fetch metadata + download, avoiding the
    redundant separate get_video_metadata() round-trip.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        opts = ytdlp_base_opts(temp_dir, video_url)

        # extract_info with download=True fetches metadata AND downloads in one call
        info_dict = _ydl_extract(opts, video_url, download=True)

        duration = info_dict.get("duration")
        title = info_dict.get("title") or ""
        extractor = info_dict.get("extractor_key") or info_dict.get("extractor") or ""

        if duration and duration > MAX_VIDEO_SECONDS:
            raise ValueError(f"Video too long ({duration}s). Max allowed is {MAX_VIDEO_SECONDS}s")

        mp3s = [f for f in os.listdir(temp_dir) if f.endswith(".mp3")]
        if not mp3s:
            raise RuntimeError("Audio extraction failed: no .mp3 produced (is ffmpeg installed?)")

        mp3_path = os.path.join(temp_dir, mp3s[0])
        with open(mp3_path, "rb") as f:
            audio_bytes = f.read()

        if not audio_bytes:
            raise RuntimeError("Extracted audio file is empty")

        return audio_bytes, {"duration": duration, "title": title, "source": extractor}

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ---------- LLM helpers ----------
def _force_json_or_raise(text: str):
    """
    Attempts to parse JSON; tries mild cleanup if model wrapped it.
    """
    if not text:
        raise ValueError("Empty LLM response")

    # Strip code fences if any
    cleaned = re.sub(r"```json\s*|```", "", text).strip()

    # If model put extra text, attempt to extract the first JSON object block
    # (simple heuristic: find first '{' and last '}' )
    if "{" in cleaned and "}" in cleaned:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        cleaned = cleaned[start:end+1]

    return json.loads(cleaned)


def extract_recipe_from_transcript_chunk(transcript_chunk: str):
    system_prompt = """You extract recipe data from cooking transcripts.
Return ONLY valid JSON matching this schema:
{
  "ingredients": [{"name": "...", "quantity": "..."}],
  "instructions": ["Step 1: ...", "Step 2: ..."]
}
Rules:
- Do NOT add explanations or markdown.
- If quantity is unknown, use "".
- Keep instructions in chronological order.
"""

    user_prompt = f"Transcript:\n{transcript_chunk}\n\nReturn the JSON now."

    # Try once with strict JSON if supported, else fallback
    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
    except Exception:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1200,
        )

    text = completion.choices[0].message.content.strip()
    return _force_json_or_raise(text)


def merge_recipe_parts(parts):
    """
    Merge multiple partial extractions into a single clean recipe.
    Dedup ingredients, re-number and clean steps with a final LLM pass.
    """
    merge_system = """You merge multiple partial recipe JSONs into ONE final recipe JSON.
Return ONLY valid JSON with schema:
{
  "ingredients": [{"name": "...", "quantity": "..."}],
  "instructions": ["Step 1: ...", "Step 2: ..."]
}
Rules:
- Deduplicate ingredients (case-insensitive).
- If the same ingredient appears with different quantities, keep the most specific quantity.
- Combine instructions, remove duplicates, ensure chronological order, and ensure steps are detailed and actionable.
- Output ONLY JSON.
"""

    merge_user = json.dumps({"parts": parts}, ensure_ascii=False)

    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": merge_system},
                {"role": "user", "content": merge_user},
            ],
            temperature=0.1,
            max_tokens=1800,
            response_format={"type": "json_object"},
        )
    except Exception:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": merge_system},
                {"role": "user", "content": merge_user},
            ],
            temperature=0.1,
            max_tokens=1800,
        )

    text = completion.choices[0].message.content.strip()
    return _force_json_or_raise(text)


def chunk_text(text: str, max_chars: int = 6000):
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # try to break on sentence boundary
        cut = text.rfind(".", start, end)
        if cut == -1 or cut < start + int(max_chars * 0.6):
            cut = end
        chunks.append(text[start:cut].strip())
        start = cut
    return [c for c in chunks if c]

# ---------- The endpoint ----------
@app.route("/extract-recipe-from-video", methods=["POST"])
def extract_recipe_from_video():
    """
    Accepts: {"videoUrl":"..."}
    Returns: {"ingredients":[...], "instructions":[...], "transcript":"...", "meta": {...}}
    """
    try:
        data = request.get_json(silent=True) or {}
        video_url = data.get("videoUrl")

        ok, err = validate_video_url(video_url)
        if not ok:
            return jsonify({"error": err}), 400

        # ── Cache check ──────────────────────────────────────────────────────────
        url_key = hashlib.sha256(video_url.encode()).hexdigest()
        with _recipe_cache_lock:
            if url_key in _recipe_cache:
                print(f"⚡ Cache hit for video URL: {video_url}")
                cached = dict(_recipe_cache[url_key])
                _ensure_cached_recipe_nutrition(cached)
                return jsonify({**_apply_extract_recipe_image_pref(cached), "cached": True})
        # ─────────────────────────────────────────────────────────────────────────

        print(f"🎥 Processing video URL: {video_url}")

        if is_youtube_url(video_url):
            try:
                info = _yt_meta(video_url)
                _assert_youtube_shorts_duration((info or {}).get("duration"))
            except YouTubeVideoTooLongError as e:
                return _youtube_too_long_response(e)
            except Exception as e:
                return jsonify({
                    "error": "Video metadata failed",
                    "user_message": "We couldn't read this video. Please try another link or add the recipe manually.",
                    "details": str(e),
                }), 500

        # 1) Download + extract audio
        t0 = time.time()
        try:
            audio_bytes, meta = download_audio_mp3(video_url)
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 413
        except yt_dlp.utils.DownloadError as de:
            # very common for IG/TikTok without cookies
            return jsonify({
                "error": "Failed to download/extract audio from video URL",
                "details": str(de),
                "hint": "TikTok/Instagram often require cookies/login. Set YTDLP_COOKIES_FILE on the server."
            }), 400
        except Exception as e:
            return jsonify({
                "error": "Audio extraction failed",
                "user_message": "We couldn't get the audio from this video. Please try another link or add the recipe manually.",
                "details": str(e),
            }), 500

        print(f"✅ Audio extracted: {len(audio_bytes)} bytes in {time.time()-t0:.2f}s")

        # 2) Transcription
        print("🎤 Transcribing audio...")
        audio_file_obj = io.BytesIO(audio_bytes)
        audio_file_obj.name = "audio.mp3"

        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file_obj,
            response_format="text",
        )
        transcript_text = transcript.strip() if isinstance(transcript, str) else str(transcript).strip()

        if not transcript_text:
            return jsonify({
                "error": "Failed to transcribe video",
                "user_message": "We couldn't understand the audio from this video. Try another link or add the recipe manually.",
                "message": "No transcript generated from video audio",
            }), 500

        print(f"✅ Transcript generated: {len(transcript_text)} characters")

        # 3) LLM extraction with chunking + merge
        # gpt-4o-mini has a 128k token context window; 80 000 chars ≈ 20 000 tokens,
        # so almost every cooking video transcript fits in a single call — no merge needed.
        print("🍳 Extracting recipe information...")
        chunks = chunk_text(transcript_text, max_chars=80000)

        # Process all chunks concurrently — for a single-chunk transcript this is a no-op,
        # but for multi-chunk transcripts it turns N sequential LLM round-trips into 1 parallel batch.
        parts = [None] * len(chunks)
        chunk_error = {}

        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            future_to_idx = {
                executor.submit(extract_recipe_from_transcript_chunk, ch): i
                for i, ch in enumerate(chunks)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    parts[idx] = future.result()
                except Exception as e:
                    chunk_error["idx"] = idx + 1
                    chunk_error["details"] = str(e)

        if chunk_error:
            return jsonify({
                "error": "Failed to extract recipe from transcript chunk",
                "user_message": "We couldn't extract a recipe from this link. Please try another or add the recipe manually.",
                "chunk": chunk_error["idx"],
                "details": chunk_error["details"],
                "transcript": transcript_text
            }), 500

        final_recipe = merge_recipe_parts(parts) if len(parts) > 1 else parts[0]

        ingredients = final_recipe.get("ingredients", []) or []
        instructions = final_recipe.get("instructions", []) or []

        # Normalize types
        if ingredients and isinstance(ingredients[0], str):
            ingredients = [{"name": ing, "quantity": ""} for ing in ingredients]
        if not isinstance(instructions, list):
            instructions = [str(instructions)]

        result = {
            "ingredients": ingredients,
            "instructions": instructions,
            "transcript": transcript_text,
            "meta": meta,
            "message": f"Successfully extracted recipe with {len(ingredients)} ingredients and {len(instructions)} instructions"
        }

        # ── Cache write ──────────────────────────────────────────────────────────
        with _recipe_cache_lock:
            _recipe_cache[url_key] = result
        # ─────────────────────────────────────────────────────────────────────────

        return jsonify(result)

    except Exception as e:
        print(f"💥 Critical error in extract_recipe_from_video: {str(e)}")
        return jsonify({
                "error": "Unexpected server error",
                "user_message": DEFAULT_500_USER_MESSAGE,
                "details": str(e),
            }), 500

#####video only code ends here#####


@app.route("/listen", methods=["POST"])
def listen():
    audio_file = request.files["file"]
    email = request.form.get("email")  # Get email from form data
    
    # Transcribe audio to text
    transcript = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=(audio_file.filename, audio_file.stream, audio_file.mimetype)
    )
    user_question = transcript.text
    
    if not email:
        return jsonify({"error": "Email is required"}), 400
    
    # Use the same chat logic as the /chat endpoint (parallelized)
    context_result = {}
    history_result = {}

    def fetch_context():
        try:
            context_result["data"] = get_user_context(email)
        except Exception as e:
            print(f"ERROR fetching user context: {e}")
            context_result["data"] = ({}, {'all': []}, {})

    def fetch_history():
        try:
            history_result["data"] = get_user_chat_history(email)
        except Exception as e:
            print(f"ERROR fetching chat history: {e}")
            history_result["data"] = []

    t1 = Thread(target=fetch_context)
    t2 = Thread(target=fetch_history)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    goal, categorized_meals, user_preferences = context_result.get("data", ({}, {'all': []}, {}))

    # Get recent chat history (limit to last 2 for performance)
    chat_history = history_result.get("data", [])[-2:]  # Only last 2 messages
    # Truncate long bot responses to keep context manageable
    formatted_history = "\n".join([
        f"User: {q}\nBot: {a[:150] + '...' if len(a) > 150 else a}" 
        for q, a in chat_history
    ])
    
    print(f"DEBUG: Raw chat history: {chat_history}")
    print(f"DEBUG: Formatted chat history: {formatted_history}")

    profile_parts = []
    if user_preferences:
        if user_preferences.get("age"):
            profile_parts.append(f"Age: {user_preferences.get('age')}")
        if user_preferences.get("gender"):
            profile_parts.append(f"Gender: {user_preferences.get('gender')}")
        if user_preferences.get("height"):
            unit = user_preferences.get("height_unit", "cm")
            profile_parts.append(f"Height: {user_preferences.get('height')} {unit}")
        if user_preferences.get("weight"):
            profile_parts.append(f"Weight: {user_preferences.get('weight')} {user_preferences.get('weigh_unit', 'kg')}")
        if user_preferences.get("target_weight"):
            profile_parts.append(f"Target weight: {user_preferences.get('target_weight')} {user_preferences.get('weight_unit', 'kg')}")
        if user_preferences.get("weight_goal"):
            profile_parts.append(f"Weight Goal: {user_preferences.get('weight_goal')}")
        if user_preferences.get("lifestyle"):
            profile_parts.append(f"Lifestyle: {user_preferences.get('lifestyle')}")
        if user_preferences.get("meal_per_day"):
            profile_parts.append(f"Meals per day: {user_preferences.get('meal_per_day')}")
        if user_preferences.get("meal_type_list"):
            cuisines = ", ".join(user_preferences.get("meal_type_list"))
            profile_parts.append(f"Preferred cuisines: {cuisines}")
        if user_preferences.get("calorie_goal") is not None:
            profile_parts.append(f"Calorie goal: {user_preferences.get('calorie_goal')} kcal")
        if user_preferences.get("protein_goal") is not None:
            profile_parts.append(f"Protein goal: {user_preferences.get('protein_goal')} g")
        if user_preferences.get("carbs_goal") is not None:
            profile_parts.append(f"Carbs goal: {user_preferences.get('carbs_goal')} g")
        # Check for fat_goal (handle both snake_case and camelCase variants)
        if "fat_goal" in user_preferences:
            profile_parts.append(f"Fat goal: {user_preferences.get('fat_goal')} g")
        elif "fatGoal" in user_preferences:
            profile_parts.append(f"Fat goal: {user_preferences.get('fatGoal')} g")


    if profile_parts:
        goal_summary = "User Profile:\n" + "\n".join(profile_parts)
    elif goal:
        goal_summary = f"User goal: {goal.get('goalType', 'not set')}, Current: {goal.get('currentWeight')}kg, Target: {goal.get('targetWeight')}kg by {goal.get('targetDate')}"
    else:
        goal_summary = "User goal: No specific goals set"
    
    # Debug: Print goal_summary to verify it contains weight_goal
    print(f"DEBUG: Goal: {goal}")
    print(f"DEBUG: User preferences dict: {user_preferences}")
    print(f"DEBUG: Goal summary for {email}: {goal_summary}")
    print(f"DEBUG: Fat goal value: {user_preferences.get('fat_goal')} or {user_preferences.get('fatGoal')}")
    
    # Get recent meals (limit to last 2 for performance)
    all_meals = categorized_meals.get('all', [])[:2]  # Only last 2 meals
    meals_summary = "\n".join(all_meals) if all_meals else "No recent meals found"
    
    # Don't truncate goal_summary - user profile data is critical and must be complete
    # The performance impact of a few hundred extra characters is negligible compared to LLM processing time
    system_context = f"""You are a nutrition assistant. User: {goal_summary}. Recent: {formatted_history[:200] if formatted_history else 'New conversation'}. Meals: {meals_summary[:200] if meals_summary else 'None'}.

CRITICAL: Meal history (shown above as "Meals:") is COMPLETELY OPTIONAL and used ONLY for understanding patterns. It is NOT required to provide meal recommendations. When the user asks for meal recommendations (dinner ideas, lunch ideas, breakfast ideas, meal plans, lower calorie options, etc.), you MUST ALWAYS provide 3-5 meal recommendations with complete recipes. NEVER refuse by saying you don't have meal history. NEVER say "I don't have the specific meal history" or similar. You MUST use: (1) User Profile information, (2) Your general nutrition knowledge, and (3) The user's specific requirements. If meal history is empty or "None", you MUST STILL provide recommendations. Refusing to provide meals is FORBIDDEN.


CRITICAL INSTRUCTIONS:
1. MEAL RECOMMENDATION REQUIREMENT - HIGHEST PRIORITY: When the user asks for meal recommendations, food suggestions, meal options, dinner ideas, lunch ideas, breakfast ideas, or ANY variation (including "dinner ideas under 400 kcal", "lower calorie options", "show more dinner ideas", "meal plans", etc.), you MUST IMMEDIATELY provide EXACTLY 3-5 meal recommendations with complete recipes. This is THE HIGHEST PRIORITY instruction and is MANDATORY. 
   - NEVER refuse to provide meals
   - NEVER say "I don't have meal history" or "I don't have the specific meal history" or "I don't have any recent meals" or any variation of this refusal
   - NEVER say you cannot provide recommendations due to lack of history
   - Meal history is COMPLETELY OPTIONAL and NOT required - it is only for understanding patterns
   - You MUST ALWAYS generate new meal recommendations using: (1) User Profile information (calorie goals, macro goals, preferences, lifestyle, age, etc.), (2) Your general nutrition knowledge and recipe database, and (3) The specific requirements in the user's question (e.g., "under 400 kcal", "lower calorie", "high protein")
   - If meal history is empty or missing, you MUST STILL provide 3-5 meal recommendations based on user profile and general knowledge
   - If you refuse to provide meals or say you don't have history, you have VIOLATED this instruction
   - Example of CORRECT response: Start immediately with "- Meal Name 1: Description (Calories: X, Protein: Yg, Carbs: Zg, Fat: Wg)" followed by Recipe, Ingredients, and Instructions
   - Example of INCORRECT response: "I don't have the specific meal history for dinner options under 400 kcal" - THIS IS FORBIDDEN
2. ALWAYS answer questions about the user's goals, weight goal, calorie goals, macro goals, preferences, etc. DIRECTLY from the "User Information" section above. DO NOT say the information is not available if it exists in the User Information section.
3. SYNONYM RECOGNITION: Recognize that different phrasings mean the same thing. For example:
   - "weight target", "target weight", "weight goal", "goal weight" all refer to the same thing
   - "calorie goal" and "calorie target" are the same
   - "protein goal" and "protein target" are the same
   - When the user asks about ANY variation of these terms, look for the relevant information in the User Information section using ALL possible field names (Weight Goal, Target Weight, etc.)
4. WEIGHT GOAL/TARGET QUESTIONS: If the user asks about their weight goal, weight target, target weight, goal weight, or any variation, look for BOTH "Weight Goal:" AND "Target Weight:" in the User Information section. Use whichever is available. NEVER say the information is not available if either field exists. If both exist, use the most relevant one or combine them.
5. If the user asks about their calorie goal, protein goal, carbs goal, fat goal, age, lifestyle, preferred cuisines, etc., extract that information directly from the User Information section. Recognize synonyms and variations of these terms as well.
6. FORMATTING REQUIREMENT: When providing meal recommendations, food suggestions, or lists of meals, ALWAYS format them as bullet points using "- " or "* " at the start of each line. Each meal recommendation MUST include:
   - Meal name and brief description
   - Nutritional information (Calories, Protein, Carbs, Fat)
   - Complete recipe with ingredients and DETAILED step-by-step cooking instructions
   Example format:
   - Meal Name 1: Description (Calories: X, Protein: Yg, Carbs: Zg, Fat: Wg)
     Recipe: 
     Ingredients: 
     - Ingredient 1: exact quantity (e.g., "1 cup", "200g", "2 tablespoons")
     - Ingredient 2: exact quantity
     - Ingredient 3: exact quantity
     Instructions:
     1. Detailed step 1 with specific actions, temperatures, and times (e.g., "Heat 1 tablespoon olive oil in a large pan over medium-high heat for 2 minutes")
     2. Detailed step 2 with specific techniques and measurements
     3. Detailed step 3 with cooking times and temperatures
     4. Continue with numbered steps until the meal is complete
   - Meal Name 2: Description (Calories: X, Protein: Yg, Carbs: Zg, Fat: Wg)
     Recipe: [same detailed format]
   - Meal Name 3: Description (Calories: X, Protein: Yg, Carbs: Zg, Fat: Wg)
     Recipe: [same detailed format]
   ALWAYS include a complete recipe for every meal you recommend, even if the recipe is not in the retrieved context. Use your knowledge to provide accurate recipes.
7. ALWAYS maintain conversation context - remember everything the user has asked and your previous responses
8. Use the complete conversation history to provide contextual and personalized responses
9. When the user asks "what was my last question", refer to the question they asked BEFORE their current question (not the current one)
10. Build upon previous conversations - if they ask follow-up questions, reference what you've already discussed
11. Be conversational and remember what you've told them before
12. When asked about specific meal types (breakfast, lunch, dinner, snacks), use only the data from that category when analyzing history or patterns.
13. The meal data includes detailed nutritional information (calories, carbs, protein, fat) for each meal.
14. If the user asks about trends or patterns, analyze their meal history across multiple entries.
15. Provide personalized insights based on their eating patterns and previous questions.
16. Maintain a helpful, friendly tone throughout the conversation.
17. Use the "User Profile" section (age, lifestyle, calorie/macro goals, preferred cuisines, etc.) to tailor every recommendation. Respect their macros, calorie targets, and cuisine preferences when possible.
18. DETAILED RECIPE REQUIREMENT: ALWAYS provide a complete, DETAILED recipe for EVERY meal you recommend. REMEMBER: You must provide 3-5 meals (never just 1), and each meal needs a full recipe. The recipe MUST include:
    - Ingredients list with EXACT quantities (e.g., "1 cup", "200g", "2 tablespoons", "1 medium onion", "3 cloves garlic")
    - Numbered step-by-step instructions that are SPECIFIC and ACTIONABLE:
      * Include exact cooking temperatures (e.g., "375°F", "medium-high heat")
      * Include exact cooking times (e.g., "cook for 5-7 minutes", "bake for 25 minutes")
      * Include specific techniques (e.g., "sauté until golden brown", "whisk until smooth", "simmer uncovered")
      * Include preparation details (e.g., "dice into 1-inch cubes", "chop finely", "slice thinly")
      * Include when to add ingredients (e.g., "add after 2 minutes", "stir in at the end")
      * Include visual/textural cues (e.g., "until tender", "until golden", "until sauce thickens")
    Never skip the recipe or provide vague instructions. The recipe should be detailed enough that someone with basic cooking knowledge can successfully prepare the meal without additional research. You have 4000 tokens available, so use them to provide 3-5 complete meal recommendations with full recipes.
19. NO CROSS-QUESTIONING OR REFUSAL FOR MEAL REQUESTS - ABSOLUTELY FORBIDDEN: If the user asks for meal ideas, meal plans, or specific meal suggestions (for example, "Show dinner ideas under 400 kcal", "lower calorie options", "show more dinner ideas", "dinner ideas", etc.), you MUST directly provide the requested meal recommendations. 
   FORBIDDEN RESPONSES (DO NOT USE THESE):
   - "I don't have the specific meal history" or "I don't have the specific meal history for dinner options under 400 kcal"
   - "I don't have meal history" or "I don't have any recent meals"
   - "I don't have the necessary meal history" or any variation
   - "Would you like me to suggest..." or any follow-up questions
   - Any response that refuses to provide meals
   REQUIRED RESPONSE: You MUST ALWAYS provide 3-5 meal recommendations with complete recipes using: (1) User Profile (calorie/macro goals, preferences, lifestyle), (2) Your general nutrition knowledge, and (3) The user's specific requirements. Meal history is completely optional and NOT required. If meal history is missing or empty, you MUST still provide recommendations based on user profile and general knowledge. Start your response immediately with the first meal recommendation in the required format.
20. FRESH MEAL RECOMMENDATIONS: When the user asks for meal recommendations, DO NOT simply repeat or select meals from their past meal history. Always generate NEW meal ideas and recipes that fit their goals and preferences. You may use history only to understand patterns and preferences, but the recommended meals themselves should be fresh suggestions, not just a recap of what they already ate.
"""
    
    print(f"DEBUG: System context being sent to AI: {system_context}")

    # Use invoke() instead of run() for better performance
    # Format the query with system context
    query = f"{system_context}\n\nUser question: {user_question}"
    result = rag_chain.invoke({"query": query})
    response = result.get("result", str(result))

    # Save chat interaction for future context
    Thread(target=save_user_chat, args=(email, user_question, response)).start()
    
    return jsonify({
        "transcript": user_question,
        "reply": response
    })

    ## video and webpage code starts here#####

    

VIDEO_DOMAINS = {
    "youtube.com", "www.youtube.com", "youtu.be",
    "tiktok.com", "www.tiktok.com",
    "instagram.com", "www.instagram.com",
    # Facebook (videos/reels/watch) — all natively supported by yt-dlp
    "facebook.com", "www.facebook.com", "m.facebook.com", "web.facebook.com",
    "fb.watch", "fb.com",
}

def is_video_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in VIDEO_DOMAINS)

def is_youtube_url(url: str) -> bool:
    """Check if URL is a YouTube video/shorts URL."""
    if not url:
        return False
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in {"youtube.com", "www.youtube.com", "youtu.be"})


class YouTubeVideoTooLongError(Exception):
    """Raised when a YouTube video exceeds the Shorts duration limit."""

    def __init__(self, duration: float, max_seconds: int | None = None):
        self.duration = float(duration)
        self.max_seconds = max_seconds if max_seconds is not None else _youtube_shorts_max_seconds()
        super().__init__(
            f"YouTube video too long ({self.duration:.0f}s). "
            f"Only Shorts under {self.max_seconds}s are supported."
        )


def _youtube_shorts_max_seconds() -> int:
    try:
        return max(1, int(os.getenv("YOUTUBE_SHORTS_MAX_SECONDS", "180")))
    except ValueError:
        return 180


def _youtube_max_duration_label(seconds: int) -> str:
    if seconds >= 60 and seconds % 60 == 0:
        mins = seconds // 60
        return f"{mins} minute{'s' if mins != 1 else ''}"
    return f"{seconds} seconds"


def _assert_youtube_shorts_duration(duration) -> None:
    """Reject YouTube videos longer than the Shorts limit (default 180s)."""
    cap = _youtube_shorts_max_seconds()
    try:
        dur = float(duration or 0)
    except (TypeError, ValueError):
        return
    if dur > 0 and dur > cap:
        raise YouTubeVideoTooLongError(dur, cap)


def _youtube_too_long_response(exc: YouTubeVideoTooLongError):
    limit = _youtube_max_duration_label(exc.max_seconds)
    msg = (
        f"Only YouTube Shorts are allowed. Videos must be {limit} or shorter. "
        "Please try a Shorts link or add the recipe manually."
    )
    return jsonify({
        "error": "Only YouTube Shorts allowed",
        "message": msg,
        "user_message": msg,
        "details": str(exc),
    }), 413


def determine_source_type(url: str) -> str:
    """
    Determines the source type based on URL.
    Returns: "TikTok", "YouTube", "Instagram", "Facebook", "Photos", or "Manual"
    """
    if not url:
        return "Manual"

    host = (urlparse(url).hostname or "").lower()

    if "tiktok.com" in host or "tiktok" in host:
        return "TikTok"
    elif "youtube.com" in host or "youtu.be" in host:
        return "YouTube"
    elif "instagram.com" in host or "instagram" in host:
        return "Instagram"
    elif "facebook.com" in host or "fb.watch" in host or "fb.com" in host:
        return "Facebook"
    elif any(ext in url.lower() for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]):
        return "Photos"
    else:
        return "Manual"


def extract_recipe_tags(recipe: dict) -> list[str]:
    """
    Extracts tags from recipe data.
    Returns list of tags: ["High Protein", "Vegetarian", "Vegan", "Quick", "Easy", "Medium", "Hard"]
    """
    tags = []
    
    if not recipe:
        return tags
    
    # Get recipe data
    ingredients = recipe.get("ingredients", [])
    prep_time = recipe.get("prep_time", "")
    cook_time = recipe.get("cook_time", "")
    total_time = recipe.get("total_time", "")
    nutrition = recipe.get("nutrition", {})
    
    # Extract ingredient names for analysis
    ingredient_names = []
    for ing in ingredients:
        if isinstance(ing, dict):
            ingredient_names.append(ing.get("name", "").lower())
        elif isinstance(ing, str):
            ingredient_names.append(ing.lower())
    
    ingredient_text = " ".join(ingredient_names)
    
    # Check for Vegetarian (no meat, fish, poultry)
    meat_keywords = ["chicken", "beef", "pork", "lamb", "turkey", "duck", "fish", "salmon", "tuna", "shrimp", "crab", "lobster", "meat", "bacon", "sausage", "ham", "steak"]
    has_meat = any(keyword in ingredient_text for keyword in meat_keywords)
    if not has_meat and ingredient_text:
        tags.append("Vegetarian")
    
    # Check for Vegan (no animal products)
    animal_keywords = meat_keywords + ["milk", "cheese", "butter", "cream", "yogurt", "egg", "honey", "gelatin", "whey", "casein"]
    has_animal_products = any(keyword in ingredient_text for keyword in animal_keywords)
    if not has_animal_products and ingredient_text:
        tags.append("Vegan")
    
    # Check for High Protein (from nutrition data or ingredient analysis)
    protein_value = None
    if isinstance(nutrition, dict):
        # Try different possible keys
        protein_value = nutrition.get("protein") or nutrition.get("proteinContent") or nutrition.get("protein_g")
        if isinstance(protein_value, str):
            # Extract number from string like "25g" or "25 g"
            match = re.search(r'(\d+\.?\d*)', protein_value)
            if match:
                protein_value = float(match.group(1))
    
    # High protein threshold: >20g per serving (rough estimate)
    high_protein_ingredients = ["chicken", "beef", "turkey", "fish", "salmon", "tuna", "eggs", "tofu", "tempeh", "lentils", "beans", "chickpeas", "quinoa", "greek yogurt", "cottage cheese", "protein powder"]
    has_high_protein_ingredients = any(ing in ingredient_text for ing in high_protein_ingredients)
    
    if protein_value and protein_value > 20:
        tags.append("High Protein")
    elif has_high_protein_ingredients and len(ingredients) > 0:
        tags.append("High Protein")
    
    # Determine difficulty based on total time and instruction complexity
    def parse_time_to_minutes(time_str: str) -> int:
        """Convert time string to minutes. Handles formats like '30 mins', '1 hr 30 mins', 'PT30M'"""
        if not time_str:
            return 0
        
        time_str = time_str.lower()
        total_minutes = 0
        
        # Parse hours
        hour_match = re.search(r'(\d+)\s*hr', time_str)
        if hour_match:
            total_minutes += int(hour_match.group(1)) * 60
        
        # Parse minutes
        minute_match = re.search(r'(\d+)\s*min', time_str)
        if minute_match:
            total_minutes += int(minute_match.group(1))
        
        # Parse ISO 8601 format (PT30M, PT1H30M)
        if time_str.startswith("pt"):
            h_match = re.search(r'(\d+)h', time_str)
            m_match = re.search(r'(\d+)m', time_str)
            if h_match:
                total_minutes += int(h_match.group(1)) * 60
            if m_match:
                total_minutes += int(m_match.group(1))
        
        return total_minutes
    
    # Use total_time if available, otherwise sum prep + cook
    time_minutes = 0
    if total_time:
        time_minutes = parse_time_to_minutes(total_time)
    else:
        prep_min = parse_time_to_minutes(prep_time) if prep_time else 0
        cook_min = parse_time_to_minutes(cook_time) if cook_time else 0
        time_minutes = prep_min + cook_min
    
    instructions = recipe.get("instructions", [])
    num_steps = len(instructions) if isinstance(instructions, list) else 0
    
    # Difficulty classification
    if time_minutes == 0 and num_steps == 0:
        pass  # Can't determine
    elif time_minutes <= 30 and num_steps <= 5:
        tags.append("Quick")
        tags.append("Easy")
    elif time_minutes <= 60 and num_steps <= 8:
        tags.append("Easy")
    elif time_minutes <= 90 and num_steps <= 12:
        tags.append("Medium")
    elif time_minutes > 90 or num_steps > 12:
        tags.append("Hard")
    else:
        tags.append("Medium")  # Default
    
    return tags

# ── Tiered webpage fetching ──────────────────────────────────────────────
# Tier 1: direct fetch with realistic browser headers (handles most sites).
# Tier 2: ScraperAPI fallback for sites behind Cloudflare / anti-bot walls
#         (e.g. AllRecipes). Only used when a block is detected AND a key is
#         configured; degrades gracefully to a clean error otherwise.
SCRAPER_API_KEY = os.getenv("SCRAPER_API_KEY", "").strip()
SCRAPER_API_ENDPOINT = "https://api.scraperapi.com/"
# ScraperAPI retries internally for up to ~60s, so its client timeout must be
# generous and independent of the (shorter) LLM budget.
SCRAPER_API_TIMEOUT = int(os.getenv("SCRAPER_API_TIMEOUT", "70"))
# ultra_premium activates advanced anti-bot bypass (needed for Cloudflare
# *managed* challenges like AllRecipes). Costs more credits — disable via env
# to save free-tier credits if you only hit lighter protections.
SCRAPER_API_ULTRA = os.getenv("SCRAPER_API_ULTRA", "true").strip().lower() == "true"

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# HTTP statuses that typically indicate an anti-bot block rather than a real
# "page missing" error — worth retrying through the scraping fallback.
_BLOCK_STATUSES = {401, 402, 403, 429, 503}


def _looks_like_challenge(html: str) -> bool:
    """Detect a Cloudflare/anti-bot interstitial returned with a 200 status."""
    if not html:
        return True
    snippet = html[:3000].lower()
    return (
        "just a moment" in snippet
        or "challenges.cloudflare.com" in snippet
        or "_cf_chl_opt" in snippet
        or "enable javascript and cookies to continue" in snippet
    )


def _fetch_via_scraperapi(url: str) -> str:
    """Fetch a URL through ScraperAPI (renders JS + solves Cloudflare)."""
    params = {
        "api_key": SCRAPER_API_KEY,
        "url": url,
        "render": "true",  # execute JS / solve managed challenge
    }
    if SCRAPER_API_ULTRA:
        params["ultra_premium"] = "true"  # advanced anti-bot bypass
    r = requests.get(SCRAPER_API_ENDPOINT, params=params, timeout=SCRAPER_API_TIMEOUT)
    r.raise_for_status()
    return r.text


def fetch_html(url: str, timeout: int | None = None) -> str:
    timeout = timeout or EXTRACT_RECIPE_TIMEOUT
    last_response = None
    direct_error = None

    # ── Tier 1: direct fetch with realistic browser headers ──────────────
    try:
        last_response = requests.get(url, headers=_BROWSER_HEADERS, timeout=timeout)
        blocked = (
            last_response.status_code in _BLOCK_STATUSES
            or _looks_like_challenge(last_response.text)
        )
        if not blocked:
            last_response.raise_for_status()
            return last_response.text
        print(f"[fetch_html] direct fetch blocked (status={last_response.status_code}) for {url[:120]}")
    except requests.exceptions.RequestException as e:
        direct_error = e
        print(f"[fetch_html] direct fetch error for {url[:120]}: {str(e)[:200]}")

    # ── Tier 2: ScraperAPI fallback (handles Cloudflare-protected sites) ──
    if SCRAPER_API_KEY:
        try:
            print(f"[fetch_html] retrying via ScraperAPI (ultra={SCRAPER_API_ULTRA}) for {url[:120]}")
            return _fetch_via_scraperapi(url)
        except requests.exceptions.RequestException as e:
            print(f"[fetch_html] ScraperAPI fetch failed for {url[:120]}: {str(e)[:200]}")
    else:
        print("[fetch_html] blocked and no SCRAPER_API_KEY set — cannot fall back")

    # ── All tiers exhausted — raise a clean error for /extract-recipe ─────
    if last_response is not None:
        # Real block status (402/403/429/503) raises with the true code so the
        # endpoint can show the right user message.
        last_response.raise_for_status()
        # 200 but challenge HTML: surface as a 403-style block.
        synthetic = requests.models.Response()
        synthetic.status_code = 403
        synthetic._content = last_response.content
        raise requests.exceptions.HTTPError(
            "Blocked by anti-bot challenge (e.g. Cloudflare)", response=synthetic
        )
    raise direct_error

def extract_og_image(soup: BeautifulSoup) -> str | None:
    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        return og["content"].strip()
    tw = soup.find("meta", attrs={"name": "twitter:image"})
    if tw and tw.get("content"):
        return tw["content"].strip()
    return None

def extract_jsonld_recipes(html: str):
    soup = BeautifulSoup(html, "lxml")
    scripts = soup.find_all("script", type="application/ld+json")
    recipes = []

    for s in scripts:
        try:
            data = json.loads(s.string or "")
        except Exception:
            continue

        # JSON-LD can be dict, list, or nested graph
        candidates = []
        if isinstance(data, list):
            candidates = data
        elif isinstance(data, dict):
            if "@graph" in data and isinstance(data["@graph"], list):
                candidates = data["@graph"]
            else:
                candidates = [data]

        for item in candidates:
            if not isinstance(item, dict):
                continue
            t = item.get("@type")
            if isinstance(t, list):
                is_recipe = any(x.lower() == "recipe" for x in t if isinstance(x, str))
            else:
                is_recipe = isinstance(t, str) and t.lower() == "recipe"
            if is_recipe:
                recipes.append(item)

    return recipes, soup

def _flatten_instructions(recipe_instructions):
    """
    Handles JSON-LD recipeInstructions that may be:
    - string
    - list of strings
    - list of HowToStep dicts
    - list of HowToSection dicts with itemListElement (steps)
    """
    steps = []

    def add_step(text: str):
        text = (text or "").strip()
        if text:
            steps.append(text)

    def handle_node(node):
        if node is None:
            return

        # Plain string
        if isinstance(node, str):
            add_step(node)
            return

        # List of nodes
        if isinstance(node, list):
            for item in node:
                handle_node(item)
            return

        # Dict node (HowToStep / HowToSection / etc.)
        if isinstance(node, dict):
            node_type = node.get("@type") or node.get("type")

            # HowToSection: keep section heading + recurse into itemListElement
            if isinstance(node_type, str) and node_type.lower() == "howtosection":
                heading = node.get("name") or node.get("headline") or ""
                if heading:
                    add_step(f"{heading.strip()}")
                handle_node(node.get("itemListElement") or node.get("steps"))
                return

            # HowToStep: take text
            if isinstance(node_type, str) and node_type.lower() == "howtostep":
                add_step(node.get("text") or node.get("name"))
                return

            # Generic fallback: try common keys
            add_step(node.get("text") or node.get("name"))
            # and recurse if there is nested list
            handle_node(node.get("itemListElement") or node.get("steps"))
            return

    handle_node(recipe_instructions)

    # Deduplicate while preserving order
    seen = set()
    out = []
    for s in steps:
        key = re.sub(r"\s+", " ", s.strip()).lower()
        if key and key not in seen:
            seen.add(key)
            out.append(s)
    return out


def normalize_duration(duration_str: str) -> str:
    """
    Converts ISO 8601 duration format (PT30M, PT1H30M, PT578M) to human-readable format.
    Examples:
        PT30M -> "30 mins"
        PT1H30M -> "1 hr 30 mins"
        PT578M -> "9 hrs 38 mins"
        PT2H -> "2 hrs"
    """
    if not duration_str or not isinstance(duration_str, str):
        return ""
    
    duration_str = duration_str.strip().upper()
    
    # If already human-readable (contains "hr", "min", "mins", etc.), return as-is
    if any(word in duration_str.lower() for word in ["hr", "hour", "min", "minute", "sec", "second"]):
        return duration_str
    
    # Parse ISO 8601 duration format (PT30M, PT1H30M, etc.)
    if not duration_str.startswith("PT"):
        return duration_str  # Not ISO 8601 format, return as-is
    
    # Extract hours, minutes, seconds
    hours = 0
    minutes = 0
    seconds = 0
    
    # Match hours (H)
    hour_match = re.search(r'(\d+)H', duration_str)
    if hour_match:
        hours = int(hour_match.group(1))
    
    # Match minutes (M)
    minute_match = re.search(r'(\d+)M', duration_str)
    if minute_match:
        minutes = int(minute_match.group(1))
    
    # Match seconds (S)
    second_match = re.search(r'(\d+)S', duration_str)
    if second_match:
        seconds = int(second_match.group(1))
        # Convert seconds to minutes if >= 60
        if seconds >= 60:
            minutes += seconds // 60
            seconds = seconds % 60
    
    # Build human-readable string
    parts = []
    if hours > 0:
        parts.append(f"{hours} hr{'s' if hours > 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} min{'s' if minutes > 1 else ''}")
    if seconds > 0 and hours == 0 and minutes == 0:  # Only show seconds if no hours/minutes
        parts.append(f"{seconds} sec{'s' if seconds > 1 else ''}")
    
    return " ".join(parts) if parts else ""


def normalize_recipe_from_jsonld(recipe_obj: dict, soup: BeautifulSoup):
    name = recipe_obj.get("name") or ""
    image = recipe_obj.get("image")
    if isinstance(image, list) and image:
        image = image[0]
    if isinstance(image, dict):
        image = image.get("url")

    if not image:
        image = extract_og_image(soup)

    ingredients = recipe_obj.get("recipeIngredient") or []
    norm_ingredients = []
    for ing in ingredients:
        if isinstance(ing, str):
            norm_ingredients.append({"name": ing, "quantity": ""})

    # instructions can be list of strings or HowToStep objects
    # --- ✅ instructions (FIXED) ---
    instructions_raw = recipe_obj.get("recipeInstructions") or []
    norm_steps = _flatten_instructions(instructions_raw)

    # Normalize servings (can be string, number, or array)
    servings_raw = recipe_obj.get("recipeYield") or ""
    if isinstance(servings_raw, list):
        # If array, take the most descriptive one (usually the last)
        servings = str(servings_raw[-1]) if servings_raw else ""
    elif isinstance(servings_raw, (int, float)):
        servings = str(servings_raw)
    else:
        servings = str(servings_raw) if servings_raw else ""
    
    # Normalize time durations from ISO 8601 format to human-readable
    prep_raw = recipe_obj.get("prepTime") or ""
    prep = normalize_duration(prep_raw) if prep_raw else ""
    
    cook_raw = recipe_obj.get("cookTime") or ""
    cook = normalize_duration(cook_raw) if cook_raw else ""
    
    total_raw = recipe_obj.get("totalTime") or ""
    total = normalize_duration(total_raw) if total_raw else ""

    nutrition = recipe_obj.get("nutrition") or {}
    norm_nutrition = {}
    if isinstance(nutrition, dict):
        for key in _RECIPE_NUTRITION_KEYS:
            parsed = _parse_recipe_nutrition_value(
                nutrition.get(key) or nutrition.get(key.replace("_g", "")),
                calories=(key == "calories"),
            )
            if parsed:
                norm_nutrition[key] = parsed
        # JSON-LD may use proteinContent, carbohydrateContent, fatContent
        alias_map = {
            "protein_g": ("proteinContent", "protein"),
            "carbs_g": ("carbohydrateContent", "carbs"),
            "fat_g": ("fatContent", "fat"),
            "calories": ("calories",),
        }
        for key, aliases in alias_map.items():
            if key in norm_nutrition:
                continue
            for alias in aliases:
                parsed = _parse_recipe_nutrition_value(
                    nutrition.get(alias), calories=(key == "calories")
                )
                if parsed:
                    norm_nutrition[key] = parsed
                    break

    # Cuisine: recipeCuisine can be string or list in JSON-LD
    cuisine_raw = recipe_obj.get("recipeCuisine") or ""
    if isinstance(cuisine_raw, list) and cuisine_raw:
        cuisine = ", ".join(str(x).strip() for x in cuisine_raw if x)
    else:
        cuisine = str(cuisine_raw).strip() if cuisine_raw else ""

    # Meal type: recipeCategory often has "Breakfast", "Lunch", etc.
    meal_type_raw = recipe_obj.get("recipeCategory") or ""
    if isinstance(meal_type_raw, list) and meal_type_raw:
        meal_type_raw = meal_type_raw[0] if meal_type_raw else ""
    meal_type_raw = str(meal_type_raw).strip() if meal_type_raw else ""

    return {
        "name": name,
        "ingredients": norm_ingredients,
        "instructions": norm_steps,
        "servings": servings,
        "prep_time": prep,
        "cook_time": cook,
        "total_time": total,
        "nutrition": norm_nutrition,
        "cuisine": cuisine,
        "meal_type": meal_type_raw,
    }, image
def clean_page_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:40000]  # cap to avoid huge prompts

def extract_recipe_from_webpage_llm(page_text: str):
    system = f"""Extract recipe data from webpage text.
Return ONLY valid JSON:
{
  "name": "",
  "ingredients": [{"name":"", "quantity":""}],
  "instructions": ["Step 1 ...", "..."],
  "servings": "",
  "prep_time": "",
  "cook_time": "",
  "total_time": "",
  "notes": [],
  "meal_type": "",
  "cuisine": "",
  "nutrition": {
    "calories": "",
    "protein_g": "",
    "carbs_g": "",
    "fat_g": ""
  }
}
Rules:
- Don't hallucinate for the structure. If fields like servings or times are unknown, use "" or [].
- meal_type: exactly one of Breakfast, Lunch, Dinner, Snack (infer from context).
- Infer cuisine from recipe name, ingredients, or context when evident (e.g. Italian, Mexican, Indian, American); otherwise use "".
- You MUST provide a best-effort numeric estimate (as strings) for nutrition macros PER SERVING: calories, protein_g, carbs_g, fat_g. Use your nutrition knowledge of typical ingredients/quantities to approximate. Only leave a macro field \"\" if there is literally no information about ingredients.
- {_RECIPE_NUTRITION_PROMPT_RULE}
- Return JSON only."""
    user = f"Webpage text:\n{page_text}"

    completion = client.chat.completions.create(
        model=RECIPE_TEXT_MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.2,
        max_tokens=1800,
        response_format={"type":"json_object"},
        timeout=EXTRACT_RECIPE_TIMEOUT,
    )
    return json.loads(completion.choices[0].message.content)


# Max images per request for recipe extraction (avoids token/rate limits)
MAX_EXTRACT_RECIPE_IMAGES = int(os.getenv("MAX_EXTRACT_RECIPE_IMAGES", "10"))

def _get_image_data_urls_from_extract_request():
    """Get one or more images as data URLs. Multipart: field 'image' (single or multiple files). JSON: imageBase64/image or images[]. Returns (list of data_urls, None) or (None, error_response_tuple)."""
    format_to_mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "gif": "image/gif", "webp": "image/webp"}

    def mime_for_filename(filename):
        f = (filename or "").lower()
        if f.endswith((".jpg", ".jpeg")): return "image/jpeg"
        if f.endswith(".png"): return "image/png"
        if f.endswith(".gif"): return "image/gif"
        if f.endswith(".webp"): return "image/webp"
        return "image/jpeg"

    # Multipart: multiple files under 'image' (e.g. <input name="image" multiple> or multiple fields)
    if "image" in request.files:
        urls = []
        for uploaded_file in request.files.getlist("image"):
            if not uploaded_file or not uploaded_file.filename:
                continue
            image_bytes = uploaded_file.read()
            if not image_bytes:
                continue
            content_type = mime_for_filename(uploaded_file.filename)
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            urls.append(f"data:{content_type};base64,{b64}")
        if urls:
            if len(urls) > MAX_EXTRACT_RECIPE_IMAGES:
                return None, (jsonify({"error": f"Too many images; max {MAX_EXTRACT_RECIPE_IMAGES}"}), 400)
            return urls, None
        # single file (legacy): some clients send one file without getlist
        single = request.files["image"]
        if single and single.filename:
            single.seek(0)
            image_bytes = single.read()
            if image_bytes:
                content_type = mime_for_filename(single.filename)
                b64 = base64.b64encode(image_bytes).decode("utf-8")
                return [f"data:{content_type};base64,{b64}"], None
        # Empty multipart image field — fall through so URL-based extraction can proceed.
        return None, None

    # JSON: images array or single imageBase64/image
    if request.is_json:
        data = request.get_json(silent=True) or {}
        images_list = data.get("images")
        if isinstance(images_list, list) and images_list:
            urls = []
            for i, item in enumerate(images_list):
                if i >= MAX_EXTRACT_RECIPE_IMAGES:
                    break
                if isinstance(item, dict):
                    b64 = item.get("imageBase64") or item.get("image")
                    fmt = (item.get("imageFormat") or "jpg").lower()
                elif isinstance(item, str):
                    b64, fmt = item, "jpg"
                else:
                    continue
                if not b64:
                    continue
                if "," in str(b64):
                    b64 = str(b64).split(",")[-1]
                content_type = format_to_mime.get(fmt, "image/jpeg")
                urls.append(f"data:{content_type};base64,{b64}")
            if urls:
                return urls, None
        # Single image (legacy)
        image_base64 = data.get("imageBase64") or data.get("image")
        image_format = (data.get("imageFormat") or "jpg").lower()
        if image_base64:
            if "," in str(image_base64):
                image_base64 = str(image_base64).split(",")[-1]
            content_type = format_to_mime.get(image_format, "image/jpeg")
            return [f"data:{content_type};base64,{image_base64}"], None
    return None, None


def _get_extract_recipe_url() -> str | None:
    """Read recipe URL from JSON body or multipart/form fields (mobile clients vary)."""
    data = request.get_json(silent=True) or {}
    url = data.get("url") or data.get("videoUrl") or data.get("recipeUrl")
    if not url:
        url = (
            request.form.get("url")
            or request.form.get("videoUrl")
            or request.form.get("recipeUrl")
        )
    if not url:
        return None
    return str(url).strip()


def _get_extract_recipe_mode() -> str:
    data = request.get_json(silent=True) or {}
    mode = data.get("mode") or request.form.get("mode") or "auto"
    return str(mode).lower()


def _get_extract_recipe_include_image() -> bool:
    """When false, omit source.image from /extract-recipe responses (default true)."""
    data = request.get_json(silent=True) or {}
    raw = data.get("include_image")
    if raw is None:
        raw = data.get("includeImage")
    if raw is None:
        raw = request.form.get("include_image") or request.args.get("include_image")
    if raw is None:
        return True
    return str(raw).strip().lower() in ("1", "true", "yes")


def _get_extract_recipe_no_cache() -> bool:
    """When true, skip _recipe_cache lookup for this request."""
    data = request.get_json(silent=True) or {}
    raw = data.get("no_cache")
    if raw is None:
        raw = data.get("noCache")
    if raw is None:
        raw = request.form.get("no_cache") or request.args.get("no_cache")
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("1", "true", "yes")


def _start_persist_image(image_url: str | None):
    """Kick off the thumbnail upload in the background so it overlaps extraction.
    Returns a Future (resolving to the Storage URL or None), or None if no URL."""
    if not image_url or not isinstance(image_url, str):
        return None
    try:
        return _image_persist_executor.submit(persist_extract_recipe_image, image_url)
    except Exception as e:
        print(f"[extract-image] could not schedule persist: {e}")
        return None


def _persist_recipe_source_image(source: dict | None, meta: dict | None = None, *, future=None) -> None:
    """Re-host the recipe image in MealMap Storage so the URL never expires, and
    rewrite source['image'] / meta['thumbnail'] in place to the stable Storage URL.

    If `future` (from _start_persist_image) is given, we wait on the already
    in-flight upload instead of starting a new one — so the upload overlaps the
    LLM call and adds ~0s to the request. Otherwise it uploads synchronously.
    Always runs so the image is persisted even when include_image=false.
    Best-effort: on failure the original URLs are left untouched.
    """
    original = None
    if isinstance(source, dict) and source.get("image"):
        original = source["image"]
    elif isinstance(meta, dict) and meta.get("thumbnail"):
        original = meta["thumbnail"]
    if not original:
        return
    try:
        if future is not None:
            stored = future.result(timeout=EXTRACT_RECIPE_TIMEOUT)
        else:
            stored = persist_extract_recipe_image(original)
    except Exception as e:
        print(f"[extract-image] persist error: {e}")
        return
    if not stored:
        return
    if isinstance(source, dict) and source.get("image") == original:
        source["image"] = stored
    if isinstance(meta, dict) and meta.get("thumbnail") == original:
        meta["thumbnail"] = stored


def _apply_extract_recipe_image_pref(payload: dict) -> dict:
    """Drop source.image when the client sets include_image=false."""
    if _get_extract_recipe_include_image():
        return payload
    source = payload.get("source")
    if not isinstance(source, dict) or "image" not in source:
        return payload
    out = dict(payload)
    out["source"] = {k: v for k, v in source.items() if k != "image"}
    return out


def extract_recipe_from_image_llm(image_data_url: str):
    """Extract recipe from a single image. Returns same structure as extract_recipe_from_webpage_llm."""
    return extract_recipe_from_images_llm([image_data_url])


def extract_recipe_from_video_frames_llm(
    image_data_urls: list, *, caption: str = "", social: bool = False, language: str | None = None
) -> dict:
    """Vision path for video frame(s). social=True uses high detail for on-screen recipe overlays."""
    if not image_data_urls:
        raise ValueError("At least one image is required")
    caption_hint = ""
    if caption and caption.strip():
        caption_hint = f" Post metadata caption (may be partial): {caption.strip()[:2000]}"
    lang_rule = _source_language_prompt_rule() if social else ""
    if social:
        system = (
            "Extract a COMPLETE recipe from TikTok/Instagram cooking video frame(s). "
            "Creators often burn the full recipe as on-screen text overlay (ingredients + numbered steps). "
            "Read ALL visible overlay text across ALL frames and merge into ONE recipe. "
            "Do not invent or simplify — extract every ingredient with quantity and every step. "
            "Return ONLY JSON: "
            '{"language":"","name":"","ingredients":[{"name":"","quantity":""}],'
            '"instructions":["Step 1: ...","Step 2: ..."],'
            '"servings":"","prep_time":"","cook_time":"","total_time":"",'
            '"notes":[],"meal_type":"","cuisine":"","nutrition":{"calories":"","protein_g":"","carbs_g":"","fat_g":""}}. '
            f"meal_type: Breakfast|Lunch|Dinner|Snack. {_RECIPE_NUTRITION_PROMPT_RULE}{lang_rule} JSON only."
        )
        user_text = (
            f"These are {len(image_data_urls)} frame(s) from a short-form cooking video. "
            f"Read every line of on-screen recipe text and return the full recipe JSON.{caption_hint}"
        )
        # Low detail when caption already provides context — much faster than high on multiple frames.
        detail = "low" if (caption_hint and len(image_data_urls) <= 2) else "high"
        max_tokens = 1400 if detail == "low" else 1600
    else:
        system = (
            "Extract recipe from cooking video frame(s). Return ONLY JSON: "
            '{"name":"","ingredients":[{"name":"","quantity":""}],'
            '"instructions":["..."],"servings":"","prep_time":"","cook_time":"","total_time":"",'
            '"notes":[],"meal_type":"","cuisine":"","nutrition":{"calories":"","protein_g":"","carbs_g":"","fat_g":""}}. '
            "Read on-screen text and visible food. meal_type: Breakfast|Lunch|Dinner|Snack. "
            f"{_RECIPE_NUTRITION_PROMPT_RULE} JSON only."
        )
        user_text = "Extract the recipe JSON from this video frame."
        detail = "low"
        max_tokens = 900
    content = [{"type": "text", "text": user_text}]
    for url in image_data_urls:
        content.append({
            "type": "image_url",
            "image_url": {"url": url, "detail": detail},
        })
    completion = client.chat.completions.create(
        model=os.getenv("RECIPE_VISION_FAST_MODEL") or os.getenv("RECIPE_LLM_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        temperature=0.1,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        timeout=EXTRACT_RECIPE_TIMEOUT,
    )
    return json.loads(completion.choices[0].message.content)


def extract_recipe_from_slideshow_llm(
    image_data_urls: list, *, caption: str = "", language: str | None = None
) -> dict:
    """Fast vision path for social slideshows: low detail, compact prompt, fewer tokens."""
    if not image_data_urls:
        raise ValueError("At least one image is required")
    caption_hint = f" Post caption/title: {caption.strip()}." if caption and caption.strip() else ""
    lang_rule = _source_language_prompt_rule()
    system = (
        "Extract a complete recipe from social-media slideshow slide(s) with on-screen text overlays."
        " Combine all slides into ONE recipe. Return ONLY JSON: "
        '{"language":"","name":"","ingredients":[{"name":"","quantity":""}],'
        '"instructions":["..."],"servings":"","prep_time":"","cook_time":"","total_time":"",'
        '"notes":[],"meal_type":"","cuisine":"","nutrition":{"calories":"","protein_g":"","carbs_g":"","fat_g":""}}. '
        "Read every visible ingredient and cooking step across slides. "
        f"meal_type: Breakfast|Lunch|Dinner|Snack. {_RECIPE_NUTRITION_PROMPT_RULE}{lang_rule} JSON only."
    )
    content = [{
        "type": "text",
        "text": (
            f"These are {len(image_data_urls)} slide(s) from a recipe carousel (sampled from a longer slideshow)."
            f"{caption_hint} Extract and merge the full recipe JSON."
        ),
    }]
    for url in image_data_urls:
        content.append({
            "type": "image_url",
            "image_url": {"url": url, "detail": "low"},
        })
    completion = client.chat.completions.create(
        model=os.getenv("RECIPE_VISION_FAST_MODEL") or os.getenv("RECIPE_LLM_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        temperature=0.1,
        max_tokens=900,
        response_format={"type": "json_object"},
        timeout=EXTRACT_RECIPE_TIMEOUT,
    )
    return json.loads(completion.choices[0].message.content)


def extract_recipe_from_images_llm(image_data_urls: list):
    """Extract one combined recipe from one or more images (e.g. multi-page recipe). Uses vision LLM."""
    if not image_data_urls:
        raise ValueError("At least one image is required")
    system = f"""Extract recipe data from the image(s). If multiple images are provided (e.g. multiple pages), combine them into ONE recipe.
Return ONLY valid JSON:
{
  "name": "",
  "ingredients": [{"name":"", "quantity":""}],
  "instructions": ["Step 1 ...", "..."],
  "servings": "",
  "prep_time": "",
  "cook_time": "",
  "total_time": "",
  "notes": [],
  "meal_type": "",
  "cuisine": "",
  "nutrition": {
    "calories": "",
    "protein_g": "",
    "carbs_g": "",
    "fat_g": ""
  }
}
Rules:
- Don't hallucinate. If unknown, use "" or [].
- meal_type: exactly one of Breakfast, Lunch, Dinner, Snack (infer from context).
- Infer cuisine from recipe name, ingredients, or visible context when evident (e.g. Italian, Mexican, Indian); otherwise use "".
- You MUST provide a best-effort numeric estimate (as strings) for nutrition macros PER SERVING: calories, protein_g, carbs_g, fat_g. Use your nutrition knowledge of typical ingredients/quantities to approximate from what you see. Only leave a macro field \"\" if there is literally no information about ingredients.
- {_RECIPE_NUTRITION_PROMPT_RULE}
- Return JSON only. Read all text visible across the images. Merge ingredients and instructions from all pages into one recipe."""
    content = [{"type": "text", "text": "Extract the recipe from these image(s) and return the JSON. If there are multiple images, treat them as one multi-page recipe and merge into a single recipe."}]
    for url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    completion = client.chat.completions.create(
        model=os.getenv("RECIPE_LLM_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        temperature=0.2,
        max_tokens=1800,
        response_format={"type": "json_object"},
        timeout=EXTRACT_RECIPE_TIMEOUT,
    )
    return json.loads(completion.choices[0].message.content)


# Allowed values for /extract-recipe response attributes
MEAL_TYPES = ("Breakfast", "Lunch", "Dinner", "Snack")
CUISINES = ("Italian", "Mexican", "American", "Asian", "Mediterranean", "Indian", "Chinese", "Japanese", "Thai", "Middle Eastern")
DIET_FLAGS = ("High Protein", "High Fiber", "Low Carb", "Keto", "Vegetarian", "Vegan", "Gluten Free", "Balanced")


def normalize_meal_type(raw: str) -> str:
    """Map free-text meal type to one of Breakfast, Lunch, Dinner, Snack."""
    if not raw or not isinstance(raw, str):
        return "Lunch / Dinner"  # default
    s = raw.strip().lower()
    if not s:
        return "Lunch / Dinner"
    if s in ("breakfast", "brunch") or "breakfast" in s or "brunch" in s:
        return "Breakfast"
    if s in ("lunch", "brunch") or "lunch" in s:
        return "Lunch"
    if s in ("dinner", "supper", "main") or "dinner" in s or "supper" in s:
        return "Dinner"
    if s in ("snack", "appetizer", "appetiser", "side", "dessert") or "snack" in s:
        return "Snack"
    return "Lunch / Dinner"


def normalize_cuisine(raw: str) -> str:
    """Map free-text cuisine to one of the allowed CUISINES."""
    if not raw or not isinstance(raw, str):
        return "American"
    s = raw.strip().lower()
    if not s:
        return "American"
    # Direct and fuzzy matches
    mapping = [
        ("italian", "Italian"), ("mexican", "Mexican"), ("american", "American"),
        ("asian", "Asian"), ("mediterranean", "Mediterranean"), ("indian", "Indian"),
        ("chinese", "Chinese"), ("japanese", "Japanese"), ("japenese", "Japanese"),
        ("thai", "Thai"), ("middle eastern", "Middle Eastern"), ("middle east", "Middle Eastern"),
        ("korean", "Asian"), ("vietnamese", "Asian"), ("greek", "Mediterranean"),
        ("spanish", "Mediterranean"), ("french", "Mediterranean"),
    ]
    for key, value in mapping:
        if key in s or s in key:
            return value
    # Fallback: if single word might be cuisine, try match
    for c in CUISINES:
        if c.lower() in s or s in c.lower():
            return c
    return "American"


def extract_recipe_diet_flags(recipe: dict) -> list[str]:
    """Return diet flags from fixed set: High Protein, High Fiber, Low Carb, Keto, Vegetarian, Vegan, Gluten Free, Balanced."""
    flags = []
    if not recipe:
        return flags
    ingredients = recipe.get("ingredients", [])
    nutrition = recipe.get("nutrition") or {}
    ingredient_names = []
    for ing in ingredients:
        if isinstance(ing, dict):
            ingredient_names.append(ing.get("name", "").lower())
        elif isinstance(ing, str):
            ingredient_names.append(ing.lower())
    ingredient_text = " ".join(ingredient_names)

    # Vegetarian (no meat/fish/poultry)
    meat_keywords = ["chicken", "beef", "pork", "lamb", "turkey", "duck", "fish", "salmon", "tuna", "shrimp", "crab", "lobster", "meat", "bacon", "sausage", "ham", "steak"]
    if not any(k in ingredient_text for k in meat_keywords) and ingredient_text:
        flags.append("Vegetarian")

    # Vegan (no animal products)
    animal_keywords = meat_keywords + ["milk", "cheese", "butter", "cream", "yogurt", "egg", "honey", "gelatin", "whey", "casein"]
    if not any(k in ingredient_text for k in animal_keywords) and ingredient_text:
        flags.append("Vegan")

    # High Protein
    protein_val = nutrition.get("protein") or nutrition.get("proteinContent") or nutrition.get("protein_g")
    if isinstance(protein_val, str):
        match = re.search(r"(\d+\.?\d*)", protein_val)
        protein_val = float(match.group(1)) if match else None
    high_protein_ings = ["chicken", "beef", "turkey", "fish", "salmon", "tuna", "eggs", "tofu", "tempeh", "lentils", "beans", "chickpeas", "quinoa", "greek yogurt", "cottage cheese", "protein powder"]
    if (protein_val and protein_val > 20) or any(ing in ingredient_text for ing in high_protein_ings):
        flags.append("High Protein")

    # High Fiber
    fiber_ings = ["oats", "lentils", "beans", "chickpeas", "quinoa", "barley", "broccoli", "avocado", "chia", "flax", "whole grain", "brown rice"]
    if any(ing in ingredient_text for ing in fiber_ings):
        flags.append("High Fiber")

    # Low Carb / Keto (heuristic: few grains/sugar)
    carb_heavy = ["flour", "pasta", "rice", "bread", "sugar", "potato", "oat", "quinoa", "corn", "beans"]
    carb_count = sum(1 for c in carb_heavy if c in ingredient_text)
    if carb_count <= 1 and ingredient_text:
        flags.append("Low Carb")
    if carb_count <= 1 and any(f in ingredient_text for f in ["avocado", "cheese", "cream", "olive oil", "coconut"]):
        flags.append("Keto")

    # Gluten Free (no wheat, barley, rye)
    if not any(g in ingredient_text for g in ["wheat", "flour", "barley", "rye", "bread", "pasta", "couscous"]):
        flags.append("Gluten Free")

    # Balanced (default if nothing else or general)
    if not flags or len(flags) >= 2:
        flags.append("Balanced")

    # Return only from allowed set, deduplicated, order preserved
    return list(dict.fromkeys(f for f in flags if f in DIET_FLAGS))


def _short_recipe_description(recipe: dict, *, language: str | None = None) -> str:
    """
    Build a short, one-line summary of the recipe (max 100 chars) for /extract-recipe response.
    Uses recipe name + diet/style flags. Ends with a full stop; no ellipsis.
    """
    if not recipe:
        return ""
    name = (recipe.get("name") or "").strip()
    if language and language != "en":
        desc = name or "Receta"
    else:
        diet_flags = recipe.get("diet_flags") or []
        diet_words = " ".join(str(d).lower() for d in diet_flags if d).strip()
        if diet_words and name:
            desc = f"{diet_words} {name}"
        elif name:
            desc = name
        else:
            desc = diet_words or "Recipe"
    desc = re.sub(r"\s+", " ", desc).strip()
    if len(desc) > 100:
        desc = desc[:99].rstrip()
        if desc and not desc.endswith("."):
            desc = desc + "."
    elif desc and not desc.endswith("."):
        desc = desc + "."
    return desc


_PREP_TIME_WORDS = (
    "marinate", "soak", "rest", "chill", "refrigerat", "proof", "rise",
    "prep", "prepare", "chop", "dice", "slice", "peel", "grate", "mix",
    "combine", "coat", "season", "toss", "assemble",
)
_COOK_TIME_WORDS = (
    "fry", "bake", "roast", "grill", "simmer", "boil", "sauté", "saute",
    "cook", "heat", "microwave", "broil", "steam", "reduce", "brown",
    "crisp", "golden", "pan-fry", "pan fry", "oven", "air fry",
)


def _metadata_fill_enabled() -> bool:
    return (os.getenv("EXTRACT_RECIPE_FILL_METADATA") or "1").strip().lower() not in (
        "0", "false", "no",
    )


def _recipe_has_named_ingredients(recipe: dict) -> bool:
    for ing in recipe.get("ingredients") or []:
        if isinstance(ing, dict) and (ing.get("name") or "").strip():
            return True
        if isinstance(ing, str) and ing.strip():
            return True
    return False


def _minutes_in_instruction_text(text: str) -> int:
    """Parse minute/hour mentions from a single instruction line."""
    lower = text.lower()
    minutes = 0
    for m in re.finditer(r"(\d+)\s*(?:h|hr|hrs|hour|hours)\b", lower):
        minutes += int(m.group(1)) * 60
    range_spans: list[tuple[int, int]] = []
    for m in re.finditer(r"(\d+)\s*[-–]\s*(\d+)\s*(?:min|mins|minutes|minute|m)\b", lower):
        minutes += (int(m.group(1)) + int(m.group(2))) // 2
        range_spans.append(m.span())
    for m in re.finditer(r"(\d+)\s*(?:min|mins|minutes|minute|m)\b", lower):
        if any(start <= m.start() < end for start, end in range_spans):
            continue
        minutes += int(m.group(1))
    return minutes


def _infer_times_from_instructions(instructions: list) -> tuple[str, str]:
    """Infer prep/cook durations from time mentions inside instruction steps."""
    prep_total = 0
    cook_total = 0
    for step in instructions:
        text = str(step).strip()
        if not text:
            continue
        mins = _minutes_in_instruction_text(text)
        if mins <= 0:
            continue
        lower = text.lower()
        is_cook = any(w in lower for w in _COOK_TIME_WORDS)
        is_prep = any(w in lower for w in _PREP_TIME_WORDS)
        if is_cook:
            cook_total += mins
        elif is_prep:
            prep_total += mins
        else:
            cook_total += mins
    prep_str = f"{prep_total} mins" if prep_total else ""
    cook_str = f"{cook_total} mins" if cook_total else ""
    return prep_str, cook_str


def _format_minutes_label(minutes: int) -> str:
    if minutes <= 0:
        return ""
    if minutes >= 60 and minutes % 60 == 0:
        hrs = minutes // 60
        return f"{hrs} hr" if hrs == 1 else f"{hrs} hrs"
    return f"{minutes} mins"


_RECIPE_NUTRITION_KEYS = ("calories", "protein_g", "carbs_g", "fat_g")

_RECIPE_NUTRITION_PROMPT_RULE = (
    "Nutrition macros (calories, protein_g, carbs_g, fat_g) are PER SERVING and MUST be numeric strings only "
    '(e.g. "120", "6.5"). Never use words like variable, approximate, varies, unknown, or descriptive text.'
)

_INVALID_NUTRITION_TEXT = (
    "variable", "varies", "approximate", "approx", "unknown", "n/a", "na", "tbd",
    "estimate", "depends", "not available", "per serving", "per piece", "per item",
)


def _format_recipe_nutrition_number(value: float, *, calories: bool = False) -> str:
    if calories:
        return str(int(round(value)))
    if abs(value - round(value)) < 0.05:
        return str(int(round(value)))
    return str(round(value, 1))


def _parse_recipe_nutrition_value(value, *, calories: bool = False) -> str | None:
    """Return a normalized numeric macro string, or None when value is missing/invalid."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if number < 0:
            return None
        return _format_recipe_nutrition_number(number, calories=calories)

    text = str(value).strip()
    if not text:
        return None

    lower = text.lower()
    if not re.search(r"\d", text) and any(token in lower for token in _INVALID_NUTRITION_TEXT):
        return None

    match = re.search(r"(\d+(?:\.\d+)?)", text.replace(",", ""))
    if not match:
        return None

    number = float(match.group(1))
    if number < 0:
        return None
    return _format_recipe_nutrition_number(number, calories=calories)


def _recipe_nutrition_macro_missing(value) -> bool:
    return _parse_recipe_nutrition_value(value) is None


def _normalize_recipe_nutrition_fields(nutrition: dict) -> set[str]:
    """Normalize macro strings in place. Returns keys still missing a valid numeric value."""
    missing: set[str] = set()
    for key in _RECIPE_NUTRITION_KEYS:
        parsed = _parse_recipe_nutrition_value(
            nutrition.get(key), calories=(key == "calories")
        )
        if parsed is None:
            nutrition.pop(key, None)
            missing.add(key)
        else:
            nutrition[key] = parsed
    return missing


def _derive_calories_from_macros(nutrition: dict) -> bool:
    """Fill calories from protein/carbs/fat when all three are known."""
    protein = _parse_recipe_nutrition_value(nutrition.get("protein_g"))
    carbs = _parse_recipe_nutrition_value(nutrition.get("carbs_g"))
    fat = _parse_recipe_nutrition_value(nutrition.get("fat_g"))
    if not (protein and carbs and fat):
        return False
    calories = 4 * float(protein) + 4 * float(carbs) + 9 * float(fat)
    nutrition["calories"] = _format_recipe_nutrition_number(calories, calories=True)
    return True


def _estimate_recipe_nutrition_only_llm(recipe: dict, *, language: str | None = None) -> dict:
    """Focused nutrition estimate when extraction returned placeholders or partial macros."""
    servings = str(recipe.get("servings") or DEFAULT_RECIPE_SERVINGS).strip()
    payload = {
        "name": recipe.get("name") or "",
        "servings": servings,
        "ingredients": recipe.get("ingredients") or [],
        "instructions": recipe.get("instructions") or [],
    }
    lang_rule = _metadata_language_rule(language)
    system = (
        "Estimate per-serving nutrition macros for a recipe from its ingredients and instructions. "
        "Return ONLY JSON: "
        '{"calories":"","protein_g":"","carbs_g":"","fat_g":""}. '
        f"Every field MUST be a numeric string. {_RECIPE_NUTRITION_PROMPT_RULE}{lang_rule} JSON only."
    )
    user = f"Recipe:\n{json.dumps(payload, ensure_ascii=False)}\n\nReturn numeric per-serving macros."
    try:
        completion = client.chat.completions.create(
            model=RECIPE_LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=180,
            response_format={"type": "json_object"},
            timeout=EXTRACT_RECIPE_TIMEOUT,
        )
        data = json.loads(completion.choices[0].message.content.strip())
        out = {}
        for key in _RECIPE_NUTRITION_KEYS:
            parsed = _parse_recipe_nutrition_value(
                (data or {}).get(key), calories=(key == "calories")
            )
            if parsed:
                out[key] = parsed
        return out
    except Exception as e:
        print(f"⚠️ Recipe nutrition estimation failed: {e}")
        return {}


def _ensure_recipe_nutrition_macros(recipe: dict, *, language: str | None = None) -> None:
    """Guarantee nutrition macros are numeric strings, estimating when needed."""
    if not recipe:
        return

    nutrition = recipe.get("nutrition")
    if not isinstance(nutrition, dict):
        nutrition = {}
        recipe["nutrition"] = nutrition

    missing = _normalize_recipe_nutrition_fields(nutrition)
    if "calories" in missing and _derive_calories_from_macros(nutrition):
        missing.discard("calories")

    if not missing:
        return
    if not _recipe_has_named_ingredients(recipe):
        return

    estimated = _estimate_recipe_nutrition_only_llm(recipe, language=language)
    for key in list(missing):
        if estimated.get(key):
            nutrition[key] = estimated[key]

    missing = _normalize_recipe_nutrition_fields(nutrition)
    if "calories" in missing and _derive_calories_from_macros(nutrition):
        missing.discard("calories")

    # Optional second attempt (off by default): a retry is a full extra LLM call
    # on the request path for marginal gain. Enable with NUTRITION_ESTIMATE_RETRY=1.
    _retry_nutrition = (os.getenv("NUTRITION_ESTIMATE_RETRY") or "").strip().lower() in ("1", "true", "yes")
    if _retry_nutrition and missing and _recipe_has_named_ingredients(recipe):
        retry = _estimate_recipe_nutrition_only_llm(recipe, language=language)
        for key in list(missing):
            if retry.get(key):
                nutrition[key] = retry[key]
        missing = _normalize_recipe_nutrition_fields(nutrition)
        if "calories" in missing and _derive_calories_from_macros(nutrition):
            missing.discard("calories")

    for key in missing:
        nutrition[key] = "0"


def _ensure_cached_recipe_nutrition(payload: dict) -> None:
    recipe = payload.get("recipe")
    if isinstance(recipe, dict):
        lang = _normalize_language_code(payload.get("language") or recipe.get("language"))
        _ensure_recipe_nutrition_macros(recipe, language=lang)


def _estimate_missing_recipe_metadata_llm(recipe: dict, *, language: str | None = None) -> dict:
    """Estimate prep/cook times and per-serving nutrition when the source omits them."""
    servings = str(recipe.get("servings") or DEFAULT_RECIPE_SERVINGS).strip()
    payload = {
        "name": recipe.get("name") or "",
        "servings": servings,
        "ingredients": recipe.get("ingredients") or [],
        "instructions": recipe.get("instructions") or [],
    }
    lang_rule = _metadata_language_rule(language)
    system = (
        "Estimate realistic home-cooking metadata for a recipe. "
        "Return ONLY JSON: "
        '{"prep_time":"","cook_time":"","total_time":"",'
        '"calories":"","protein_g":"","carbs_g":"","fat_g":""}. '
        "Times must be human-readable strings like \"15 mins\" or \"1 hr\" (not ISO). "
        "Infer prep_time from chopping/mixing/coating steps; cook_time from frying/baking/boiling steps. "
        "total_time should equal prep + cook. "
        f"{_RECIPE_NUTRITION_PROMPT_RULE}{lang_rule} JSON only."
    )
    user = f"Recipe:\n{json.dumps(payload, ensure_ascii=False)}\n\nReturn estimates."
    try:
        completion = client.chat.completions.create(
            model=RECIPE_LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=250,
            response_format={"type": "json_object"},
            timeout=EXTRACT_RECIPE_TIMEOUT,
        )
        data = json.loads(completion.choices[0].message.content.strip())
        out = {}
        for key in ("prep_time", "cook_time", "total_time"):
            val = str((data or {}).get(key) or "").strip()
            if val:
                out[key] = val
        for key in ("calories", "protein_g", "carbs_g", "fat_g"):
            parsed = _parse_recipe_nutrition_value(
                (data or {}).get(key), calories=(key == "calories")
            )
            if parsed:
                out[key] = parsed
        return out
    except Exception as e:
        print(f"⚠️ Recipe metadata estimation failed: {e}")
        return {}


def _recipe_field_empty(recipe: dict, key: str) -> bool:
    return not str(recipe.get(key) or "").strip()


def _fill_missing_recipe_metadata(recipe: dict, *, language: str | None = None) -> None:
    """Backfill only fields the extraction LLM left empty — never overwrite provided values."""
    if not recipe or not _metadata_fill_enabled():
        return

    # Snapshot which fields were empty in the extraction LLM response.
    fill_servings = _recipe_field_empty(recipe, "servings")
    fill_prep = _recipe_field_empty(recipe, "prep_time")
    fill_cook = _recipe_field_empty(recipe, "cook_time")
    fill_total = _recipe_field_empty(recipe, "total_time")

    nutrition = recipe.get("nutrition")
    if not isinstance(nutrition, dict):
        nutrition = {}
        recipe["nutrition"] = nutrition
    fill_nutrition = {
        key: _recipe_nutrition_macro_missing(nutrition.get(key))
        for key in _RECIPE_NUTRITION_KEYS
    }

    instructions = recipe.get("instructions") or []
    if isinstance(instructions, str):
        instructions = [instructions]

    if fill_servings:
        recipe["servings"] = str(DEFAULT_RECIPE_SERVINGS)

    if instructions and (fill_prep or fill_cook):
        infer_prep, infer_cook = _infer_times_from_instructions(instructions)
        if fill_prep and infer_prep:
            recipe["prep_time"] = infer_prep
        if fill_cook and infer_cook:
            recipe["cook_time"] = infer_cook

    need_times_llm = (
        (fill_prep and _recipe_field_empty(recipe, "prep_time"))
        or (fill_cook and _recipe_field_empty(recipe, "cook_time"))
        or (fill_total and _recipe_field_empty(recipe, "total_time"))
    )
    need_nutrition_llm = any(fill_nutrition.values()) and _recipe_has_named_ingredients(recipe)
    if (need_times_llm or need_nutrition_llm) and (instructions or _recipe_has_named_ingredients(recipe)):
        estimated = _estimate_missing_recipe_metadata_llm(recipe, language=language)
        if fill_prep and _recipe_field_empty(recipe, "prep_time") and estimated.get("prep_time"):
            recipe["prep_time"] = estimated["prep_time"]
        if fill_cook and _recipe_field_empty(recipe, "cook_time") and estimated.get("cook_time"):
            recipe["cook_time"] = estimated["cook_time"]
        if fill_total and _recipe_field_empty(recipe, "total_time") and estimated.get("total_time"):
            recipe["total_time"] = estimated["total_time"]
        for key, should_fill in fill_nutrition.items():
            if should_fill and estimated.get(key):
                nutrition[key] = estimated[key]

    if fill_total and _recipe_field_empty(recipe, "total_time"):
        prep_m = _time_str_to_minutes_for_cook_time(recipe.get("prep_time") or "")
        cook_m = _time_str_to_minutes_for_cook_time(recipe.get("cook_time") or "")
        total_m = prep_m + cook_m
        if total_m > 0:
            recipe["total_time"] = _format_minutes_label(total_m)


def _enrich_recipe_response(recipe: dict, *, language: str | None = None) -> None:
    """Set meal_type, cuisine, diet_flags, description, and inferred metadata on recipe."""
    if not recipe:
        return
    is_english = not language or language == "en"
    if is_english:
        recipe["meal_type"] = normalize_meal_type(recipe.get("meal_type"))
        recipe["cuisine"] = normalize_cuisine(recipe.get("cuisine"))
        recipe["diet_flags"] = extract_recipe_diet_flags(recipe)
    else:
        if not (recipe.get("meal_type") or "").strip():
            recipe["meal_type"] = ""
        if not (recipe.get("cuisine") or "").strip():
            recipe["cuisine"] = ""
        recipe["diet_flags"] = []
    _fill_missing_recipe_metadata(recipe, language=language)
    _ensure_recipe_nutrition_macros(recipe, language=language)
    recipe["description"] = _short_recipe_description(recipe, language=language)


def _guess_difficulty_from_tags(tags: list[str]) -> str:
    tags = [t.lower() for t in (tags or [])]
    if "easy" in tags:
        return "Easy"
    if "medium" in tags:
        return "Medium"
    if "hard" in tags:
        return "Hard"
    return "Easy"


def _time_str_to_minutes_for_cook_time(time_str: str) -> int:
    """
    Convert a human-readable time string into minutes (numeric only for cookTime).
    Handles formats like '20 mins', '1 hr 30 mins', '45 min', '2 hours', 'PT30M'.
    """
    if not time_str:
        return 0
    s = str(time_str).strip().lower()
    if not s:
        return 0

    total = 0

    # Hours
    m = re.search(r"(\d+)\s*(h|hr|hrs|hour|hours)", s)
    if m:
        total += int(m.group(1)) * 60

    # Minutes
    m = re.search(r"(\d+)\s*(m|min|mins|minute|minutes)", s)
    if m:
        total += int(m.group(1))

    # ISO 8601 PTxxHxxM
    if s.startswith("pt"):
        h = re.search(r"(\d+)h", s)
        m2 = re.search(r"(\d+)m", s)
        if h:
            total += int(h.group(1)) * 60
        if m2:
            total += int(m2.group(1))

    # Fallback: if still zero, try to parse first integer as minutes
    if total == 0:
        m = re.search(r"(\d+)", s)
        if m:
            total = int(m.group(1))

    return total


@app.route("/bulk-import-recipes", methods=["POST"])
def bulk_import_recipes():
    """
    Bulk operations for recipes using the recipe-log collection.

    - Default behavior (no `action`): accept a CSV file (form field `file`)
      with a `recipe_url` column. For each row, call /extract-recipe with that
      URL and save the result into the Firestore `recipe-log` collection.

    - Delete behavior (`action=delete` + `createdDate=YYYY-MM-DD`): delete all
      recipe-log documents whose createdAt falls on that date (UTC).
    """
    action = (request.form.get("action") or "").strip().lower()
    created_date_str = (request.form.get("createdDate") or "").strip()

    db = init_mealmap_firestore()
    collection = db.collection("recipe-log")

    # If action=delete, delete all entries for that created date and return.
    if action == "delete":
        if not created_date_str:
            return jsonify({"error": "createdDate is required when action=delete (format: YYYY-MM-DD)"}), 400
        try:
            day = datetime.fromisoformat(created_date_str).date()
        except ValueError:
            return jsonify({"error": "createdDate must be in YYYY-MM-DD format"}), 400

        start = datetime(day.year, day.month, day.day)
        end = start + timedelta(days=1)

        docs = collection.where("createdAt", ">=", start).where("createdAt", "<", end).stream()
        deleted = 0
        for doc in docs:
            doc.reference.delete()  
            deleted += 1

        return jsonify({
            "action": "delete",
            "createdDate": created_date_str,
            "deleted": deleted,
        })

    # Existing behavior: CSV import
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "CSV file is required (field name: 'file')"}), 400

    try:
        content = file.read().decode("utf-8")
    except Exception:
        return jsonify({"error": "Failed to read CSV file; must be UTF-8 text"}), 400

    reader = csv.DictReader(io.StringIO(content))
    if "recipe_url" not in reader.fieldnames:
        return jsonify({"error": "CSV must contain a 'recipe_url' column"}), 400

    # Base URL for this service (can point to Render, etc.)
    base_url = os.getenv("EXTRACT_RECIPE_BASE_URL", "http://localhost:5002")

    imported = 0
    failed = []

    for idx, row in enumerate(reader, start=1):
        url = (row.get("recipe_url") or "").strip()
        if not url:
            failed.append({"row": idx, "reason": "Empty recipe_url"})
            continue

        try:
            resp = requests.post(
                f"{base_url}/extract-recipe",
                json={"url": url, "mode": "auto"},
                timeout=120,
            )
        except Exception as e:
            failed.append({"row": idx, "url": url, "reason": f"Request error: {e}"})
            continue

        try:
            data = resp.json()
        except Exception:
            failed.append({"row": idx, "url": url, "reason": "Non-JSON response from /extract-recipe"})
            continue

        if not resp.ok or not data.get("recipe"):
            failed.append({
                "row": idx,
                "url": url,
                "reason": data.get("user_message") or data.get("error") or "extract-recipe failed",
            })
            continue

        recipe = data["recipe"]
        source = data.get("source") or {}
        tags = data.get("tags") or []

        # Ensure enrichment (just in case /extract-recipe behavior changes)
        _enrich_recipe_response(recipe)

        # Map to Firestore recipe-log schema (as per screenshot)
        recipe_id = str(uuid.uuid4())
        title = (recipe.get("name") or "").strip()
        cook_time_str = recipe.get("cook_time") or recipe.get("total_time", "")
        cook_time_minutes = _time_str_to_minutes_for_cook_time(cook_time_str)

        # Short description (max 100 chars, ends with full stop; already set by _enrich_recipe_response)
        desc = (recipe.get("description") or title or "").strip()
        if len(desc) > 100:
            desc = desc[:99].rstrip()
            if desc and not desc.endswith("."):
                desc = desc + "."
        elif desc and not desc.endswith("."):
            desc = desc + "."

        # Steps: Firestore format [{ instruction, order, duration }, ...] (plain text, no "1.", "2." in instruction)
        raw_instructions = recipe.get("instructions") or []
        steps = []
        for i, raw in enumerate(raw_instructions, start=1):
            text = (raw.get("instruction", raw) if isinstance(raw, dict) else str(raw)).strip()
            # Strip leading "1. ", "Step 1: ", etc.
            text = re.sub(r"^(?:\d+[.)]\s*|step\s*\d+\s*[.:]\s*)", "", text, flags=re.IGNORECASE).strip()
            steps.append({
                "instruction": text,
                "order": i,
                "duration": None,
            })

        doc = {
            "recipeId": recipe_id,
            "category": recipe.get("meal_type", "Dinner"),
            "cookTime": cook_time_minutes,
            "createdAt": datetime.utcnow(),
            "title": title,
            "description": desc,
            "difficulty": _guess_difficulty_from_tags(tags),
            "imageUrl": source.get("image") or "",
            "ingredients": [
                {
                    "name": ing.get("name", ""),
                    "amount": ing.get("quantity", ""),
                }
                for ing in (recipe.get("ingredients") or [])
                if isinstance(ing, dict)
            ],
            "isFavorite": False,
            "nutrition": recipe.get("nutrition") or {},
            "steps": steps,
            "sourceUrl": source.get("url") or url,
            "cuisine": recipe.get("cuisine", ""),
            "dietFlags": recipe.get("diet_flags", []),
            "tags": tags,
        }

        try:
            # Store under deterministic recipeId so each recipe has a stable ID
            collection.document(recipe_id).set(doc)
            imported += 1
        except Exception as e:
            failed.append({"row": idx, "url": url, "reason": f"Firestore error: {e}"})

    return jsonify({
        "imported": imported,
        "failed": failed,
    })


@app.route("/extract-recipe", methods=["POST"])
def extract_recipe():
    url = _get_extract_recipe_url()
    mode = _get_extract_recipe_mode()

    # Image input: multipart (field 'image') or JSON (imageBase64/images array)
    image_data_urls, image_error = _get_image_data_urls_from_extract_request()
    if image_error and not url:
        return image_error[0], image_error[1]

    if image_data_urls and not url:
        try:
            recipe = extract_recipe_from_images_llm(image_data_urls)
            _enrich_recipe_response(recipe)
            tags = extract_recipe_tags(recipe)
            source = {
                "type": "image",
                "url": None,
                "provider": None,
                "title": recipe.get("name", "") or "Recipe from image",
                "image": None,
                "source_type": "Photos",
            }
            return jsonify(_apply_extract_recipe_image_pref({
                "source": source,
                "recipe": recipe,
                "tags": tags,
                "transcript": None,
                "extraction": {"method": "image_vision", "confidence": 0.6},
            }))
        except Exception as e:
            return jsonify({
                "error": "Failed to extract recipe from image",
                "user_message": "We couldn't extract a recipe from this image. Please try another photo or add the recipe manually.",
                "details": str(e),
            }), 500

    if not url:
        print(f"[extract-recipe] 400: missing url (content-type={request.content_type})")
        return jsonify({
            "error": "url or image is required",
            "user_message": "Please provide a recipe URL or image.",
        }), 400

    ok, err = validate_video_url(url)
    if not ok:
        print(f"[extract-recipe] 400: invalid url: {err}")
        return jsonify({"error": err, "user_message": "That link doesn't look valid. Please check the URL."}), 400

    # Decide type
    url_is_video = is_video_url(url)
    if mode == "video":
        url_is_video = True
    elif mode == "webpage":
        url_is_video = False

    if url_is_video:
        # Call your existing video pipeline
        # (audio -> whisper -> chunk+merge LLM)
        return extract_recipe_from_video_internal(url)

    # Webpage pipeline
    # ── Cache check ──────────────────────────────────────────────────────────
    webpage_key = hashlib.sha256(url.encode()).hexdigest()
    if not _get_extract_recipe_no_cache():
        with _recipe_cache_lock:
            if webpage_key in _recipe_cache:
                print(f"⚡ Cache hit for webpage URL: {url}")
                cached = dict(_recipe_cache[webpage_key])
                _ensure_cached_recipe_nutrition(cached)
                return jsonify({**_apply_extract_recipe_image_pref(cached), "cached": True})
    # ─────────────────────────────────────────────────────────────────────────

    try:
        html = fetch_html(url)
        recipes, soup = extract_jsonld_recipes(html)

        image = None
        recipe = None
        method = None

        if recipes:
            recipe, image = normalize_recipe_from_jsonld(recipes[0], soup)
            method = "jsonld"
        else:
            # fallback to LLM on cleaned page text
            page_text = clean_page_text(html)
            recipe = extract_recipe_from_webpage_llm(page_text)
            image = extract_og_image(soup)
            method = "html_llm"

        # Determine source type and extract tags
        source_type = determine_source_type(url)
        _enrich_recipe_response(recipe)
        tags = extract_recipe_tags(recipe)

        source = {
            "type": "webpage",
            "url": url,
            "provider": (urlparse(url).hostname or ""),
            "title": recipe.get("name", "") or (soup.title.string.strip() if soup.title and soup.title.string else ""),
            "image": image,
            "source_type": source_type,
        }

        # Re-host the recipe image in MealMap Storage and rewrite source.image to
        # the stable Storage URL before caching, so both the response and the
        # cache carry the permanent URL.
        _persist_recipe_source_image(source)

        _result = {
            "source": source,
            "recipe": recipe,
            "tags": tags,
            "transcript": None,
            "extraction": {"method": method, "confidence": 0.7 if method == "jsonld" else 0.5},
        }

        # ── Cache write ──────────────────────────────────────────────────────
        with _recipe_cache_lock:
            _recipe_cache[webpage_key] = _result
        # ────────────────────────────────────────────────────────────────────

        return jsonify(_apply_extract_recipe_image_pref(_result))

    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else 502

        if status == 402:
            user_message = "We couldn't extract a recipe from this link. This site may be paywalled or block automated access—try another link or add the recipe manually."
        elif 400 <= status < 500:
            user_message = "We couldn't extract a recipe from this link. Please check the URL or try a different recipe page."
        else:
            user_message = "We couldn't extract a recipe from this link. Please try again later or add the recipe manually."

        return jsonify({
            "error": "WEBPAGE_FETCH_FAILED",
            "status": status,
            "user_message": user_message,
            "details": str(e),
        }), 400

    except Exception as e:
        return jsonify({
            "error": "UNEXPECTED_EXTRACT_RECIPE_ERROR",
            "user_message": "We couldn't extract a recipe from this link. Please try another link or add the recipe manually.",
            "details": str(e),
        }), 500



def _chunk_text(text: str, max_chars: int = 6000):
    text = (text or "").strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        cut = text.rfind(".", start, end)
        if cut == -1 or cut < start + int(max_chars * 0.6):
            cut = end
        chunks.append(text[start:cut].strip())
        start = cut
    return [c for c in chunks if c]


def _force_json(text: str) -> dict:
    if not text:
        raise ValueError("Empty LLM response")
    cleaned = re.sub(r"```json\s*|```", "", text).strip()
    # Heuristic: pull first JSON object if extra text exists
    if "{" in cleaned and "}" in cleaned:
        cleaned = cleaned[cleaned.find("{"): cleaned.rfind("}") + 1]
    return json.loads(cleaned)


def _extract_recipe_chunk(transcript_chunk: str) -> dict:
    system_prompt = f"""You extract recipe data from cooking transcripts.
Return ONLY valid JSON matching:
{
  "name": "",
  "ingredients": [{"name": "...", "quantity": "..."}],
  "instructions": ["Step 1: ...", "Step 2: ..."],
  "servings": "",
  "prep_time": "",
  "cook_time": "",
  "total_time": "",
  "notes": [],
  "meal_type": "",
  "cuisine": "",
  "nutrition": {
    "calories": "",
    "protein_g": "",
    "carbs_g": "",
    "fat_g": ""
  }
}
Rules:
- Don't hallucinate for the structure. If fields like servings or times are unknown, use "" or [].
- meal_type: exactly one of Breakfast, Lunch, Dinner, Snack (infer from context).
- Infer cuisine from recipe name, ingredients, or speaker context when evident (e.g. Italian, Mexican, Indian); otherwise use "".
- Ingredients must include quantities when stated; else quantity "".
- Instructions must be actionable, chronological, and detailed.
- You MUST provide a best-effort numeric estimate (as strings) for nutrition macros PER SERVING: calories, protein_g, carbs_g, fat_g. Use your nutrition knowledge of typical ingredients/quantities and transcript context to approximate. Only leave a macro field \"\" if there is literally no information about ingredients.
- {_RECIPE_NUTRITION_PROMPT_RULE}
- Output JSON only (no markdown, no commentary)."""

    user_prompt = f"Transcript:\n{transcript_chunk}\n\nReturn the JSON now."

    # Prefer strict JSON if supported
    try:
        completion = client.chat.completions.create(
            model=RECIPE_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1400,
            response_format={"type": "json_object"},
            timeout=EXTRACT_RECIPE_TIMEOUT,
        )
    except Exception:
        completion = client.chat.completions.create(
            model=RECIPE_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1400,
            timeout=EXTRACT_RECIPE_TIMEOUT,
        )

    return _force_json(completion.choices[0].message.content.strip())


def _merge_recipe_parts(parts: list[dict]) -> dict:
    system_prompt = f"""Merge multiple partial recipe JSONs into ONE final recipe JSON.
Return ONLY valid JSON matching:
{
  "name": "",
  "ingredients": [{"name": "...", "quantity": "..."}],
  "instructions": ["Step 1: ...", "Step 2: ..."],
  "servings": "",
  "prep_time": "",
  "cook_time": "",
  "total_time": "",
  "notes": [],
  "meal_type": "",
  "cuisine": "",
  "nutrition": {
    "calories": "",
    "protein_g": "",
    "carbs_g": "",
    "fat_g": ""
  }
}
Rules:
- Deduplicate ingredients case-insensitively; keep most specific quantity.
- Remove duplicate steps, ensure correct chronological order.
- Ensure steps are detailed and actionable.
- For meal_type: use the first non-empty from parts (one of Breakfast, Lunch, Dinner, Snack); if none, use "Dinner".
- For cuisine: use the first non-empty cuisine from the parts; if none, use "".
- Merge/average any provided nutrition macros (calories, protein_g, carbs_g, fat_g) into a single best-effort estimate PER SERVING. If some parts omit macros, use available information from other parts. Only leave a macro field \"\" if ALL parts lack enough information.
- {_RECIPE_NUTRITION_PROMPT_RULE}
- Output JSON only."""

    user_prompt = json.dumps({"parts": parts}, ensure_ascii=False)

    try:
        completion = client.chat.completions.create(
            model=RECIPE_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1800,
            response_format={"type": "json_object"},
            timeout=EXTRACT_RECIPE_TIMEOUT,
        )
    except Exception:
        completion = client.chat.completions.create(
            model=RECIPE_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1800,
            timeout=EXTRACT_RECIPE_TIMEOUT,
        )

    return _force_json(completion.choices[0].message.content.strip())


def _yt_meta(video_url: str) -> dict:
    """Extract metadata without downloading. Uses cookies if available."""
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # Instagram's graphql/query endpoint often 403s even with cookies; default
        # retries (3) make metadata fetch swing from ~1s to ~5s. Cap retries and
        # add a socket timeout so a blocked request fails fast. On failure,
        # _fetch_social_reel_meta falls back to the page caption, and the video
        # download step re-extracts anyway.
        "extractor_retries": int(os.getenv("YTDLP_META_EXTRACTOR_RETRIES", "1")),
        "retries": int(os.getenv("YTDLP_META_RETRIES", "1")),
        "socket_timeout": int(os.getenv("YTDLP_META_SOCKET_TIMEOUT", "8")),
        "extractor_args": {"youtube": _yt_extractor_args()},
    }
    cookie_temp_dir = tempfile.mkdtemp()
    try:
        # Add cookies if available (needed for Instagram/TikTok/Facebook).
        # Copied to a writable path because yt-dlp writes the cookie jar back.
        _cf = _prepare_cookiefile(cookie_temp_dir, video_url)
        if _cf:
            opts["cookiefile"] = _cf
            print("🍪 Using cookies file for metadata")

        # Proxy: YouTube via YT_PROXY, social (FB/IG/TikTok) via SOCIAL_PROXY
        _proxy = _ytdlp_proxy(video_url)
        if _proxy:
            opts["proxy"] = _proxy
            print(f"🌐 Using proxy for metadata: {_proxy}")

        info = _ydl_extract(opts, video_url, download=False)
        return info or {}
    finally:
        shutil.rmtree(cookie_temp_dir, ignore_errors=True)


def _download_audio_mp3(video_url: str):
    """
    Returns: (audio_bytes, meta_dict)
    """
    temp_dir = tempfile.mkdtemp()
    try:
        info = _yt_meta(video_url)
        duration = info.get("duration")
        title = info.get("title") or ""
        extractor = info.get("extractor_key") or info.get("extractor") or ""
        webpage_url = info.get("webpage_url") or video_url
        thumbnail = info.get("thumbnail")  # often present for YouTube/IG/TikTok

        if duration and duration > MAX_VIDEO_SECONDS:
            raise ValueError(f"Video too long ({duration}s). Max allowed is {MAX_VIDEO_SECONDS}s")

        ydl_opts = {
            # IMPORTANT: robust selector with fallbacks (handles many edge cases)
            "format": "bestaudio/best/best",
            "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
            "restrictfilenames": True,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,

            # Helps for YouTube signature issues & format availability
            "extractor_args": {
                "youtube": _yt_extractor_args()
            },

            # Convert to mp3
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
        }

        # Cookies (very helpful for TikTok/IG/Facebook + some YouTube).
        # Copied to a writable path because yt-dlp writes the cookie jar back
        # (Render secret files are read-only → would raise Errno 30).
        cookies_used = False
        _cf = _prepare_cookiefile(temp_dir, video_url)
        if _cf:
            ydl_opts["cookiefile"] = _cf
            cookies_used = True
            print("🍪 Using cookies file")

        # For Instagram, add additional extractor args if cookies are available
        if "instagram.com" in video_url.lower() and cookies_used:
            ydl_opts.setdefault("extractor_args", {})["instagram"] = {
                "webpage_display": ["Desktop"]
            }
        
        # Proxy: YouTube via YT_PROXY, social (FB/IG/TikTok) via SOCIAL_PROXY
        _proxy = _ytdlp_proxy(video_url)
        if _proxy:
            ydl_opts["proxy"] = _proxy
            print(f"🌐 Using proxy: {_proxy}")

        _ydl_extract(ydl_opts, video_url, download=True)

        mp3s = [f for f in os.listdir(temp_dir) if f.endswith(".mp3")]
        if not mp3s:
            raise RuntimeError("No .mp3 produced. Check ffmpeg installation and yt-dlp extraction.")

        mp3_path = os.path.join(temp_dir, mp3s[0])
        with open(mp3_path, "rb") as f:
            audio_bytes = f.read()

        if not audio_bytes:
            raise RuntimeError("Extracted audio is empty")

        meta = {
            "duration": duration,
            "title": title,
            "description": info.get("description") or "",
            "provider": (urlparse(video_url).hostname or ""),
            "extractor": extractor,
            "webpage_url": webpage_url,
            "thumbnail": thumbnail,
        }
        return audio_bytes, meta

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# When transcript is shorter than this (chars), treat as no speech and use frame+vision fallback
MIN_TRANSCRIPT_LENGTH_FOR_VIDEO = int(os.getenv("MIN_TRANSCRIPT_LENGTH_VIDEO", "80"))
# Fewer frames = faster (set NUM_VIDEO_FRAMES_RECIPE=8 for more accuracy)
NUM_VIDEO_FRAMES_FOR_RECIPE = min(int(os.getenv("NUM_VIDEO_FRAMES_RECIPE", "3")), MAX_EXTRACT_RECIPE_IMAGES)
# Timeout for OpenAI recipe/vision calls (seconds)
RECIPE_LLM_TIMEOUT = int(os.getenv("RECIPE_LLM_TIMEOUT", "120"))


def _is_social_video_url(url: str) -> bool:
    lowered = (url or "").lower()
    return "tiktok.com" in lowered or "instagram.com" in lowered


def _is_tiktok_video_url(url: str) -> bool:
    lowered = (url or "").lower()
    return any(h in lowered for h in ("tiktok.com", "vt.tiktok.com", "vm.tiktok.com"))


_LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
    "pt": "Portuguese", "id": "Indonesian", "ms": "Malay", "tl": "Filipino", "fil": "Filipino",
    "vi": "Vietnamese", "th": "Thai", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
    "ar": "Arabic", "hi": "Hindi", "ru": "Russian", "tr": "Turkish", "nl": "Dutch",
    "pl": "Polish", "sv": "Swedish", "no": "Norwegian", "da": "Danish", "fi": "Finnish",
}


def _normalize_language_code(code: str | None) -> str | None:
    if not code:
        return None
    raw = str(code).strip().lower().replace("_", "-")
    if not raw:
        return None
    primary = raw.split("-")[0]
    if len(primary) == 2 and primary.isalpha():
        return primary
    return None


def _detect_text_language(text: str) -> str:
    """Heuristic ISO-639-1 code from post text when platform metadata has no language."""
    text = (text or "").strip()
    if not text:
        return "en"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    if re.search(r"[\uac00-\ud7af]", text):
        return "ko"
    if re.search(r"[\u0600-\u06ff]", text):
        return "ar"
    if re.search(r"[\u0900-\u097f]", text):
        return "hi"
    if re.search(r"[\u0400-\u04ff]", text):
        return "ru"
    if re.search(r"[\u0e00-\u0e7f]", text):
        return "th"
    lower = f" {text.lower()} "
    # French before Spanish — many function words (de, la, que) and accents overlap.
    french_words = (
        " à ", " au ", " aux ", " des ", " les ", " une ", " recette", " ingrédient", " crème",
        " pâte", " cuillère", " pour ", " avec ", " étape", " franc", " traditionnel", " abonner",
        " cuisson", " farine", " beurre", " lard ", " vraie ", " cuisine", " lorrain", " tourte",
        " pense ", " région", " vrai ",
    )
    french_score = sum(w in lower for w in french_words)
    if french_score >= 2 or (
        french_score >= 1 and re.search(r"[àâçéèêëîïôùûœ]", lower)
    ):
        return "fr"
    if re.search(r"[ñ¿¡]", text) or re.search(r"ción|ando\b|ación", lower):
        return "es"
    spanish_words = (
        " con ", " para ", " los ", " las ", " del ", " una ", " receta", " cucharada",
        " paso", " pasos", " harina", " fácil", " déjenme", " enseñar", " el ", " sin ", " más ",
        " agua ", " cocinar", " cebolla", " también", " como ",
    )
    spanish_score = sum(w in lower for w in spanish_words)
    if spanish_score >= 2:
        return "es"
    if spanish_score >= 1 and re.search(r"ñ", lower):
        return "es"
    if sum(w in lower for w in (" dan ", " dengan ", " resep", " langkah", " bumbu ")) >= 2:
        return "id"
    if sum(w in lower for w in (" dan ", " resipi", " langkah", " sudu ")) >= 2:
        return "ms"
    if sum(w in lower for w in (" und ", " mit ", " rezept", " esslöffel", " zutaten ")) >= 2:
        return "de"
    if sum(w in lower for w in (" com ", " receita", " colher", " passo ")) >= 2:
        return "pt"
    if sum(w in lower for w in (" và ", " với ", " công thức", " muốn ")) >= 2:
        return "vi"
    return "en"


def _llm_language_detect_enabled() -> bool:
    return (os.getenv("EXTRACT_RECIPE_LLM_LANGUAGE_DETECT") or "1").strip().lower() not in (
        "0", "false", "no",
    )


def _source_language_prompt_rule() -> str:
    """Universal extraction rule — works for any language without pre-detection."""
    return (
        ' Include "language": ISO-639-1 code of the source content (e.g. "en", "es", "fr", "de", "id", "ja"). '
        "Write ALL recipe fields (name, ingredients, instructions, notes, meal_type, cuisine, "
        "servings, prep_time, cook_time, total_time) in the SAME language as the source — "
        "never translate to English unless the source is English."
    )


def _metadata_language_rule(language: str | None) -> str:
    if not language or language == "en":
        return " Use the same language as the recipe name and instructions."
    label = _LANGUAGE_NAMES.get(language, language)
    return f" Write all time strings in {label}."


def _detect_language_llm(text: str) -> str | None:
    """LLM ISO-639-1 detection — language-agnostic, cached per text snippet."""
    text = (text or "").strip()
    if not text or not _llm_language_detect_enabled():
        return None
    sample = text[:1200]
    key = hashlib.sha256(sample.encode()).hexdigest()
    with _recipe_cache_lock:
        cached = _lang_detect_cache.get(key)
    if cached:
        return cached
    try:
        completion = client.chat.completions.create(
            model=os.getenv("RECIPE_LANGUAGE_MODEL") or RECIPE_LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Detect the primary human language of the text. "
                        'Return ONLY JSON: {"language":"xx"} where xx is ISO-639-1 '
                        "(en, es, fr, de, it, pt, id, ms, vi, th, ja, ko, zh, ar, hi, ru, tr, nl, pl, sv, da, fi, no)."
                    ),
                },
                {"role": "user", "content": sample},
            ],
            temperature=0,
            max_tokens=24,
            response_format={"type": "json_object"},
            timeout=min(12, EXTRACT_RECIPE_TIMEOUT),
        )
        data = json.loads(completion.choices[0].message.content.strip())
        code = _normalize_language_code((data or {}).get("language"))
        if code:
            with _recipe_cache_lock:
                _lang_detect_cache[key] = code
            return code
    except Exception as e:
        print(f"⚠️ LLM language detection failed: {e}")
    return None


def _platform_language_code(meta: dict) -> str | None:
    for key in ("textLanguage", "descLanguage", "language", "lang"):
        code = _normalize_language_code(meta.get(key))
        if code:
            return code
    return None


def _resolve_response_language(meta: dict, recipe: dict | None = None, *, use_llm: bool | None = None) -> str:
    """
    Language-agnostic resolution priority:
    1) language field from extraction LLM
    2) platform textLanguage (TikTok page JSON)
    3) LLM detect on recipe/caption (optional, off during meta fetch)
    4) heuristic fallback
    """
    if use_llm is None:
        use_llm = _llm_language_detect_enabled()

    if recipe:
        code = _normalize_language_code(recipe.get("language"))
        if code:
            return code
        parts = [recipe.get("name") or ""]
        instructions = recipe.get("instructions") or []
        if instructions:
            parts.append(str(instructions[0]))
        sample = " ".join(p for p in parts if p).strip()
        if sample:
            if use_llm:
                detected = _detect_language_llm(sample)
                if detected:
                    return detected
            heuristic = _detect_text_language(sample)
            if heuristic != "en":
                return heuristic

    platform = _platform_language_code(meta)
    if platform and platform != "en":
        return platform

    caption = _video_caption_text(meta)
    if caption:
        if use_llm:
            detected = _detect_language_llm(caption)
            if detected:
                return detected
        heuristic = _detect_text_language(caption)
        if heuristic != "en":
            return heuristic

    return platform or "en"


def _pop_recipe_language(recipe: dict | None) -> str | None:
    """Remove internal language field from recipe dict; return normalized ISO code."""
    if not recipe:
        return None
    raw = recipe.pop("language", None)
    return _normalize_language_code(raw)


def _finalize_recipe_for_response(recipe: dict, meta: dict) -> str:
    """Resolve language, enrich recipe, return ISO code for API response."""
    lang = _pop_recipe_language(recipe)
    if not lang:
        lang = _resolve_response_language(meta, recipe, use_llm=False)
    _enrich_recipe_response(recipe, language=lang)
    return lang


def _is_unrelated_tiktok_description(description: str, title: str) -> bool:
    """True when yt-dlp/page JSON injected promo text unrelated to the video title."""
    if not description or not title:
        return False
    desc_lang = _detect_text_language(description)
    title_lang = _detect_text_language(title)
    if desc_lang != title_lang and title_lang != "en":
        return True
    promo_markers = (
        "contest", "giveaway", "voucher", "fairprice", "heritagefest", "heritage fest",
        "ntuc", "follow us", "tag your friend", "win a", "win $", "terms and conditions",
        "sponsored", "#ad", "paid partnership", "local food", "heritagefest",
    )
    lower = description.lower()
    if any(m in lower for m in promo_markers) and not any(m in title.lower() for m in promo_markers):
        return True
    title_words = [
        w for w in re.findall(r"\w{4,}", title.lower())
        if w not in ("para", "con", "de", "que", "los", "las", "del", "una", "con", "pour", "avec", "les")
    ]
    if title_words and not any(w in lower for w in title_words[:4]):
        if desc_lang != title_lang:
            return True
    return False


def _tiktok_post_language(meta: dict) -> str:
    """Backward-compatible alias — delegates to language-agnostic resolver."""
    return _resolve_response_language(meta)


def _apply_response_language(payload: dict, video_url: str, meta: dict) -> dict:
    """Attach language to social-video extract responses."""
    if not (_is_social_video_url(video_url) or _is_tiktok_video_url(video_url)):
        return payload
    lang = payload.get("language")
    if not lang:
        recipe = payload.get("recipe") or {}
        lang = _resolve_response_language(meta, recipe, use_llm=False)
    payload["language"] = lang
    if isinstance(payload.get("meta"), dict):
        payload["meta"] = {**payload["meta"], "language": lang}
    return payload


def _return_video_extract_result(
    result: dict, url_key: str, video_url: str, meta: dict, *, language: str | None = None
) -> tuple:
    if language:
        result["language"] = language
    result = _apply_response_language(result, video_url, meta)
    # Re-host the (ephemeral) thumbnail in MealMap Storage and rewrite the URL to
    # the stable Storage URL *before* caching, so both the response and the cache
    # carry the permanent URL. If the upload was started early (overlapping the
    # LLM), resolve that in-flight Future instead of uploading synchronously.
    _img_future = _img_persist_futures.pop(url_key, None)
    _persist_recipe_source_image(result.get("source"), result.get("meta"), future=_img_future)
    with _recipe_cache_lock:
        _recipe_cache[url_key] = result
    return jsonify(_apply_extract_recipe_image_pref(result)), 200


def _social_video_frames_for_duration(
    url: str, duration_sec: float | None, *, meta: dict | None = None
) -> int:
    """TikTok/IG: sample multiple frames when vision fallback is needed."""
    if not _is_social_video_url(url):
        return _video_frames_for_duration(duration_sec)
    if meta and meta.get("vision_distributed_sections"):
        n = len(meta["vision_distributed_sections"])
        return min(max(1, n), MAX_EXTRACT_RECIPE_IMAGES)
    try:
        n = int(os.getenv("EXTRACT_RECIPE_SOCIAL_VIDEO_FRAMES", "2"))
    except ValueError:
        n = 2
    return min(max(1, n), MAX_EXTRACT_RECIPE_IMAGES)


def _video_frames_for_duration(duration_sec: float | None) -> int:
    """Use fewer frames on short reels — 1 middle frame is enough for on-screen recipe text."""
    cap = NUM_VIDEO_FRAMES_FOR_RECIPE
    if not duration_sec or duration_sec <= 0:
        return min(1, cap)
    if duration_sec <= 20:
        return 1
    if duration_sec <= 45:
        return min(2, cap)
    return cap


def _normalize_extracted_recipe(recipe: dict) -> dict:
    ingredients = recipe.get("ingredients") or []
    instructions = recipe.get("instructions") or []
    if ingredients and isinstance(ingredients[0], str):
        ingredients = [{"name": ing, "quantity": ""} for ing in ingredients]
    if not isinstance(instructions, list):
        instructions = [str(instructions)]
    recipe["ingredients"] = ingredients
    recipe["instructions"] = instructions
    return recipe


def _recipe_has_instructions(recipe: dict | None) -> bool:
    if not recipe:
        return False
    return any(str(step).strip() for step in (recipe.get("instructions") or []))


def _recipe_has_usable_content(recipe: dict | None) -> bool:
    if not recipe:
        return False
    instructions = recipe.get("instructions") or []
    if any(str(step).strip() for step in instructions):
        return True
    ingredients = recipe.get("ingredients") or []
    for ing in ingredients:
        if isinstance(ing, dict):
            if (ing.get("name") or "").strip():
                return True
        elif str(ing).strip():
            return True
    return bool((recipe.get("name") or "").strip())


def _social_recipe_is_complete(recipe: dict | None) -> bool:
    """True when a social-video recipe has enough ingredients and steps to be trustworthy."""
    if not recipe:
        return False
    ing_count = sum(
        1 for ing in (recipe.get("ingredients") or [])
        if isinstance(ing, dict) and (ing.get("name") or "").strip()
    )
    inst_count = sum(
        1 for step in (recipe.get("instructions") or [])
        if str(step).strip()
    )
    return ing_count >= 3 and inst_count >= 2


def _recipe_missing_sections(recipe: dict | None) -> list[str]:
    """Sections entirely absent from a recipe (for partial API responses)."""
    if not recipe:
        return ["ingredients", "instructions"]
    ingredients = [
        ing for ing in (recipe.get("ingredients") or [])
        if isinstance(ing, dict) and (ing.get("name") or "").strip()
    ]
    instructions = [str(s).strip() for s in (recipe.get("instructions") or []) if str(s).strip()]
    missing: list[str] = []
    if not ingredients:
        missing.append("ingredients")
    if not instructions:
        missing.append("instructions")
    return missing


_CAPTION_QTY_PATTERN = re.compile(
    r'\d+(?:\.\d+)?\s*(?:'
    r'tbsp|tsp|tablespoons?|teaspoons?|cups?|cloves?|'
    r'g\b|gram|grams|kg|ml\b|l\b|litre|liter|oz\b|lb|pcs|pieces?|'
    r'cuillères?|c\.?\s*à\s*s\.?|c\.?\s*à\s*c\.?|'
    r'cucharadas?|cucharaditas?|tazas?'
    r')',
    re.IGNORECASE,
)


def _caption_has_numbered_steps(caption: str) -> bool:
    """Numbered steps in any language (supports accented letters)."""
    return bool(re.search(r'(?:^|\n)\s*\d+[\.\):\-]\s+[^\d\s]', caption, re.MULTILINE))


def _caption_has_instructions(caption: str) -> bool:
    """True when the post caption text itself contains cooking steps (not just ingredients).

    Detects steps written as numbered lists, under a "how to make it"/"directions"
    style header, OR as free-flowing action-verb sentences — including run-on
    paragraphs with no line breaks between steps (common in TikTok/IG captions).
    This raises recall of genuine steps; it does NOT accept invented instructions,
    since the caption text itself must contain the step-like content.
    """
    caption = (caption or "").strip()
    if not caption:
        return False
    if _caption_has_numbered_steps(caption):
        return True
    if re.search(r'\b(?:step|étape|paso|langkah)\s*\d+', caption, re.IGNORECASE):
        return True
    # Instruction-section headers creators commonly use (incl. "how to make it").
    if re.search(
        r'\b(?:instructions?|directions?|method|steps?|'
        r'how\s+to\s+make(?:\s+it)?|to\s+make(?:\s+it)?|here\'?s\s+how|let\'?s\s+make|'
        r'préparation|preparation|étapes?|pasos?|c[oó]mo\s+hacer|modo\s+de\s+preparo)\b\s*:?',
        caption,
        re.IGNORECASE,
    ):
        return True
    action_starts = (
        "mix ", "add ", "fry ", "bake ", "cook ", "heat ", "serve ", "combine ",
        "stir ", "boil ", "simmer ", "coat ", "season ", "place ", "remove ",
        "drain ", "slice ", "chop ", "melt ", "whisk ", "pour ", "spread ",
        "reduce ", "toss ", "marinate ", "preheat ", "transfer ", "top ",
        "pan-fry ", "deep fry ", "air fry ",
        "mélange", "ajoute", "cuire", "faire ", "verser ", "étaler ", "préchauff",
        "mezcl", "añad", "cocin", "calient", "hornea", "agrega",
    )
    # Split on line breaks AND sentence punctuation so steps written as sentences
    # (not just one-per-line) are counted.
    fragments = re.split(r'(?:\r?\n|(?<=[.!?])\s+)', caption)
    action_lines = 0
    for frag in fragments:
        frag = frag.strip().lstrip("•-*▪→ ")
        if len(frag) < 12:
            continue
        lower = frag.lower()
        if re.match(r"^\d+[\.\)]\s", frag):
            action_lines += 1
            continue
        if any(lower.startswith(v) for v in action_starts):
            action_lines += 1
    if action_lines >= 2:
        return True
    # Run-on paragraph with no separators: count capitalized cooking verbs that
    # start new steps mid-text (e.g. "...dish Add ... Stir ... Bake ..."). A high
    # count is a strong signal the caption contains a real method.
    cap_verbs = re.findall(
        r'\b(?:Mix|Add|Fry|Bake|Cook|Heat|Serve|Combine|Stir|Boil|Simmer|Coat|'
        r'Season|Place|Remove|Drain|Slice|Chop|Melt|Whisk|Pour|Spread|Reduce|'
        r'Toss|Marinate|Preheat|Transfer|Top|Blend|Fold|Grill|Roast|Layer|Cover|'
        r'Bring|Rest|Garnish|Sprinkle|Cut|Dice|Fill|Repeat|Flip|Knead|Divide)\b',
        caption,
    )
    return len(cap_verbs) >= 3


def _social_caption_recipe_acceptable(recipe: dict | None, caption: str) -> bool:
    """Caption-first: require steps present in source caption — never accept invented instructions."""
    if not _recipe_has_usable_content(recipe):
        return False
    caption = (caption or "").strip()
    if not _caption_has_instructions(caption):
        inst_count = sum(
            1 for step in (recipe.get("instructions") or [])
            if str(step).strip()
        )
        if inst_count > 0:
            print("⚠️ Rejecting caption result: instructions not in source caption (would be invented)")
        else:
            print("⚠️ Caption has ingredients only; will use vision for instructions")
        return False
    ing_count = sum(
        1 for ing in (recipe.get("ingredients") or [])
        if isinstance(ing, dict) and (ing.get("name") or "").strip()
    )
    inst_count = sum(
        1 for step in (recipe.get("instructions") or [])
        if str(step).strip()
    )
    cap_len = len(caption)
    min_ing = 2 if cap_len >= 400 else 3
    min_inst = 1 if cap_len >= 400 else 2
    if ing_count >= min_ing and inst_count >= min_inst:
        return True
    print(
        f"⚠️ Caption extraction incomplete ({ing_count} ingredient(s), "
        f"{inst_count} step(s)); will try vision"
    )
    return False


def _sanitize_partial_caption_recipe(recipe: dict | None, caption: str) -> tuple[dict | None, list[str]]:
    """Caption-only extraction: keep the parts the caption actually contains and
    report what's missing, instead of falling back to vision.

    - Ingredients are trusted as extracted (creators list them literally).
    - Instructions are kept only if the source caption actually contains steps
      (prevents the LLM from inventing a method).

    Returns (recipe, missing) where missing ⊆ ['ingredients', 'instructions'],
    or (None, missing) when the caption yielded neither (caller falls to vision).
    """
    if not recipe:
        return None, ["ingredients", "instructions"]
    caption = (caption or "").strip()
    ingredients = [
        ing for ing in (recipe.get("ingredients") or [])
        if isinstance(ing, dict) and (ing.get("name") or "").strip()
    ]
    instructions = [str(s).strip() for s in (recipe.get("instructions") or []) if str(s).strip()]
    # Anti-hallucination: only keep instructions the caption itself contains.
    if instructions and not _caption_has_instructions(caption):
        instructions = []
    recipe["ingredients"] = ingredients
    recipe["instructions"] = instructions
    missing = []
    if not ingredients:
        missing.append("ingredients")
    if not instructions:
        missing.append("instructions")
    if not ingredients and not instructions:
        return None, missing
    return recipe, missing


def _caption_has_full_recipe_text(caption: str) -> bool:
    """True when post metadata likely contains a full recipe (TikTok 'more' text, IG caption)."""
    caption = (caption or "").strip()
    if len(caption) < 120:
        return False
    if _caption_has_numbered_steps(caption):
        return True
    if len(caption) >= 280 and _caption_looks_like_recipe(caption):
        return True
    if re.search(r'\b(?:step|étape|paso)\s*\d+', caption, re.IGNORECASE):
        return True
    qty_hits = len(_CAPTION_QTY_PATTERN.findall(caption))
    if qty_hits >= 3:
        return True
    if len(caption) >= 200 and qty_hits >= 2:
        return True
    return False


def _should_try_social_caption_first(meta: dict, url: str) -> bool:
    """Try caption LLM only when post text includes real steps — not ingredients-only captions."""
    if (os.getenv("EXTRACT_RECIPE_SOCIAL_CAPTION_FIRST") or "1").strip().lower() in ("0", "false", "no"):
        return False
    if not _is_social_video_url(url):
        return False
    caption = _video_caption_text(meta)
    if len(caption) < 150:
        return False
    if not (_caption_looks_like_recipe(caption) or _caption_has_full_recipe_text(caption)):
        return False
    # Try the caption LLM whenever the post looks like a recipe — including
    # ingredients-only or instructions-only captions. Partial results are
    # returned with the missing section flagged (see _sanitize_partial_caption_recipe)
    # instead of falling back to slow vision.
    return True


def _video_caption_text(meta: dict) -> str:
    """Best available post caption/description from yt-dlp metadata."""
    description = (meta.get("description") or "").strip()
    title = (meta.get("title") or "").strip()
    if title and description and _is_unrelated_tiktok_description(description, title):
        return title
    if description and len(description) >= len(title):
        return description
    if title and description and description not in title and title not in description:
        return f"{title}\n\n{description}"
    return description or title


def _caption_looks_like_recipe(text: str) -> bool:
    if len(text) < 50:
        return False
    lowered = text.lower()
    signals = (
        "ingredient", "ingrédient", "ingrediente", "bahan", "zutaten",
        "cup", "cups", "tbsp", "tsp", "tablespoon", "teaspoon", "cuillère", "cucharada",
        "step", "steps", "étape", "étapes", "paso", "pasos", "langkah",
        "layer", "mix", "mélange", "bake", "cook", "cuire", "cocinar", "fry", "simmer",
        "grams", "gram", "ounce", "oz ", "season", "sauce", "garlic", "chicken",
        "recette", "receta", "resep", "rezept", "œuf", "oeuf", "farine", "beurre", "crème",
    )
    if sum(1 for signal in signals if signal in lowered) >= 2:
        return True
    if len(_CAPTION_QTY_PATTERN.findall(text)) >= 2 and len(text) >= 100:
        return True
    return False


def _extract_recipe_from_social_caption_llm(caption: str, *, language: str | None = None) -> dict:
    """Extract a full recipe from TikTok/IG caption text (ingredients + numbered steps)."""
    lang_rule = _source_language_prompt_rule()
    system_prompt = f"""You extract a complete recipe from a TikTok or Instagram post caption.
Captions often contain:
- A recipe name
- Ingredient lists with quantities (tbsp, tsp, cups, cloves, etc.)
- Numbered steps or short instruction lines

Return ONLY valid JSON matching:
{{
  "language": "",
  "name": "",
  "ingredients": [{{"name": "...", "quantity": "..."}}],
  "instructions": ["Step 1: ...", "Step 2: ..."],
  "servings": "",
  "prep_time": "",
  "cook_time": "",
  "total_time": "",
  "notes": [],
  "meal_type": "",
  "cuisine": "",
  "nutrition": {{
    "calories": "",
    "protein_g": "",
    "carbs_g": "",
    "fat_g": ""
  }}
}}
Rules:
- Extract EVERY ingredient with its quantity when stated.
- Extract EVERY cooking step that appears in the caption, in order.
- If the caption has NO cooking steps, return "instructions": [] — do NOT invent or infer steps.
- Do not skip, merge, or omit steps that appear in the caption.
- Ignore hashtags, @mentions, and engagement text ("follow for more", "link in bio").
- meal_type: exactly one of Breakfast, Lunch, Dinner, Snack.
- nutrition: best-effort per-serving estimates as strings from ingredients.{lang_rule}
- Output JSON only (no markdown)."""

    user_prompt = f"Post caption:\n{caption}\n\nReturn the complete recipe JSON now."

    try:
        completion = client.chat.completions.create(
            model=RECIPE_TEXT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1400,
            response_format={"type": "json_object"},
            timeout=EXTRACT_RECIPE_TIMEOUT,
        )
    except Exception:
        completion = client.chat.completions.create(
            model=RECIPE_TEXT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1400,
            timeout=EXTRACT_RECIPE_TIMEOUT,
        )
    return _force_json(completion.choices[0].message.content.strip())


def _extract_recipe_from_video_caption(meta: dict, *, language: str | None = None) -> dict | None:
    """Parse recipe from social post caption when vision/transcript miss on-screen text."""
    caption = _video_caption_text(meta)
    if not _caption_looks_like_recipe(caption):
        return None
    try:
        recipe = _extract_recipe_from_social_caption_llm(caption)
        return _normalize_extracted_recipe(recipe)
    except Exception as e:
        print(f"⚠️ Caption recipe extraction failed: {e}")
        return None


def _build_video_recipe_source(video_url: str, meta: dict, recipe: dict) -> dict:
    return {
        "type": "video",
        "url": video_url,
        "provider": meta.get("provider", ""),
        "title": meta.get("title", "") or recipe.get("name", ""),
        "image": meta.get("thumbnail"),
        "source_type": determine_source_type(video_url),
    }


def _video_meta_from_yt_info(info: dict, video_url: str) -> dict:
    description = _combine_yt_caption_fields(info, video_url)
    title = (info.get("title") or "").strip()
    if _is_tiktok_video_url(video_url) and title and description and _is_unrelated_tiktok_description(description, title):
        description = title
    return {
        "duration": info.get("duration"),
        "title": title,
        "description": description,
        "provider": (urlparse(video_url).hostname or ""),
        "extractor": info.get("extractor_key") or info.get("extractor") or "",
        "webpage_url": info.get("webpage_url") or video_url,
        "thumbnail": info.get("thumbnail"),
    }


def _combine_yt_caption_fields(info: dict, video_url: str = "") -> str:
    """Merge all caption-like text yt-dlp may return (TikTok/IG descriptions vary by extractor)."""
    candidates: list[str] = []
    for key in ("description", "title", "fulltitle", "alt_title"):
        val = (info.get(key) or "").strip()
        if val and val not in candidates:
            candidates.append(val)
    if not candidates:
        return ""
    title = (info.get("title") or "").strip()
    description = (info.get("description") or "").strip()
    url = video_url or info.get("webpage_url") or ""
    if _is_tiktok_video_url(url) and title and description and _is_unrelated_tiktok_description(description, title):
        return title
    primary = max(candidates, key=len)
    extras = [c for c in candidates if c != primary and c not in primary]
    if extras:
        return primary + "\n\n" + "\n\n".join(extras)
    return primary


def _youtube_captions_enabled() -> bool:
    return (os.getenv("EXTRACT_RECIPE_YOUTUBE_CAPTIONS_FIRST") or "1").strip().lower() not in (
        "0", "false", "no",
    )


def _vtt_to_plain_text(vtt: str) -> str:
    """Convert YouTube WebVTT captions to plain transcript text."""
    lines: list[str] = []
    for raw in vtt.splitlines():
        line = raw.strip()
        if not line or line == "WEBVTT" or "-->" in line:
            continue
        if line.startswith(("Kind:", "Language:", "NOTE")):
            continue
        if re.match(r"^\d+$", line):
            continue
        if re.search(r"<\d{2}:\d{2}", line):
            continue
        line = re.sub(r"<[^>]+>", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if not line or re.fullmatch(r"\[[^\]]+\]", line):
            continue
        if lines and lines[-1] == line:
            continue
        if lines and line.startswith(lines[-1]) and len(line) > len(lines[-1]):
            lines[-1] = line
            continue
        if lines and lines[-1].startswith(line) and len(lines[-1]) > len(line):
            continue
        lines.append(line)
    return " ".join(lines)


def _fetch_youtube_caption_transcript(video_url: str) -> str | None:
    """Download YouTube manual/auto captions via yt-dlp (~2s vs minutes of audio+Whisper)."""
    if not _youtube_captions_enabled():
        return None
    temp_dir = tempfile.mkdtemp()
    try:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en", "en-US", "en-GB"],
            "subtitlesformat": "vtt/best",
            "outtmpl": os.path.join(temp_dir, "%(id)s"),
            "noplaylist": True,
            "extractor_args": {"youtube": _yt_extractor_args()},
        }
        _proxy = _ytdlp_proxy(video_url)
        if _proxy:
            ydl_opts["proxy"] = _proxy
        _ydl_extract(ydl_opts, video_url, download=True)
        vtt_files = sorted(f for f in os.listdir(temp_dir) if f.endswith(".vtt"))
        if not vtt_files:
            return None
        with open(os.path.join(temp_dir, vtt_files[0]), encoding="utf-8", errors="ignore") as f:
            text = _vtt_to_plain_text(f.read())
        return text if len(text) >= MIN_TRANSCRIPT_LENGTH_FOR_VIDEO else None
    except Exception as e:
        print(f"⚠️ YouTube caption fetch failed: {e}")
        return None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _youtube_whisper_max_seconds() -> int:
    try:
        return max(30, int(os.getenv("EXTRACT_RECIPE_YOUTUBE_WHISPER_MAX_SECONDS", "120")))
    except ValueError:
        return 120


def _youtube_vision_max_seconds() -> int:
    try:
        return max(30, int(os.getenv("EXTRACT_RECIPE_YOUTUBE_VISION_MAX_SECONDS", "60")))
    except ValueError:
        return 60


def _trim_youtube_transcript_for_llm(text: str) -> str:
    """Bound very long caption transcripts so the recipe LLM stays fast."""
    try:
        cap = int(os.getenv("EXTRACT_RECIPE_YOUTUBE_TRANSCRIPT_MAX_CHARS", "12000"))
    except ValueError:
        cap = 12000
    text = (text or "").strip()
    if len(text) <= cap:
        return text
    return text[:cap].rstrip() + "\n[...truncated]"


def _transcribe_youtube_audio(video_url: str, meta: dict) -> str:
    """Whisper fallback for YouTube — only first N seconds on long videos."""
    duration = float(meta.get("duration") or 0)
    max_sec = _youtube_whisper_max_seconds()
    if duration > max_sec:
        print(f"⚡ YouTube bounded Whisper: first {max_sec}s of {duration:.0f}s")
        temp_dir, video_path, _ = _download_video_to_file(
            video_url, fast=True, max_seconds=max_sec
        )
        try:
            audio_bytes = _extract_audio_from_video_file(video_path)
            if audio_bytes:
                return _transcribe_audio_bytes(audio_bytes)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return ""
    audio_bytes, _ = download_audio_mp3(video_url)
    return _transcribe_audio_bytes(audio_bytes)


def _fetch_youtube_transcript(video_url: str, *, meta: dict | None = None) -> tuple[str, dict]:
    """Fast YouTube path: captions first, bounded Whisper fallback. No video download."""
    if meta is None:
        info = _yt_meta(video_url)
        meta = _video_meta_from_yt_info(info, video_url)
    _assert_youtube_shorts_duration(meta.get("duration"))

    t0 = time.time()
    caption_text = _fetch_youtube_caption_transcript(video_url)
    if caption_text and not _is_likely_non_speech(caption_text):
        meta["transcript_source"] = "youtube_captions"
        print(f"📝 YouTube captions fetched in {time.time() - t0:.2f}s ({len(caption_text)} chars)")
        return caption_text, meta

    print("🎤 YouTube captions unavailable; falling back to bounded audio+Whisper...")
    t1 = time.time()
    transcript_text = _transcribe_youtube_audio(video_url, meta)
    meta["transcript_source"] = "whisper"
    print(f"🎤 YouTube Whisper transcript in {time.time() - t1:.2f}s ({len(transcript_text)} chars)")
    return transcript_text, meta


def _social_vision_max_download_seconds(meta: dict, url: str) -> int | None:
    """Cap IG/TikTok video bytes downloaded for frame extraction."""
    try:
        cap = int(os.getenv("EXTRACT_RECIPE_SOCIAL_VISION_MAX_SECONDS", "20"))
    except ValueError:
        cap = 20
    if cap <= 0:
        return None
    lowered = (url or "").lower()
    if "tiktok.com" not in lowered and "instagram.com" not in lowered:
        return None
    duration = float(meta.get("duration") or 0)
    if duration > cap:
        return cap
    return None


def _download_sections_total_seconds(specs: list[str]) -> float:
    total = 0.0
    for spec in specs:
        match = re.match(r"\*(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)", (spec or "").strip())
        if match:
            total += max(0.0, float(match.group(2)) - float(match.group(1)))
    return total


def _social_vision_sample_section_specs(duration: float) -> list[str]:
    """Sparse yt-dlp sections: legacy opening clip + optional late clips for long reels."""
    duration = max(1.0, float(duration or 60))
    try:
        cap = int(os.getenv("EXTRACT_RECIPE_SOCIAL_VISION_MAX_SECONDS", "20"))
    except ValueError:
        cap = 20
    try:
        section_len = max(2.0, min(float(os.getenv("EXTRACT_RECIPE_SOCIAL_VISION_SECTION_SECONDS", "4")), 8.0))
    except ValueError:
        section_len = 4.0

    specs: list[str] = []
    seen: set[str] = set()

    def _add(spec: str) -> None:
        if spec and spec not in seen:
            seen.add(spec)
            specs.append(spec)

    # Preserve legacy behaviour: always sample the opening of the reel first.
    early_end = int(min(max(1.0, float(cap)), duration))
    _add(f"*0-{early_end}")

    if duration <= cap:
        return specs

    # Long reel: add sparse clips after the intro for recipes that start later.
    try:
        num_late = max(1, min(int(os.getenv("EXTRACT_RECIPE_SOCIAL_VISION_LATE_SECTIONS", "2")), 4))
    except ValueError:
        num_late = 2
    try:
        skip_intro = float(os.getenv("EXTRACT_RECIPE_SOCIAL_VISION_SKIP_INTRO_SECONDS", "0"))
    except ValueError:
        skip_intro = 0.0
    if skip_intro <= 0:
        skip_intro = min(15.0, max(8.0, duration * 0.18))

    start_min = max(float(early_end), skip_intro)
    start_max = max(start_min, duration - section_len)
    if start_max <= start_min:
        return specs

    for i in range(num_late):
        frac = (i + 1) / (num_late + 1)
        start = start_min + frac * (start_max - start_min)
        start = min(start, duration - section_len)
        end = min(start + section_len, duration)
        _add(f"*{int(start)}-{int(max(start + 1, end))}")
    return specs


def _social_vision_download_plan(meta: dict, url: str) -> tuple[list[str] | None, int | None]:
    """Return (distributed_section_specs, legacy_max_seconds) for social vision downloads."""
    lowered = (url or "").lower()
    if "tiktok.com" not in lowered and "instagram.com" not in lowered:
        return None, None
    duration = float(meta.get("duration") or 0)
    cap = _social_vision_max_download_seconds(meta, url)
    distributed = (
        os.getenv("EXTRACT_RECIPE_SOCIAL_VISION_DISTRIBUTED", "1").strip().lower()
        not in ("0", "false", "no")
    )
    if distributed and cap and duration > cap:
        sections = _social_vision_sample_section_specs(duration)
        if len(sections) > 1:
            print(
                f"⚡ Distributed vision download: opening + {len(sections) - 1} late clip(s) "
                f"across {duration:.0f}s reel"
            )
            return sections, None
    return None, cap


def _effective_vision_duration(meta: dict, url: str) -> float:
    duration = float(meta.get("duration") or 0)
    cap = _social_vision_max_download_seconds(meta, url)
    if cap and duration > cap:
        return float(cap)
    return duration


def _prefer_vision_first_for_video_url(url: str) -> bool:
    """IG/TikTok video reels: fetch post caption first; vision is fallback when caption is incomplete."""
    if (os.getenv("EXTRACT_RECIPE_SOCIAL_VISION_FIRST") or "1").strip().lower() in ("0", "false", "no"):
        return False
    if _is_tiktok_photo_url(url):
        return False
    lowered = (url or "").lower()
    if "instagram.com/reel" in lowered or "instagram.com/p/" in lowered:
        return True
    return _is_tiktok_video_reel_url(url)


def _frame_jpeg_max_width() -> int:
    try:
        return max(256, min(int(os.getenv("EXTRACT_RECIPE_FRAME_MAX_WIDTH", "512")), 1024))
    except ValueError:
        return 512


def _extract_middle_frame_data_url(video_path: str, duration_sec: float | None) -> str | None:
    """One small JPEG at mid-reel — fastest path for short social recipe videos."""
    out_dir = tempfile.mkdtemp()
    out_path = os.path.join(out_dir, "frame.jpg")
    ts = max(0.5, (duration_sec or 10.0) / 2.0)
    w = _frame_jpeg_max_width()
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-ss", str(ts), "-i", video_path,
                "-vframes", "1", "-vf", f"scale={w}:-1", "-q:v", "7",
                out_path,
            ],
            capture_output=True,
            timeout=EXTRACT_RECIPE_TIMEOUT,
        )
        if result.returncode != 0 or not os.path.exists(out_path):
            return None
        with open(out_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def _transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """Transcribe mp3/audio bytes for the video recipe pipeline."""
    if not audio_bytes:
        return ""
    model = (os.getenv("VIDEO_TRANSCRIBE_MODEL") or "gpt-4o-mini-transcribe").strip()
    audio_file_obj = io.BytesIO(audio_bytes)
    audio_file_obj.name = "audio.mp3"
    t = client.audio.transcriptions.create(
        model=model,
        file=audio_file_obj,
        response_format="text",
    )
    return t.strip() if isinstance(t, str) else str(t).strip()


def _merge_video_download_meta(video_url: str, video_meta: dict | None, audio_meta: dict | None) -> dict:
    """Combine metadata from parallel audio + video yt-dlp calls."""
    meta = dict(video_meta or {})
    audio_meta = audio_meta or {}
    if not meta.get("duration") and audio_meta.get("duration"):
        meta["duration"] = audio_meta["duration"]
    if not meta.get("title") and audio_meta.get("title"):
        meta["title"] = audio_meta["title"]
    if not meta.get("description") and audio_meta.get("description"):
        meta["description"] = audio_meta["description"]
    old_desc = (meta.get("description") or "").strip()
    new_desc = (audio_meta.get("description") or "").strip()
    if len(new_desc) > len(old_desc):
        meta["description"] = new_desc
    meta.setdefault("provider", urlparse(video_url).hostname or "")
    if not meta.get("extractor") and audio_meta.get("source"):
        meta["extractor"] = audio_meta["source"]
    return meta


def _download_video_to_file(
    video_url: str,
    *,
    fast: bool = False,
    max_seconds: int | None = None,
    download_sections: list[str] | None = None,
):
    """
    Download video (not just audio) to a temp file for frame extraction.
    Uses a single yt-dlp call (extract_info + download=True) to avoid the
    redundant separate metadata round-trip from _yt_meta().
    fast=True prefers ≤480p for quicker social-reel downloads (vision only needs rough frames).
    max_seconds: when set, only download the first N seconds (for long social reels).
    download_sections: yt-dlp section specs (e.g. ["*20-24","*40-44"]) for sparse timeline sampling.
    Returns: (temp_dir, video_path, meta_dict). Caller must shutil.rmtree(temp_dir) when done.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        if fast:
            fmt = "best[height<=480]/best[height<=720]/best"
        else:
            fmt = "best[height<=720]/best"
        ydl_opts = {
            "format": fmt,
            "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
            "restrictfilenames": True,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            # Resilience against truncated/incomplete downloads ("N bytes read,
            # M more expected"): retry the download and its fragments, and pull
            # in bounded HTTP chunks (Range requests) so a mid-stream cut can
            # resume instead of failing the whole request.
            "retries": int(os.getenv("YTDLP_DL_RETRIES", "3")),
            "fragment_retries": int(os.getenv("YTDLP_DL_FRAGMENT_RETRIES", "3")),
            "http_chunk_size": int(os.getenv("YTDLP_DL_HTTP_CHUNK_SIZE", str(10 * 1024 * 1024))),
            "socket_timeout": int(os.getenv("YTDLP_DL_SOCKET_TIMEOUT", "15")),
            "extractor_args": {"youtube": _yt_extractor_args()},
        }
        # Copy cookies to a writable path (yt-dlp writes the jar back; Render
        # secret files are read-only → would raise Errno 30).
        _cf = _prepare_cookiefile(temp_dir, video_url)
        if _cf:
            ydl_opts["cookiefile"] = _cf
        if "instagram.com" in video_url.lower() and ydl_opts.get("cookiefile"):
            ydl_opts.setdefault("extractor_args", {})["instagram"] = {"webpage_display": ["Desktop"]}
        _proxy = _ytdlp_proxy(video_url)
        if _proxy:
            ydl_opts["proxy"] = _proxy
        if download_sections:
            ydl_opts["download_sections"] = download_sections
        elif max_seconds and max_seconds > 0:
            ydl_opts["download_sections"] = [f"*0-{int(max_seconds)}"]

        # Single call: fetches metadata AND downloads in one network session
        info = _ydl_extract(ydl_opts, video_url, download=True)

        duration = info.get("duration")
        title = info.get("title") or ""
        extractor = info.get("extractor_key") or info.get("extractor") or ""
        webpage_url = info.get("webpage_url") or video_url
        thumbnail = info.get("thumbnail")

        if duration and duration > MAX_VIDEO_SECONDS:
            raise ValueError(f"Video too long ({duration}s). Max allowed is {MAX_VIDEO_SECONDS}s")

        candidates = [f for f in os.listdir(temp_dir) if f.endswith((".mp4", ".webm", ".mkv", ".mov"))]
        if not candidates:
            raise RuntimeError("No video file produced by yt-dlp")
        video_path = os.path.join(temp_dir, candidates[0])
        meta = _video_meta_from_yt_info(info, video_url)
        if download_sections:
            meta["vision_distributed_sections"] = download_sections
            meta["vision_duration"] = _download_sections_total_seconds(download_sections)
            if len(download_sections) > 1:
                meta["vision_distributed"] = True
                print(
                    f"⚡ Sparse vision clips ready (~{meta['vision_duration']:.0f}s total from "
                    f"{len(download_sections)} section(s))"
                )
            else:
                print(
                    f"⚡ Partial video download: first {meta['vision_duration']:.0f}s "
                    f"of {duration}s"
                )
        elif max_seconds and duration and duration > max_seconds:
            meta["vision_duration"] = float(max_seconds)
            print(f"⚡ Partial video download: first {max_seconds}s of {duration}s")
        return temp_dir, video_path, meta
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _profile_discovery_opts(profile_url: str, limit: int) -> dict:
    """yt-dlp opts for discovering video URLs from a profile/channel page."""
    opts = {
        "quiet": True,
        "no_warnings": True,
        # Surface discovery failures so API can return actionable errors.
        "ignoreerrors": False,
        "noplaylist": False,
        "extract_flat": "in_playlist",
        "playlistend": max(1, min(limit * 4, 50)),
        "extractor_args": {"youtube": _yt_extractor_args()},
    }
    _cf = _prepare_cookiefile(video_url=profile_url)
    if _cf:
        opts["cookiefile"] = _cf
    _proxy = _ytdlp_proxy(profile_url)
    if _proxy:
        opts["proxy"] = _proxy
    elif not is_youtube_url(profile_url):
        # No social proxy configured: force direct connection for IG/TikTok
        # discovery (avoid inheriting an unrelated HTTP(S)_PROXY from the env).
        opts["proxy"] = ""
    return opts


def _entry_to_video_url(entry: dict) -> str | None:
    """Best-effort conversion from yt-dlp flat entry to a playable URL."""
    if not isinstance(entry, dict):
        return None
    for key in ("webpage_url", "url", "original_url"):
        val = entry.get(key)
        if isinstance(val, str) and val.startswith(("http://", "https://")):
            return val

    extractor = str(entry.get("extractor_key") or entry.get("extractor") or "").lower()
    vid = entry.get("id")
    if isinstance(vid, str) and vid:
        if "youtube" in extractor:
            return f"https://www.youtube.com/watch?v={vid}"
        if "instagram" in extractor:
            return f"https://www.instagram.com/reel/{vid}/"
        if "tiktok" in extractor:
            uploader = entry.get("uploader_id") or entry.get("channel_id")
            if uploader:
                return f"https://www.tiktok.com/@{uploader}/video/{vid}"
    return None


def _probe_format_duration_sec(path: str) -> float | None:
    """Best-effort container duration from ffprobe (seconds)."""
    r = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        text=True,
    )
    s = (r.stdout or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _normalize_for_concat(
    input_path: str,
    output_path: str,
    *,
    max_input_duration_sec: float | None = None,
) -> None:
    """
    Re-encode a video to 1280x720 / 30fps / h264 / aac stereo 44100 Hz.
    If the source has no audio track, a silent audio track is added automatically.
    This ensures both clips are identical in format before FFmpeg concat.

    If max_input_duration_sec is set, only the first N seconds of the input are read.
    """
    # Probe for audio streams — use a robust fallback so a bad ffprobe response
    # never silently replaces real audio with silence.
    probe_result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-print_format", "json",
            input_path,
        ],
        capture_output=True, text=True,
    )
    try:
        probe_data = json.loads(probe_result.stdout or "{}")
        has_audio = bool(probe_data.get("streams"))
    except (json.JSONDecodeError, ValueError):
        has_audio = False

    fmt_dur = _probe_format_duration_sec(input_path)
    cap = float(max_input_duration_sec) if max_input_duration_sec and max_input_duration_sec > 0 else None
    if cap and fmt_dur:
        pad_target = min(fmt_dur, cap)
    elif cap:
        pad_target = cap
    else:
        pad_target = fmt_dur

    scale_pad = (
        "scale=1280:720:force_original_aspect_ratio=decrease,"
        "pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30"
    )

    # When cap is set, trim video and audio to the same [0, cap] window so stitched
    # output carries source audio for the prefix segment and CTA audio for the CTA segment.
    if cap:
        t = f"{cap:.6f}".rstrip("0").rstrip(".")
        vchain = (
            f"[0:v]trim=start=0:duration={t},setpts=PTS-STARTPTS,{scale_pad}[vout]"
        )
        if has_audio:
            achain = (
                f"[0:a]atrim=start=0:duration={t},asetpts=PTS-STARTPTS,"
                "aresample=44100,aformat=channel_layouts=stereo[aout]"
            )
            filter_complex = f"{vchain};{achain}"
            cmd_opts_tail: list[str] = []
        else:
            filter_complex = (
                f"{vchain};"
                f"aevalsrc=0|0:sample_rate=44100:channel_layout=stereo:duration={t}[aout]"
            )
            cmd_opts_tail = []
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", "[aout]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            *cmd_opts_tail,
            output_path,
        ]
    else:
        video_filter = f"[0:v]{scale_pad}[vout]"
        if has_audio:
            # Simply resample + reformat the existing audio track.  Do NOT use
            # apad=whole_dur here: that filter relies on the container-level
            # duration probe which can differ slightly from the real audio stream
            # duration, causing apad to produce misaligned timestamps that the
            # concat filter then reads as empty/silent audio on the CTA segment.
            filter_complex = (
                f"{video_filter};"
                "[0:a]aresample=44100,aformat=channel_layouts=stereo[aout]"
            )
            cmd_opts_tail = []
        else:
            filter_complex = (
                f"{video_filter};"
                "aevalsrc=0|0:sample_rate=44100:channel_layout=stereo[aout]"
            )
            cmd_opts_tail = ["-shortest"]

        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", "[aout]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            *cmd_opts_tail,
            output_path,
        ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg normalization failed:\n{result.stderr[-800:]}")


def _black_silent_gap_for_concat(output_path: str, duration_sec: float) -> None:
    """
    Black video + silent stereo AAC, same nominal format as _normalize_for_concat
    (1280x720, 30fps, h264, aac 128k) for concat demuxer compatibility.
    """
    if duration_sec <= 0:
        raise ValueError("gap duration must be positive")
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=black:s=1280x720:r=30",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-t", str(duration_sec),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg gap generation failed:\n{result.stderr[-800:]}")


def _stitch_videos_ffmpeg(
    video1_path: str,
    video2_path: str,
    output_path: str,
    *,
    gap_seconds: float = 0,
    source_prefix_seconds: float | None = None,
) -> str:
    """
    Stitch [video1 (or its first source_prefix_seconds)] then [video2].

    Audio for the ENTIRE output comes from video1 (the source) only.
    video1's audio plays continuously through both the source-prefix segment
    AND the CTA segment — the CTA's own audio track is ignored entirely.

    The source input is looped (-stream_loop -1) so that its audio always
    covers the full stitched duration even when the source file is shorter
    than source_prefix_seconds + CTA duration.  The audio is then trimmed
    with atrim to exactly total_dur so there is no overhang.

    If video1 has no audio, a silent track of the appropriate length is used.

    Both clips are scaled to 1280×720 / 30fps / H.264 + AAC 128k in a single pass.
    """
    scale_pad = (
        "scale=1280:720:force_original_aspect_ratio=decrease,"
        "pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30"
    )

    # ── Probe whether video1 has an audio stream ──────────────────────────────
    _pr = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-print_format", "json",
            video1_path,
        ],
        capture_output=True, text=True,
    )
    try:
        src_has_audio = bool(json.loads(_pr.stdout or "{}").get("streams"))
    except (json.JSONDecodeError, ValueError):
        src_has_audio = False

    src_sec = (
        float(source_prefix_seconds)
        if source_prefix_seconds and source_prefix_seconds > 0
        else None
    )
    gap_sec = float(gap_seconds) if gap_seconds and gap_seconds > 0 else 0.0

    # Compute total output duration for precise audio trimming.
    # With -stream_loop -1 the source loops, so v1_dur only matters for the
    # no-src_sec case where we must bound the video to one natural play-through.
    v1_dur = _probe_format_duration_sec(video1_path) or 0.0
    v2_dur = _probe_format_duration_sec(video2_path) or 0.0
    # When src_sec is set, the source video is always trimmed to exactly src_sec
    # (looping fills any gap if v1_dur < src_sec).
    used_v1_dur = float(src_sec) if src_sec else v1_dur
    total_dur = used_v1_dur + gap_sec + v2_dur  # seconds

    # ── Build filter_complex ──────────────────────────────────────────────────
    parts: list[str] = []

    # Video 1: always trim — either to src_sec (explicit prefix) or to the
    # file's natural duration (no prefix).  This prevents infinite looping of
    # the source video when -stream_loop -1 is active.
    if src_sec:
        t = f"{src_sec:.6f}".rstrip("0").rstrip(".")
    elif v1_dur > 0:
        t = f"{v1_dur:.3f}"
    else:
        t = None

    if t:
        parts.append(
            f"[0:v]trim=start=0:duration={t},setpts=PTS-STARTPTS,{scale_pad}[vA]"
        )
    else:
        parts.append(f"[0:v]{scale_pad}[vA]")

    # Video 2 (CTA): scale only — audio is NOT taken from this input
    parts.append(f"[1:v]{scale_pad}[vB]")

    # Video-only concat
    if gap_sec > 0:
        gap_t = f"{gap_sec:.3f}"
        # Inline black-frame source via lavfi color filter
        parts.append(
            f"color=c=black:size=1280x720:rate=30:duration={gap_t},format=yuv420p[gv]"
        )
        parts.append("[vA][gv][vB]concat=n=3:v=1:a=0[outv]")
    else:
        parts.append("[vA][vB]concat=n=2:v=1:a=0[outv]")

    # Audio: source video1 only.
    # Because the source input is looped (-stream_loop -1), the audio stream
    # is effectively infinite — atrim simply cuts it at total_dur.
    # No apad needed: looping always supplies enough samples.
    extra_flags: list[str] = []
    if src_has_audio:
        if total_dur > 0:
            t_total = f"{total_dur:.3f}"
            parts.append(
                f"[0:a]atrim=start=0:duration={t_total},asetpts=PTS-STARTPTS,"
                f"aresample=44100,aformat=channel_layouts=stereo[outa]"
            )
        else:
            # Duration unknown — resample; -shortest will stop at video end
            parts.append(
                "[0:a]aresample=44100,aformat=channel_layouts=stereo[outa]"
            )
            extra_flags = ["-shortest"]
    else:
        # No audio in source — generate silence for the full output
        if total_dur > 0:
            t_total = f"{total_dur:.3f}"
            parts.append(
                f"aevalsrc=0|0:sample_rate=44100:channel_layout=stereo"
                f":duration={t_total}[outa]"
            )
        else:
            parts.append(
                "aevalsrc=0|0:sample_rate=44100:channel_layout=stereo[outa]"
            )
            extra_flags = ["-shortest"]

    filter_complex = ";".join(parts)

    result = subprocess.run(
        [
            "ffmpeg", "-y",
            # Loop the source so its audio repeats to cover the full output.
            # The video trim filter above bounds how many frames are used.
            "-stream_loop", "-1", "-i", video1_path,
            "-i", video2_path,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "[outa]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            *extra_flags,
            output_path,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg stitch failed:\n{result.stderr[-800:]}")

    return output_path


def _stitch_downloaded_with_cta(download_item: dict, cta_video_path: str, stitch_output_dir: str) -> dict:
    """Stitch one downloaded video with CTA (downloaded clip first, CTA second)."""
    source_url = download_item.get("source_url", "")
    downloaded_path = download_item.get("filepath", "")
    if not downloaded_path or not os.path.exists(downloaded_path):
        return {
            "source_url": source_url,
            "error": "Downloaded file missing; skipping stitch",
        }

    stitched_filename = f"stitched_{os.path.basename(downloaded_path)}"
    stitched_path = os.path.join(stitch_output_dir, stitched_filename)
    try:
        _stitch_videos_ffmpeg(downloaded_path, cta_video_path, stitched_path)
        return {
            "source_url": source_url,
            "downloaded_filepath": downloaded_path,
            "stitched_filepath": stitched_path,
        }
    except Exception as stitch_err:
        return {
            "source_url": source_url,
            "downloaded_filepath": downloaded_path,
            "error": str(stitch_err),
        }


def _download_and_stitch_video(
    video_url: str,
    download_dir: str,
    idx: int,
    uploaded_video_path: str | None,
    stitch_output_dir: str | None,
) -> dict:
    """
    Worker that runs the full pipeline for a single video:
      1. Download the video to download_dir.
      2. If an uploaded_video_path is provided, wait 5 seconds after the download
         completes, then stitch [uploaded video] → [downloaded video] using FFmpeg.
      3. Return a combined result dict with keys:
           download  – the download manifest (filepath, source_url, …)
           stitch    – stitched_filepath or error (None if stitching not requested)
    This function is meant to be submitted to a ThreadPoolExecutor so that every
    video runs its download + 5s wait + stitch in parallel with the others.
    """
    # Step 1 – download
    download_result = _download_single_profile_video(video_url, download_dir)
    download_result["index"] = idx

    stitch_result = None

    # Step 2 – stitch (only if a video was uploaded and download succeeded)
    if uploaded_video_path and stitch_output_dir:
        downloaded_path = download_result.get("filepath", "")
        if downloaded_path and os.path.exists(downloaded_path):
            # Wait 5 seconds after THIS video finished downloading before stitching
            time.sleep(5)

            stitched_filename = f"stitched_{os.path.basename(downloaded_path)}"
            stitched_path = os.path.join(stitch_output_dir, stitched_filename)
            try:
                _stitch_videos_ffmpeg(uploaded_video_path, downloaded_path, stitched_path)
                stitch_result = {
                    "source_url": video_url,
                    "downloaded_filepath": downloaded_path,
                    "stitched_filepath": stitched_path,
                }
            except Exception as stitch_err:
                stitch_result = {
                    "source_url": video_url,
                    "downloaded_filepath": downloaded_path,
                    "error": str(stitch_err),
                }
        else:
            stitch_result = {
                "source_url": video_url,
                "error": "Download did not produce a file; skipping stitch",
            }

    return {"download": download_result, "stitch": stitch_result}


def _collect_instagram_video_urls(profile_url: str, limit: int) -> list[str]:
    """
    Collect up to `limit` video post URLs from an Instagram profile using instaloader.
    instaloader uses Instagram's mobile API and is far more reliable than yt-dlp's
    instagram:user extractor for profile-level discovery.

    Auth: loads a session file from INSTALOADER_SESSION_FILE env var (created by running
      `instaloader --login=<username>` once on your machine).
    Falls back to an anonymous (rate-limited) request if no session is configured.

    Returns a list of individual reel/post URLs that yt-dlp can then download.
    """
    ig_username = os.getenv("INSTAGRAM_USERNAME", "")
    session_file = os.getenv("INSTALOADER_SESSION_FILE", "")

    L = instaloader.Instaloader(
        quiet=True,
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        post_metadata_txt_pattern="",
    )

    # Load saved session so we're authenticated
    if ig_username and session_file and os.path.exists(session_file):
        try:
            L.load_session_from_file(ig_username, session_file)
        except Exception as sess_err:
            print(f"⚠️  instaloader: could not load session ({sess_err}); trying anonymously")

    # Extract the username from the profile URL
    # Handles: https://instagram.com/username, https://instagram.com/username/, etc.
    parsed_path = urlparse(profile_url).path.strip("/")
    insta_user = parsed_path.split("/")[0].lstrip("@")
    if not insta_user:
        raise ValueError(f"Could not extract Instagram username from URL: {profile_url}")

    try:
        profile = instaloader.Profile.from_username(L.context, insta_user)
    except instaloader.exceptions.ProfileNotExistsException:
        raise RuntimeError(f"Instagram profile @{insta_user} does not exist or is not accessible")
    except instaloader.exceptions.LoginRequiredException:
        raise RuntimeError(
            f"Instagram profile @{insta_user} requires authentication. "
            "Set INSTAGRAM_USERNAME and INSTALOADER_SESSION_FILE env vars. "
            "Create the session with: instaloader --login=<your_username>"
        )

    video_urls = []
    for post in profile.get_posts():
        if post.is_video:
            # Use the reel/post page URL so yt-dlp can resolve and download it
            video_urls.append(f"https://www.instagram.com/p/{post.shortcode}/")
            if len(video_urls) >= limit:
                break

    return video_urls


def _collect_profile_video_urls(profile_url: str, limit: int) -> list[str]:
    """Resolve up to `limit` supported video URLs from a profile/channel URL."""
    # Instagram's yt-dlp user extractor is frequently broken; use instaloader instead
    if "instagram.com" in (urlparse(profile_url).hostname or ""):
        return _collect_instagram_video_urls(profile_url, limit)

    opts = _profile_discovery_opts(profile_url, limit)
    info = _ydl_extract(opts, profile_url, download=False)

    if not info:
        raise RuntimeError("yt-dlp returned no metadata for this profile URL")

    entries = info.get("entries") if isinstance(info, dict) else None
    if not isinstance(entries, list):
        single = _entry_to_video_url(info if isinstance(info, dict) else {})
        if single and is_video_url(single):
            return [single]
        raise RuntimeError("yt-dlp did not return playable video entries for this profile URL")

    urls = []
    seen = set()
    for entry in entries:
        video_url = _entry_to_video_url(entry)
        if not video_url or not is_video_url(video_url) or video_url in seen:
            continue
        seen.add(video_url)
        urls.append(video_url)
        if len(urls) >= limit:
            break
    if not urls:
        raise RuntimeError("yt-dlp resolved profile metadata but returned zero video entries")
    return urls


def _profile_storage_paths(profile_url: str) -> tuple[str, str]:
    profile_key = hashlib.sha256(profile_url.encode("utf-8")).hexdigest()[:16]
    profile_dir = os.path.join(PROFILE_VIDEO_DOWNLOADS_DIR, profile_key)
    index_path = os.path.join(profile_dir, "index.json")
    return profile_dir, index_path


def _load_profile_index(index_path: str) -> dict:
    if not os.path.exists(index_path):
        return {"downloaded_urls": {}, "last_updated": None}
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"downloaded_urls": {}, "last_updated": None}
        if not isinstance(data.get("downloaded_urls"), dict):
            data["downloaded_urls"] = {}
        return data
    except Exception:
        return {"downloaded_urls": {}, "last_updated": None}


def _save_profile_index(index_path: str, index_data: dict) -> None:
    index_data["last_updated"] = datetime.utcnow().isoformat() + "Z"
    tmp_path = f"{index_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=True, indent=2)
    os.replace(tmp_path, index_path)


def _download_single_profile_video(video_url: str, output_dir: str) -> dict:
    """Download one video URL to output_dir and return manifest info."""
    temp_cookies_path = None
    ydl_opts = {
        "format": "best[height<=720]/best",
        "outtmpl": os.path.join(output_dir, "%(extractor)s_%(id)s.%(ext)s"),
        "restrictfilenames": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {"youtube": _yt_extractor_args()},
    }
    # Copy cookies to a writable path (yt-dlp writes the jar back; Render
    # secret files are read-only → would raise Errno 30).
    temp_cookies_path = _prepare_cookiefile(output_dir, video_url)
    if temp_cookies_path:
        ydl_opts["cookiefile"] = temp_cookies_path

    if "instagram.com" in video_url.lower() and ydl_opts.get("cookiefile"):
        ydl_opts.setdefault("extractor_args", {})["instagram"] = {"webpage_display": ["Desktop"]}
    _proxy = _ytdlp_proxy(video_url)
    if _proxy:
        ydl_opts["proxy"] = _proxy

    filepath = ""
    try:
        info = _ydl_extract(ydl_opts, video_url, download=True) or {}
        if isinstance(info, dict):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as _ydl:
                    filepath = _ydl.prepare_filename(info)
            except Exception:
                filepath = ""
            requested = info.get("requested_downloads") or []
            if requested and isinstance(requested[0], dict):
                filepath = requested[0].get("filepath") or filepath
            if not filepath:
                maybe_name = info.get("_filename")
                if isinstance(maybe_name, str):
                    filepath = maybe_name
    finally:
        if temp_cookies_path and os.path.exists(temp_cookies_path):
            try:
                os.unlink(temp_cookies_path)
            except Exception:
                pass

    if not filepath and isinstance(info, dict):
        requested = info.get("requested_downloads") or []
        if requested and isinstance(requested[0], dict):
            filepath = requested[0].get("filepath") or ""

    return {
        "source_url": video_url,
        "title": (info.get("title") if isinstance(info, dict) else "") or "",
        "duration": (info.get("duration") if isinstance(info, dict) else None),
        "filepath": filepath,
    }


def _extract_one_frame(video_path: str, timestamp: float, out_path: str) -> str | None:
    """Extract a single frame at timestamp; return data URL or None. Used for parallel extraction."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-ss", str(timestamp), "-i", video_path,
                "-vframes", "1", "-q:v", "2", out_path,
            ],
            capture_output=True,
            timeout=EXTRACT_RECIPE_TIMEOUT,
        )
        if result.returncode != 0 or not os.path.exists(out_path):
            return None
        with open(out_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def _extract_frames_batch_ffmpeg(
    video_path: str, duration_sec: float | None, num_frames: int, out_dir: str
) -> list[str]:
    """Single ffmpeg pass — faster than N separate subprocess calls on short reels."""
    pattern = os.path.join(out_dir, "frame_%02d.jpg")
    w = _frame_jpeg_max_width()
    if duration_sec and duration_sec > 0:
        vf = f"fps={num_frames / duration_sec},scale={w}:-1"
    else:
        vf = f"fps=1/5,scale={w}:-1"
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", vf, "-frames:v", str(num_frames),
            "-q:v", "7", pattern,
        ],
        capture_output=True,
        timeout=EXTRACT_RECIPE_TIMEOUT,
    )
    if result.returncode != 0:
        return []
    urls = []
    for name in sorted(f for f in os.listdir(out_dir) if f.endswith(".jpg")):
        path = os.path.join(out_dir, name)
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        urls.append(f"data:image/jpeg;base64,{b64}")
    return urls


def _extract_frame_data_urls_from_video(video_path: str, duration_sec: float | None, num_frames: int = 8) -> list:
    """
    Extract evenly spaced frames from video as JPEG data URLs.
    Tries one ffmpeg pass first; falls back to parallel single-frame extraction.
    """
    num_frames = min(max(1, num_frames), MAX_EXTRACT_RECIPE_IMAGES)
    if num_frames == 1:
        single = _extract_middle_frame_data_url(video_path, duration_sec)
        if single:
            return [single]

    out_dir = tempfile.mkdtemp()
    try:
        batch = _extract_frames_batch_ffmpeg(video_path, duration_sec, num_frames, out_dir)
        if batch:
            return batch

        if duration_sec and duration_sec > 0:
            interval = duration_sec / (num_frames + 1)
            timestamps = [interval * (i + 1) for i in range(num_frames)]
        else:
            timestamps = [float(i * 10) for i in range(num_frames)]

        results = [None] * len(timestamps)
        max_workers = min(4, len(timestamps))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _extract_one_frame,
                    video_path,
                    t,
                    os.path.join(out_dir, f"frame_{i:02d}.jpg"),
                ): i
                for i, t in enumerate(timestamps)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    data_url = future.result()
                    if data_url:
                        results[idx] = data_url
                except Exception:
                    pass
        return [r for r in results if r is not None]
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def _is_likely_non_speech(transcript: str) -> bool:
    """True if transcript looks like music/no recipe speech (e.g. '[Music]', '[Applause]', or very short)."""
    t = (transcript or "").strip()
    if len(t) < MIN_TRANSCRIPT_LENGTH_FOR_VIDEO:
        return True
    cleaned = re.sub(r"\[[\w\s]+\]", "", t, flags=re.IGNORECASE).strip()
    return len(cleaned) < MIN_TRANSCRIPT_LENGTH_FOR_VIDEO


def _extract_audio_from_video_file(video_path: str) -> bytes | None:
    """Extract audio from an already-downloaded video file using ffmpeg. Returns mp3 bytes or None on failure."""
    out_dir = tempfile.mkdtemp()
    try:
        out_path = os.path.join(out_dir, "audio.mp3")
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "libmp3lame", "-q:a", "2", out_path,
            ],
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0 or not os.path.exists(out_path):
            return None
        with open(out_path, "rb") as f:
            return f.read()
    except Exception:
        return None
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def _run_frame_vision_fallback(video_url: str, meta: dict) -> tuple[dict | None, dict | None, str | None]:
    """
    Fallback when audio transcript doesn't yield a recipe: download video, extract frames, run vision.
    Returns (recipe_dict, source_dict, extraction_method) or (None, None, None) on failure.
    meta can be from _download_audio_mp3 (we have duration/title/etc.) or from _download_video_to_file.
    """
    temp_dir = None
    try:
        temp_dir, video_path, video_meta = _download_video_to_file(video_url, fast=True)
        duration = video_meta.get("duration") or 0
        n_frames = _social_video_frames_for_duration(video_url, duration, meta=video_meta)
        frame_urls = _extract_frame_data_urls_from_video(
            video_path, duration, num_frames=n_frames
        )
        if not frame_urls:
            return None, None, None
        recipe = extract_recipe_from_video_frames_llm(
            frame_urls,
            caption=_video_caption_text(video_meta),
            social=_is_social_video_url(video_url),
        )
        method = "video_frames_vision"
        if recipe:
            recipe = _normalize_extracted_recipe(recipe)
        caption_text = _video_caption_text(video_meta)
        vision_sparse = (
            not _social_recipe_is_complete(recipe)
            if _is_social_video_url(video_url)
            else not _recipe_has_usable_content(recipe)
        )
        if vision_sparse:
            caption_recipe = _extract_recipe_from_video_caption(video_meta)
            if _social_caption_recipe_acceptable(caption_recipe, caption_text):
                recipe = caption_recipe
                method = "caption_llm"
        if not _recipe_has_usable_content(recipe):
            return None, None, None
        source = _build_video_recipe_source(video_url, video_meta, recipe)
        return recipe, source, method
    except Exception as e:
        print(f"⚠️ Frame+vision fallback failed: {e}")
        return None, None, None
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _run_frame_vision_fallback_from_path(
    video_path: str, video_url: str, meta: dict
) -> tuple[dict | None, dict | None, str | None]:
    """
    Frame+vision fallback using an already-downloaded video file (no extra download).
    Returns (recipe_dict, source_dict, extraction_method) or (None, None, None) on failure.
    """
    try:
        duration = float(meta.get("vision_duration") or meta.get("duration") or 0)
        n_frames = _social_video_frames_for_duration(video_url, duration, meta=meta)
        print(f"🎬 Vision fallback: extracting {n_frames} frame(s) from {duration}s video")
        frame_urls = _extract_frame_data_urls_from_video(
            video_path, duration, num_frames=n_frames
        )
        if not frame_urls:
            return None, None, None
        t_llm = time.time()
        caption_text = _video_caption_text(meta)
        recipe = extract_recipe_from_video_frames_llm(
            frame_urls,
            caption=caption_text,
            social=_is_social_video_url(video_url),
        )
        print(f"🎬 Vision LLM finished in {time.time() - t_llm:.2f}s ({len(frame_urls)} frame(s))")
        method = "video_frames_vision"
        if recipe:
            recipe = _normalize_extracted_recipe(recipe)
        vision_sparse = (
            not _social_recipe_is_complete(recipe)
            if _is_social_video_url(video_url)
            else not _recipe_has_usable_content(recipe)
        )
        if vision_sparse:
            print("📝 Vision empty/sparse; trying post caption...")
            caption_recipe = _extract_recipe_from_video_caption(meta)
            if _social_caption_recipe_acceptable(caption_recipe, caption_text):
                recipe = caption_recipe
                method = "caption_llm"
        if not _recipe_has_usable_content(recipe):
            return None, None, None
        source = _build_video_recipe_source(video_url, meta, recipe)
        return recipe, source, method
    except Exception as e:
        print(f"⚠️ Frame+vision fallback (from path) failed: {e}")
        return None, None, None


def _max_slideshow_images() -> int:
    try:
        return max(2, min(int(os.getenv("EXTRACT_RECIPE_SLIDESHOW_MAX_IMAGES", "6")), 12))
    except ValueError:
        return 6


def _slideshow_frame_max_width() -> int:
    try:
        return max(256, min(int(os.getenv("EXTRACT_RECIPE_SLIDESHOW_MAX_WIDTH", "512")), 768))
    except ValueError:
        return 512


def _select_slideshow_slide_urls(image_urls: list[str]) -> list[str]:
    """Evenly sample slides so long carousels stay fast without skipping first/last context."""
    max_n = _max_slideshow_images()
    n = len(image_urls)
    if n <= max_n:
        return image_urls
    indices = [round(i * (n - 1) / (max_n - 1)) for i in range(max_n)]
    seen: set[int] = set()
    selected: list[str] = []
    for idx in indices:
        if idx in seen:
            continue
        seen.add(idx)
        selected.append(image_urls[idx])
    return selected


def _resize_image_bytes_to_jpeg_data_url(image_bytes: bytes) -> str:
    """Downscale slide bytes for faster vision LLM (same approach as video frame extraction)."""
    out_dir = tempfile.mkdtemp()
    in_path = os.path.join(out_dir, "slide_in")
    out_path = os.path.join(out_dir, "slide.jpg")
    w = _slideshow_frame_max_width()
    try:
        with open(in_path, "wb") as f:
            f.write(image_bytes)
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", in_path,
                "-vf", f"scale={w}:-1", "-q:v", "7",
                out_path,
            ],
            capture_output=True,
            timeout=EXTRACT_RECIPE_TIMEOUT,
        )
        if result.returncode != 0 or not os.path.exists(out_path):
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded}"
        with open(out_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def _slideshow_request_headers() -> dict:
    """Desktop Chrome UA — TikTok photo slideshow JSON is empty with mobile/iPhone UA."""
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }


def _strip_url_query(url: str) -> str:
    """Drop tracking query params before fetching TikTok pages."""
    p = urlparse(url or "")
    if not p.scheme or not p.netloc:
        return url
    return urlunparse((p.scheme, p.netloc, p.path, "", "", ""))


def _slideshow_requests_session() -> requests.Session:
    sess = requests.Session()
    cf = _prepare_cookiefile()
    if cf:
        try:
            cj = http.cookiejar.MozillaCookieJar(cf)
            cj.load(ignore_discard=True, ignore_expires=True)
            sess.cookies = cj
        except Exception:
            pass
    return sess


def _resolve_tiktok_short_url(url: str) -> str:
    """Resolve TikTok short links only — used by slideshow fallback, not the main video pipeline."""
    try:
        host = (urlparse(url).hostname or "").lower()
        if host not in {"vt.tiktok.com", "vm.tiktok.com", "tiktok.com", "www.tiktok.com"}:
            return url
        if host in {"tiktok.com", "www.tiktok.com"} and not re.search(r"/t/", url):
            return url
        sess = _slideshow_requests_session()
        resp = sess.get(
            url,
            headers=_slideshow_request_headers(),
            allow_redirects=True,
            timeout=EXTRACT_RECIPE_TIMEOUT,
        )
        return resp.url or url
    except Exception as e:
        print(f"[slideshow] TikTok short-url resolution failed for {url}: {e}")
        return url


def _is_tiktok_photo_url(url: str) -> bool:
    """True for TikTok /photo/ slideshow posts (not /video/ reels)."""
    try:
        p = urlparse(url or "")
        host = (p.hostname or "").lower()
        if not any(h in host for h in ("tiktok.com", "vt.tiktok.com", "vm.tiktok.com")):
            return False
        return bool(re.search(r"/photo/\d+", p.path, re.I))
    except Exception:
        return False


def _is_tiktok_video_reel_url(url: str) -> bool:
    """TikTok video/reel URLs — excludes /photo/ slideshow posts."""
    return _is_tiktok_video_url(url) and not _is_tiktok_photo_url(url)


def _is_instagram_post_url(url: str) -> bool:
    lowered = (url or "").lower()
    return "instagram.com/p/" in lowered


def _instagram_post_shortcode(url: str) -> str | None:
    match = re.search(r"instagram\.com/(?:p|reel|tv)/([^/?#]+)", url or "", re.I)
    return match.group(1) if match else None


def _make_instaloader() -> instaloader.Instaloader:
    ig_username = os.getenv("INSTAGRAM_USERNAME", "")
    session_file = os.getenv("INSTALOADER_SESSION_FILE", "")
    L = instaloader.Instaloader(
        quiet=True,
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        post_metadata_txt_pattern="",
    )
    if ig_username and session_file and os.path.exists(session_file):
        try:
            L.load_session_from_file(ig_username, session_file)
        except Exception as sess_err:
            print(f"⚠️  instaloader: could not load session ({sess_err}); trying anonymously")
    return L


def _collect_tiktok_desc_from_json(obj, best: list[str], best_lang: list[str] | None = None, locked: list[bool] | None = None) -> None:
    """Walk TikTok page JSON; prefer itemStruct.desc (actual post) over unrelated longer strings."""
    if locked is None:
        locked = [False]
    if isinstance(obj, dict):
        desc = None
        item_struct = obj.get("itemStruct")
        if isinstance(item_struct, dict):
            desc = item_struct.get("desc")
            if best_lang is not None:
                for lk in ("textLanguage", "language", "descLanguage", "lang"):
                    code = _normalize_language_code(item_struct.get(lk))
                    if code:
                        best_lang[:] = [code]
            if isinstance(desc, str) and desc.strip():
                best[:] = [desc.strip()]
                locked[0] = True
        elif not locked[0] and isinstance(obj.get("desc"), str) and any(
            k in obj for k in ("id", "video", "createTime", "author", "stats")
        ):
            desc = obj.get("desc")
        if not locked[0] and isinstance(desc, str):
            text = desc.strip()
            if len(text) > len(best[0] if best else ""):
                best[:] = [text]
        for value in obj.values():
            _collect_tiktok_desc_from_json(value, best, best_lang, locked)
    elif isinstance(obj, list):
        for item in obj:
            _collect_tiktok_desc_from_json(item, best, best_lang, locked)


def _fetch_tiktok_page_info(url: str) -> tuple[str | None, str | None]:
    """Fetch TikTok post caption + language from page HTML (shown after tapping 'more')."""
    if (os.getenv("EXTRACT_RECIPE_TIKTOK_PAGE_CAPTION") or "1").strip().lower() in ("0", "false", "no"):
        return None, None
    if "tiktok.com" not in (url or "").lower():
        return None, None
    try:
        resolved = _resolve_tiktok_short_url(url)
        sess = _slideshow_requests_session()
        resp = sess.get(
            resolved,
            headers=_slideshow_request_headers(),
            allow_redirects=True,
            timeout=EXTRACT_RECIPE_TIMEOUT,
        )
        resp.raise_for_status()
        best: list[str] = [""]
        best_lang: list[str] = []
        for pattern in (
            r'<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__"[^>]*>(.*?)</script>',
            r'<script id="SIGI_STATE"[^>]*>(.*?)</script>',
        ):
            match = re.search(pattern, resp.text, re.DOTALL)
            if not match:
                continue
            try:
                data = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            _collect_tiktok_desc_from_json(data, best, best_lang)
        caption = best[0] or None
        lang = best_lang[0] if best_lang else None
        if caption:
            print(f"📝 TikTok page caption fetched ({len(caption)} chars)")
        if lang:
            print(f"🌐 TikTok post language: {lang}")
        return caption, lang
    except Exception as e:
        print(f"⚠️ TikTok page caption fetch failed: {e}")
        return None, None


def _fetch_tiktok_page_caption(url: str) -> str | None:
    caption, _ = _fetch_tiktok_page_info(url)
    return caption


def _instaloader_import_cookies(L, cookiefile: str) -> bool:
    """Inject Instagram cookies from a Netscape cookies.txt into instaloader's
    session so requests are authenticated. Returns True if an Instagram
    'sessionid' cookie was found (i.e. the session should be logged in)."""
    try:
        import http.cookiejar
        cj = http.cookiejar.MozillaCookieJar()
        cj.load(cookiefile, ignore_discard=True, ignore_expires=True)
    except Exception as e:
        print(f"⚠️ instaloader: could not read cookies file: {e}")
        return False
    found_sessionid = False
    for c in cj:
        if "instagram" not in (c.domain or "").lower():
            continue
        try:
            L.context._session.cookies.set(c.name, c.value, domain=c.domain)
        except Exception:
            continue
        if c.name == "sessionid" and c.value:
            found_sessionid = True
    return found_sessionid


def _fetch_instagram_page_caption(url: str) -> str | None:
    """Fetch full Instagram post/reel caption via instaloader.

    Instagram blocks *anonymous* instaloader access (the source of the
    'NoneType object is not subscriptable' errors), so we only attempt the fetch
    when the session is authenticated — either via a loaded session file or by
    importing the existing yt-dlp Instagram cookies. When unauthenticated we skip
    quietly, since yt-dlp already supplies the caption via the post description.
    """
    if (os.getenv("EXTRACT_RECIPE_INSTAGRAM_PAGE_CAPTION") or "1").strip().lower() in ("0", "false", "no"):
        return None
    if "instagram.com" not in (url or "").lower():
        return None
    shortcode = _instagram_post_shortcode(url)
    if not shortcode:
        return None

    L = _make_instaloader()  # loads INSTALOADER_SESSION_FILE if configured

    # If the session file didn't log us in, try the yt-dlp Instagram cookies.
    if not getattr(L.context, "is_logged_in", False):
        cookie_tmp = None
        try:
            cookie_tmp = tempfile.mkdtemp()
            cf = _prepare_cookiefile(cookie_tmp, url)
            if cf:
                _instaloader_import_cookies(L, cf)
        except Exception as e:
            print(f"⚠️ instaloader: cookie import failed: {e}")
        finally:
            if cookie_tmp:
                shutil.rmtree(cookie_tmp, ignore_errors=True)

    if not getattr(L.context, "is_logged_in", False):
        # Anonymous access is reliably blocked by Instagram — skip without noise.
        return None

    try:
        post = instaloader.Post.from_shortcode(L.context, shortcode)
        caption = (post.caption or "").strip()
        if caption:
            print(f"📝 Instagram page caption fetched ({len(caption)} chars)")
        return caption or None
    except Exception as e:
        print(f"⚠️ Instagram page caption fetch failed: {e}")
        return None


def _fetch_social_reel_meta(video_url: str) -> dict:
    """Parallel page fetch + yt-dlp info for TikTok/IG."""
    lowered = (video_url or "").lower()
    page_box: dict = {"caption": None, "lang": None}
    yt_box: dict = {"meta": None, "error": None}

    def _fetch_page():
        if "tiktok.com" in lowered:
            page_box["caption"], page_box["lang"] = _fetch_tiktok_page_info(video_url)
        elif "instagram.com" in lowered:
            page_box["caption"] = _fetch_instagram_page_caption(video_url)

    def _fetch_yt():
        try:
            info = _yt_meta(video_url)
            meta = _video_meta_from_yt_info(info, video_url)
            yt_lang = _normalize_language_code(info.get("language") or info.get("lang"))
            if yt_lang:
                meta["language"] = yt_lang
            yt_box["meta"] = meta
        except Exception as e:
            yt_box["error"] = e

    with ThreadPoolExecutor(max_workers=2) as executor:
        f_page = executor.submit(_fetch_page)
        f_yt = executor.submit(_fetch_yt)
        f_page.result()
        f_yt.result()

    page_caption = page_box["caption"]
    page_lang = page_box["lang"]

    if yt_box["meta"]:
        meta = yt_box["meta"]
    else:
        print(f"⚠️ yt-dlp metadata failed, using page caption only: {yt_box['error']}")
        meta = {
            "duration": None,
            "title": "",
            "description": page_caption or "",
            "provider": urlparse(video_url).hostname or "",
            "extractor": "",
            "webpage_url": video_url,
            "thumbnail": None,
        }

    existing = (meta.get("description") or "").strip()
    title = (meta.get("title") or "").strip()
    if page_caption and len(page_caption) > len(existing):
        if not _is_unrelated_tiktok_description(page_caption, title):
            print(f"📝 Post caption enriched: {len(existing)} → {len(page_caption)} chars")
            meta["description"] = page_caption
        else:
            print("📝 Skipping unrelated page caption (language/topic mismatch with title)")
    if page_lang:
        meta["language"] = page_lang
    return meta


def _enrich_social_meta_from_page(meta: dict, video_url: str) -> dict:
    """Replace truncated yt-dlp descriptions with full post captions from the platform page."""
    meta = dict(meta)
    existing = (meta.get("description") or "").strip()
    page_caption = None
    lowered = (video_url or "").lower()
    if "tiktok.com" in lowered:
        page_caption = _fetch_tiktok_page_caption(video_url)
    elif "instagram.com" in lowered:
        page_caption = _fetch_instagram_page_caption(video_url)
    if page_caption and len(page_caption) > len(existing):
        print(f"📝 Post caption enriched: {len(existing)} → {len(page_caption)} chars")
        meta["description"] = page_caption
    return meta


def _merge_meta_keep_longer_description(meta: dict, update: dict) -> dict:
    merged = dict(meta)
    prev_lang = merged.get("language")
    merged.update(update)
    old = (meta.get("description") or "").strip()
    new = (update.get("description") or "").strip()
    if len(old) > len(new):
        merged["description"] = old
    if prev_lang:
        merged["language"] = prev_lang
    return merged


def _tiktok_photo_to_video_url(url: str) -> str:
    """TikTok photo posts expose full item JSON at the equivalent /video/ URL."""
    match = re.search(
        r"(https?://(?:www\.)?tiktok\.com/@[^/]+/)photo/(\d+)",
        url or "",
        re.I,
    )
    if match:
        return f"{match.group(1)}video/{match.group(2)}"
    return url


def _extract_tiktok_image_post_from_page(data: dict) -> tuple[list[str], dict] | None:
    """Parse slideshow images from TikTok __UNIVERSAL_DATA_FOR_REHYDRATION__ JSON."""
    if not isinstance(data, dict):
        return None
    scope = data.get("__DEFAULT_SCOPE__", data)
    if not isinstance(scope, dict):
        return None
    video_detail = scope.get("webapp.video-detail")
    if not isinstance(video_detail, dict):
        return None
    if video_detail.get("statusCode") not in (None, 0):
        return None
    item = (video_detail.get("itemInfo") or {}).get("itemStruct") or {}
    image_post = item.get("imagePost")
    if not isinstance(image_post, dict):
        return None
    images = image_post.get("images") or []
    urls: list[str] = []
    for img in images:
        if not isinstance(img, dict):
            continue
        image_url = img.get("imageURL") or img.get("imageUrl") or {}
        if isinstance(image_url, dict):
            url_list = image_url.get("urlList") or image_url.get("url_list") or []
            if url_list:
                urls.append(url_list[0])
    if not urls:
        return None
    title = (item.get("desc") or image_post.get("title") or "").strip()
    lang = None
    for lk in ("textLanguage", "language", "descLanguage", "lang"):
        lang = _normalize_language_code(item.get(lk))
        if lang:
            break
    meta = {
        "title": title,
        "description": title,
        "slide_count": len(urls),
        "provider": "tiktok.com",
        "extractor": "tiktok_photo",
    }
    if lang:
        meta["language"] = lang
    return urls, meta


def _walk_tiktok_image_post(obj):
    """Fallback walk for imagePost nested elsewhere in page JSON."""
    if isinstance(obj, dict):
        image_post = obj.get("imagePost")
        if isinstance(image_post, dict):
            images = image_post.get("images")
            if isinstance(images, list) and images:
                urls = []
                for img in images:
                    if not isinstance(img, dict):
                        continue
                    image_url = img.get("imageURL") or img.get("imageUrl")
                    if isinstance(image_url, dict):
                        url_list = image_url.get("urlList") or image_url.get("url_list") or []
                        if url_list:
                            urls.append(url_list[0])
                if urls:
                    title = image_post.get("title") or obj.get("desc") or obj.get("title") or ""
                    return urls, {"title": title, "slide_count": len(urls)}
        for value in obj.values():
            found = _walk_tiktok_image_post(value)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _walk_tiktok_image_post(item)
            if found:
                return found
    return None


def _parse_tiktok_slideshow_html(html: str) -> tuple[list[str], dict] | None:
    """Extract slideshow image URLs from TikTok page HTML (UNIVERSAL_DATA or SIGI_STATE)."""
    for pattern in (
        r'<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__"[^>]*>(.*?)</script>',
        r'<script id="SIGI_STATE"[^>]*>(.*?)</script>',
    ):
        match = re.search(pattern, html, re.DOTALL)
        if not match:
            continue
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        found = _extract_tiktok_image_post_from_page(data)
        if not found:
            found = _walk_tiktok_image_post(data)
        if found:
            return found
    return None


def _tiktok_slideshow_page_urls(url: str, *, resolved: str | None = None) -> list[str]:
    """Ordered page URLs to try — video page first, then original /photo/ (legacy)."""
    bases: list[str] = []
    for candidate in (_strip_url_query(url), _strip_url_query(resolved) if resolved else None):
        if candidate and candidate not in bases:
            bases.append(candidate)
    ordered: list[str] = []
    seen: set[str] = set()

    def _add(page_url: str) -> None:
        if page_url and page_url not in seen:
            seen.add(page_url)
            ordered.append(page_url)

    for base in bases:
        _add(_tiktok_photo_to_video_url(base))
        _add(base)
    return ordered


def _fetch_tiktok_photo_slideshow(url: str, *, resolved: str | None = None) -> tuple[list[str], dict] | None:
    """Download TikTok photo/slideshow pages and extract image CDN URLs."""
    if "tiktok.com" not in (url or "").lower():
        return None
    sess = _slideshow_requests_session()
    for fetch_url in _tiktok_slideshow_page_urls(url, resolved=resolved):
        try:
            print(f"📸 Trying TikTok slideshow page: {fetch_url}")
            resp = sess.get(
                fetch_url,
                headers=_slideshow_request_headers(),
                allow_redirects=True,
                timeout=EXTRACT_RECIPE_TIMEOUT,
            )
            resp.raise_for_status()
            found = _parse_tiktok_slideshow_html(resp.text)
            if found:
                image_urls, meta = found
                meta.setdefault("provider", "tiktok.com")
                meta.setdefault("extractor", "tiktok_photo")
                print(f"📸 TikTok photo slideshow: {len(image_urls)} image(s) from {fetch_url}")
                return image_urls, meta
        except Exception as e:
            print(f"⚠️ TikTok slideshow page failed ({fetch_url}): {e}")
    return None


def _fetch_instagram_carousel_slideshow(url: str) -> tuple[list[str], dict] | None:
    """Use instaloader to collect image URLs from a multi-image Instagram carousel."""
    if not _is_instagram_post_url(url):
        return None
    shortcode = _instagram_post_shortcode(url)
    if not shortcode:
        return None
    L = _make_instaloader()
    try:
        post = instaloader.Post.from_shortcode(L.context, shortcode)
    except instaloader.exceptions.InstaloaderException as e:
        print(f"⚠️ instaloader carousel lookup failed for {url}: {e}")
        return None

    if post.typename != "GraphSidecar":
        return None

    image_urls: list[str] = []
    for node in post.get_sidecar_nodes():
        if node.is_video:
            continue
        if node.display_url:
            image_urls.append(node.display_url)

    if len(image_urls) < 2:
        return None

    return image_urls, {
        "title": post.title or post.caption or "",
        "provider": "instagram.com",
        "extractor": "instagram_carousel",
        "slide_count": len(image_urls),
    }


def _slideshow_image_urls(url: str, *, resolved: str | None = None) -> tuple[list[str], dict] | None:
    """Return TikTok /photo/ slideshow image URLs only (never probes /video/ reels)."""
    if _is_tiktok_photo_url(url) or (resolved and _is_tiktok_photo_url(resolved)):
        return _fetch_tiktok_photo_slideshow(url, resolved=resolved)
    return None


def _tiktok_download_error_suggests_photo_post(error_str: str) -> bool:
    """yt-dlp 'Unsupported URL' on a /video/ link may be a photo-mode post."""
    lower = (error_str or "").lower()
    return any(
        m in lower
        for m in (
            "unsupported url",
            "photomode",
            "image post",
            "no video formats",
        )
    )


def _fetch_image_data_urls(image_urls: list[str]) -> list[str]:
    """Download sampled slide images, downscale, and return JPEG data URLs for vision LLM."""
    urls = _select_slideshow_slide_urls(image_urls)
    headers = _slideshow_request_headers()

    def _download_one(remote_url: str) -> str:
        resp = requests.get(remote_url, headers=headers, timeout=EXTRACT_RECIPE_TIMEOUT)
        resp.raise_for_status()
        return _resize_image_bytes_to_jpeg_data_url(resp.content)

    if len(urls) == 1:
        return [_download_one(urls[0])]

    data_urls = [None] * len(urls)
    with ThreadPoolExecutor(max_workers=min(8, len(urls))) as executor:
        futures = {executor.submit(_download_one, remote_url): idx for idx, remote_url in enumerate(urls)}
        for future in as_completed(futures):
            idx = futures[future]
            data_urls[idx] = future.result()
    return [u for u in data_urls if u]


def _extract_recipe_from_slideshow(url: str, url_key: str, slideshow: tuple[list[str], dict]):
    """Run fast vision LLM on confirmed slideshow images. Returns Flask (response, status)."""
    image_urls, meta = slideshow
    selected = _select_slideshow_slide_urls(image_urls)
    meta = dict(meta)
    meta["slides_used"] = len(selected)
    print(
        f"📸 Slideshow ({meta.get('extractor', 'unknown')}): "
        f"{len(selected)}/{len(image_urls)} slide(s) sampled"
    )
    t0 = time.time()
    try:
        t_dl = time.time()
        data_urls = _fetch_image_data_urls(image_urls)
        print(f"📥 Slideshow images ready in {time.time() - t_dl:.2f}s")
        t_llm = time.time()
        recipe = extract_recipe_from_slideshow_llm(
            data_urls, caption=meta.get("title", "")
        )
        print(f"🧠 Slideshow vision LLM finished in {time.time() - t_llm:.2f}s")
        lang = _finalize_recipe_for_response(recipe, meta)
        tags = extract_recipe_tags(recipe)
        source = {
            "type": "slideshow",
            "url": url,
            "provider": meta.get("provider", ""),
            "title": meta.get("title", "") or recipe.get("name", ""),
            "image": image_urls[0] if image_urls else None,
            "source_type": determine_source_type(url),
        }
        result = {
            "source": source,
            "recipe": recipe,
            "tags": tags,
            "transcript": None,
            "extraction": {"method": "slideshow_vision", "confidence": 0.6},
            "meta": meta,
            "language": lang,
        }
        result = _apply_response_language(result, url, meta)
        with _recipe_cache_lock:
            _recipe_cache[url_key] = result
        print(f"✅ Slideshow recipe extracted in {time.time() - t0:.2f}s")
        return jsonify(_apply_extract_recipe_image_pref(result)), 200
    except Exception as e:
        print(f"⚠️ Slideshow extraction failed: {e}")
        return jsonify({
            "error": "Failed to extract recipe from slideshow",
            "user_message": "We couldn't read this photo slideshow. Please try another link or add the recipe manually.",
            "details": str(e),
        }), 500


def _try_slideshow_fallback_on_download_error(
    url: str, url_key: str, *, download_error: str | None = None
):
    """
    Only invoked after the normal video download path fails.
    TikTok /photo/ and IG carousels only — regular TikTok/IG/YouTube videos are untouched.
    Returns a Flask (response, status) tuple on success/failure, or None to keep the original error.
    """
    resolved = _resolve_tiktok_short_url(url)
    slideshow = _slideshow_image_urls(url, resolved=resolved)
    if (
        not slideshow
        and _is_tiktok_video_reel_url(url)
        and _tiktok_download_error_suggests_photo_post(download_error or "")
    ):
        print("📸 TikTok download unsupported — probing photo slideshow JSON on video page")
        slideshow = _fetch_tiktok_photo_slideshow(url, resolved=resolved)
    if not slideshow and _is_instagram_post_url(url):
        slideshow = _fetch_instagram_carousel_slideshow(url)
    if not slideshow:
        return None
    cache_url = resolved if _is_tiktok_photo_url(resolved) else url
    cache_key = hashlib.sha256(cache_url.encode()).hexdigest() if cache_url != url else url_key
    return _extract_recipe_from_slideshow(cache_url, cache_key, slideshow)


def _try_tiktok_photo_slideshow_fast_path(url: str, url_key: str):
    """
    Skip yt-dlp for confirmed TikTok /photo/ posts — they always fail video download.
    Only runs for TikTok URLs that resolve to /photo/; video/reel URLs are untouched.
    Returns None when this handler does not apply (caller continues video pipeline).
    """
    lowered = (url or "").lower()
    if "tiktok.com" not in lowered and "vt.tiktok.com" not in lowered and "vm.tiktok.com" not in lowered:
        return None
    resolved = _resolve_tiktok_short_url(url)
    is_photo = _is_tiktok_photo_url(url) or _is_tiktok_photo_url(resolved)
    if not is_photo:
        return None
    slideshow = _fetch_tiktok_photo_slideshow(url, resolved=resolved)
    if slideshow:
        print(f"⚡ TikTok photo fast-path: {url} -> {resolved}")
        return _extract_recipe_from_slideshow(resolved, url_key, slideshow)
    return None


def _tiktok_photo_slideshow_error_response():
    return jsonify({
        "error": "Failed to fetch TikTok photo slideshow",
        "user_message": "We couldn't read this photo slideshow. Please try another link or add the recipe manually.",
        "details": "Could not extract images from this TikTok photo post.",
    }), 400


def extract_recipe_from_video_internal(video_url: str):
    """
    Internal helper used by unified /extract-recipe endpoint.
    Parallel audio + low-res video download → transcribe → LLM, or on-demand frame+vision fallback.
    Returns a Flask response: jsonify({...}), status_code
    """
    ok, err = validate_video_url(video_url)
    if not ok:
        return jsonify({"error": err}), 400

    # Resolve share/short links (e.g. fb.watch) to a canonical URL so yt-dlp's
    # generic extractor doesn't loop on the redirect.
    resolved = _resolve_share_url(video_url)
    if resolved != video_url:
        print(f"🔗 Resolved share link {video_url} -> {resolved}")
        video_url = resolved

    # ── Cache check ──────────────────────────────────────────────────────────
    url_key = hashlib.sha256(video_url.encode()).hexdigest()
    if not _get_extract_recipe_no_cache():
        with _recipe_cache_lock:
            if url_key in _recipe_cache:
                print(f"⚡ Cache hit for video URL: {video_url}")
                cached = _apply_response_language(
                    dict(_recipe_cache[url_key]), video_url, _recipe_cache[url_key].get("meta") or {}
                )
                _ensure_cached_recipe_nutrition(cached)
                return jsonify({**_apply_extract_recipe_image_pref(cached), "cached": True}), 200
    # ─────────────────────────────────────────────────────────────────────────

    tiktok_photo_resp = _try_tiktok_photo_slideshow_fast_path(video_url, url_key)
    if tiktok_photo_resp is not None:
        return tiktok_photo_resp

    resolved_tt = _resolve_tiktok_short_url(video_url)
    if _is_tiktok_photo_url(video_url) or _is_tiktok_photo_url(resolved_tt):
        return _tiktok_photo_slideshow_error_response()

    yt_prefetched_meta: dict | None = None
    if is_youtube_url(video_url):
        try:
            info = _yt_meta(video_url)
            yt_prefetched_meta = _video_meta_from_yt_info(info, video_url)
            _assert_youtube_shorts_duration(yt_prefetched_meta.get("duration"))
        except YouTubeVideoTooLongError as e:
            return _youtube_too_long_response(e)
        except Exception as e:
            return jsonify({
                "error": "Video metadata failed",
                "user_message": "We couldn't read this video. Please try another link or add the recipe manually.",
                "details": str(e),
            }), 500

    temp_dir = None
    video_path = None
    meta: dict = {}
    vision_result: dict = {}
    try:
        print(f"🎥 Processing video URL: {video_url}")

        def _run_vision():
            nonlocal temp_dir, video_path, meta
            if vision_result.get("done"):
                return
            if not video_path:
                try:
                    max_sec = None
                    if is_youtube_url(video_url):
                        duration = float(meta.get("duration") or 0)
                        cap = _youtube_vision_max_seconds()
                        if duration > cap:
                            max_sec = cap
                    else:
                        sections, max_sec = _social_vision_download_plan(meta, video_url)
                    print("🎬 Vision fallback: downloading video...")
                    t_vdl = time.time()
                    temp_dir, video_path, dl_meta = _download_video_to_file(
                        video_url,
                        fast=True,
                        max_seconds=max_sec,
                        download_sections=sections,
                    )
                    meta = _merge_video_download_meta(video_url, meta, dl_meta)
                    print(f"🎬 Vision video ready in {time.time() - t_vdl:.2f}s")
                except Exception as e:
                    vision_result["error"] = str(e)
                    vision_result["done"] = True
                    return
            try:
                t_v = time.time()
                recipe, source, method = _run_frame_vision_fallback_from_path(video_path, video_url, meta)
                print(f"🎬 Vision fallback finished in {time.time() - t_v:.2f}s")
                vision_result["recipe"] = recipe
                vision_result["source"] = source
                vision_result["method"] = method
            except Exception as e:
                vision_result["error"] = str(e)
            vision_result["done"] = True

        # IG/TikTok reels: caption from post text first (fast) → vision only if needed
        if _prefer_vision_first_for_video_url(video_url):
            print("⚡ Social reel fast path (caption → vision fallback)")
            t0 = time.time()
            try:
                meta = _fetch_social_reel_meta(video_url)
            except Exception as e:
                return jsonify({
                    "error": "Video metadata failed",
                    "user_message": "We couldn't read this video. Please try another link or add the recipe manually.",
                    "details": str(e),
                }), 500
            print(f"📋 Metadata fetched in {time.time() - t0:.2f}s")

            # Start the thumbnail upload now so it overlaps the caption/vision LLM
            # (resolved in _return_video_extract_result before the response).
            if meta.get("thumbnail"):
                _img_persist_futures[url_key] = _start_persist_image(meta["thumbnail"])

            extraction_method = "caption_llm"
            recipe = None
            source = None
            missing_sections: list[str] = []
            caption_from_partial = False
            caption_text = _video_caption_text(meta)
            if _should_try_social_caption_first(meta, video_url):
                print(f"📝 Caption-first mode ({len(caption_text)} chars)")
                t_cap = time.time()
                recipe = _extract_recipe_from_video_caption(meta)
                print(f"📝 Caption LLM finished in {time.time() - t_cap:.2f}s")
                # Accept whatever the caption provides — ingredients-only or
                # instructions-only included — and flag the missing section
                # instead of falling back to slow vision.
                recipe, missing_sections = _sanitize_partial_caption_recipe(recipe, caption_text)
                if recipe:
                    source = _build_video_recipe_source(video_url, meta, recipe)
                    caption_from_partial = True
                    if missing_sections:
                        print(
                            f"📝 Caption partial — no {', '.join(missing_sections)} in caption; "
                            "returning available info (skipping vision)"
                        )
            else:
                print(
                    f"📝 Caption not used for extraction ({len(caption_text)} chars); "
                    "using vision"
                )

            if not _recipe_has_usable_content(recipe):
                # Caption yielded nothing usable → vision fallback (clear caption flags).
                missing_sections = []
                caption_from_partial = False
                sections, max_sec = _social_vision_download_plan(meta, video_url)
                t_dl = time.time()
                try:
                    temp_dir, video_path, dl_meta = _download_video_to_file(
                        video_url,
                        fast=True,
                        max_seconds=max_sec,
                        download_sections=sections,
                    )
                    meta = _merge_meta_keep_longer_description(meta, dl_meta)
                except ValueError as ve:
                    return jsonify({"error": str(ve)}), 413
                except yt_dlp.utils.DownloadError as de:
                    slideshow_resp = _try_slideshow_fallback_on_download_error(
                        video_url, url_key, download_error=str(de)
                    )
                    if slideshow_resp is not None:
                        return slideshow_resp
                    print(f"[extract-recipe] 400: video download failed for {video_url[:120]}: {str(de)[:200]}")
                    return jsonify({
                        "error": "Failed to download video",
                        "user_message": "We couldn't download this video. Please try another link or add the recipe manually.",
                        "details": str(de),
                        "hint": "For Instagram/TikTok, set YTDLP_COOKIES_FILE or YTDLP_COOKIES_B64.",
                        "cookies_configured": (
                            (YTDLP_COOKIES_FILE and os.path.exists(YTDLP_COOKIES_FILE))
                            or bool(YTDLP_COOKIES_B64)
                        ),
                    }), 400
                except Exception as e:
                    return jsonify({
                        "error": "Video download failed",
                        "user_message": "We couldn't download this video. Please try another link or add the recipe manually.",
                        "details": str(e),
                    }), 500
                print(f"✅ Video downloaded in {time.time() - t_dl:.2f}s (vision-first)")
                _run_vision()
                recipe = vision_result.get("recipe")
                source = vision_result.get("source")
                extraction_method = vision_result.get("method") or "video_frames_vision"

            if recipe and source and _recipe_has_usable_content(recipe):
                vision_partial = False
                if (
                    not caption_from_partial
                    and _is_social_video_url(video_url)
                    and not _social_recipe_is_complete(recipe)
                ):
                    partial_missing = _recipe_missing_sections(recipe)
                    if partial_missing:
                        vision_partial = True
                        missing_sections = partial_missing
                        print(
                            f"🎬 Vision partial — no {', '.join(missing_sections)} in sampled frames; "
                            "returning available info"
                        )
                    else:
                        # Has both sections but below completeness threshold — keep legacy 500.
                        return jsonify({
                            "error": "Failed to extract recipe from video",
                            "user_message": (
                                "We couldn't read the full recipe from this video. "
                                "Please try another link or add the recipe manually."
                            ),
                            "message": "Incomplete recipe extraction (missing ingredients or steps).",
                        }), 500
                lang = _finalize_recipe_for_response(recipe, meta)
                tags = extract_recipe_tags(recipe)
                if caption_from_partial and missing_sections:
                    extraction_method = "caption_llm_partial"
                    recipe["missing"] = missing_sections
                elif vision_partial:
                    extraction_method = "video_frames_vision_partial"
                    if missing_sections:
                        recipe["missing"] = missing_sections
                _result = {
                    "source": source,
                    "recipe": recipe,
                    "tags": tags,
                    "transcript": None,
                    "extraction": {
                        "method": extraction_method,
                        "confidence": 0.45 if vision_partial else 0.5,
                    },
                    "meta": meta,
                }
                if caption_from_partial and missing_sections:
                    _result["warnings"] = [
                        f"No {section} found in the caption." for section in missing_sections
                    ]
                elif vision_partial:
                    if missing_sections:
                        _result["warnings"] = [
                            f"No {section} found in the video." for section in missing_sections
                        ]
                    else:
                        _result["warnings"] = ["Recipe extraction may be incomplete."]
                return _return_video_extract_result(_result, url_key, video_url, meta, language=lang)
            return jsonify({
                "error": "Failed to extract recipe from video",
                "user_message": "We couldn't extract a recipe from this link. Please try another or add the recipe manually.",
                "message": "Frame+vision extraction failed.",
            }), 500

        transcript_text = ""
        transcript_extraction_method = "transcript_llm"

        if is_youtube_url(video_url):
            print("⚡ YouTube fast path (captions → bounded Whisper)")
            t_yt = time.time()
            try:
                transcript_text, meta = _fetch_youtube_transcript(
                    video_url, meta=yt_prefetched_meta
                )
            except YouTubeVideoTooLongError as e:
                return _youtube_too_long_response(e)
            except ValueError as ve:
                return jsonify({"error": str(ve)}), 413
            except Exception as e:
                return jsonify({
                    "error": "YouTube transcript failed",
                    "user_message": "We couldn't read this video. Please try another link or add the recipe manually.",
                    "details": str(e),
                }), 500
            if meta.get("transcript_source") == "youtube_captions":
                transcript_extraction_method = "youtube_captions_llm"
            transcript_text = _trim_youtube_transcript_for_llm(transcript_text)
            print(f"✅ YouTube transcript path finished in {time.time() - t_yt:.2f}s")
        else:
            # 1) Parallel: low-res video (for vision fallback) + audio-only (for fast transcription)
            t0 = time.time()
            audio_dl: dict = {}
            video_dl: dict = {}

            def _dl_audio():
                try:
                    audio_dl["bytes"], audio_dl["meta"] = download_audio_mp3(video_url)
                except Exception as e:
                    audio_dl["error"] = e

            def _dl_video():
                try:
                    video_dl["temp_dir"], video_dl["path"], video_dl["meta"] = _download_video_to_file(
                        video_url, fast=True
                    )
                except Exception as e:
                    video_dl["error"] = e

            def _transcribe_when_audio_ready():
                nonlocal transcript_text
                _dl_audio()
                try:
                    audio_bytes = audio_dl.get("bytes")
                    if audio_bytes:
                        transcript_text = _transcribe_audio_bytes(audio_bytes)
                except Exception as e:
                    print(f"⚠️ Transcription failed: {e}")

            t1 = time.time()
            with ThreadPoolExecutor(max_workers=3) as executor:
                fv = executor.submit(_dl_video)
                ft = executor.submit(_transcribe_when_audio_ready)
                fv.result()
                ft.result()
            print(f"🎤 Transcription finished in {time.time() - t1:.2f}s")

            if audio_dl.get("error") and video_dl.get("error"):
                de = video_dl["error"]
                if isinstance(de, yt_dlp.utils.DownloadError):
                    slideshow_resp = _try_slideshow_fallback_on_download_error(
                        video_url, url_key, download_error=str(de)
                    )
                    if slideshow_resp is not None:
                        return slideshow_resp
                    error_str = str(de)
                    lowered_url = video_url.lower()
                    is_instagram = "instagram.com" in lowered_url
                    is_facebook = ("facebook.com" in lowered_url) or ("fb.watch" in lowered_url) or ("fb.com" in lowered_url)
                    cookies_configured = (YTDLP_COOKIES_FILE and os.path.exists(YTDLP_COOKIES_FILE)) or YTDLP_COOKIES_B64
                    if is_instagram:
                        if not cookies_configured:
                            hint = "Instagram requires authentication. Please set YTDLP_COOKIES_FILE or YTDLP_COOKIES_B64."
                        elif "empty media response" in error_str.lower() or "unavailable" in error_str.lower():
                            hint = "Instagram post may be private or cookies may be expired."
                        else:
                            hint = "Instagram extraction failed. The post may be private or require fresh cookies."
                    elif is_facebook:
                        if not cookies_configured:
                            hint = "Facebook often requires authentication. Please set YTDLP_COOKIES_FILE or YTDLP_COOKIES_B64."
                        elif "private" in error_str.lower() or "unavailable" in error_str.lower() or "login" in error_str.lower():
                            hint = "Facebook post may be private/restricted or cookies may be expired."
                        else:
                            hint = "Facebook extraction failed. The post may be private or require fresh cookies."
                    else:
                        hint = "For TikTok/Instagram/Facebook (and some YouTube), set YTDLP_COOKIES_FILE."
                    return jsonify({
                        "error": "Failed to download video",
                        "details": error_str,
                        "hint": hint,
                        "cookies_configured": cookies_configured,
                    }), 400
                if isinstance(de, ValueError):
                    return jsonify({"error": str(de)}), 413
                return jsonify({
                    "error": "Video download failed",
                    "user_message": "We couldn't download this video. Please try another link or add the recipe manually.",
                    "details": str(de),
                }), 500

            temp_dir = video_dl.get("temp_dir")
            video_path = video_dl.get("path")
            meta = _merge_video_download_meta(video_url, video_dl.get("meta"), audio_dl.get("meta"))
            print(f"✅ Downloads finished in {time.time() - t0:.2f}s (audio={'ok' if audio_dl.get('bytes') else 'skip'}, video={'ok' if video_path else 'skip'})")

            # 2) Fallback: extract audio from video file if audio-only download failed
            if not transcript_text and video_path:
                try:
                    audio_bytes = _extract_audio_from_video_file(video_path)
                    if audio_bytes:
                        transcript_text = _transcribe_audio_bytes(audio_bytes)
                except Exception as e:
                    print(f"⚠️ Transcription from video file failed: {e}")

        has_usable_transcript = transcript_text and not _is_likely_non_speech(transcript_text)

        # 3) Vision only when transcript is not usable (no upfront parallel vision)
        if not has_usable_transcript:
            print("🎬 No usable transcript; running frame+vision fallback...")
            _run_vision()
            recipe = vision_result.get("recipe")
            source = vision_result.get("source")
            extraction_method = vision_result.get("method")
            if recipe and source:
                lang = _finalize_recipe_for_response(recipe, meta)
                tags = extract_recipe_tags(recipe)
                _result = {"source": source, "recipe": recipe, "tags": tags, "transcript": None, "extraction": {"method": extraction_method, "confidence": 0.5}, "meta": meta}
                return _return_video_extract_result(_result, url_key, video_url, meta, language=lang)
            return jsonify({
                "error": "Failed to extract recipe from video",
                "user_message": "We couldn't extract a recipe from this link. Please try another or add the recipe manually.",
                "message": "No usable transcript and frame+vision fallback failed.",
            }), 500

        # 4) LLM extraction from transcript (chunking + merge)
        # 80 000 chars ≈ 20 000 tokens — fits in gpt-4o-mini's 128k window, eliminating the merge step for most videos
        print("🍳 Extracting recipe information from transcript...")
        chunks = _chunk_text(transcript_text, max_chars=80000)
        if not chunks:
            print("🎬 No chunks from transcript; running frame+vision fallback...")
            _run_vision()
            recipe = vision_result.get("recipe")
            source = vision_result.get("source")
            extraction_method = vision_result.get("method")
            if recipe and source:
                lang = _finalize_recipe_for_response(recipe, meta)
                tags = extract_recipe_tags(recipe)
                _result = {"source": source, "recipe": recipe, "tags": tags, "transcript": None, "extraction": {"method": extraction_method, "confidence": 0.5}, "meta": meta}
                return _return_video_extract_result(_result, url_key, video_url, meta, language=lang)
            return jsonify({
                "error": "Transcript is empty after cleanup",
                "user_message": "We couldn't extract a recipe from this link. Please try another or add the recipe manually.",
            }), 500

        # Extract recipe from transcript chunks in parallel for faster response
        parts = [None] * len(chunks)
        chunk_failed = {"idx": None, "error": None}
        max_workers = min(4, len(chunks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_extract_recipe_chunk, ch): i for i, ch in enumerate(chunks)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    parts[idx] = future.result()
                except Exception as e:
                    chunk_failed["idx"] = idx + 1
                    chunk_failed["error"] = e
                    break
        if chunk_failed["idx"] is not None:
            print(f"⚠️ Transcript chunk extraction failed (chunk {chunk_failed['idx']}); running frame+vision fallback...")
            _run_vision()
            recipe = vision_result.get("recipe")
            source = vision_result.get("source")
            extraction_method = vision_result.get("method")
            if recipe and source:
                lang = _finalize_recipe_for_response(recipe, meta)
                tags = extract_recipe_tags(recipe)
                _result = {"source": source, "recipe": recipe, "tags": tags, "transcript": None, "extraction": {"method": extraction_method, "confidence": 0.5}, "meta": meta}
                return _return_video_extract_result(_result, url_key, video_url, meta, language=lang)
            return jsonify({
                "error": "Failed to extract recipe from transcript chunk",
                "user_message": "We couldn't extract a recipe from this link. Please try another or add the recipe manually.",
                "chunk": chunk_failed["idx"],
                "details": str(chunk_failed["error"]),
                "transcript": transcript_text
            }), 500
        parts = [p for p in parts if p is not None]
        if not parts:
            return jsonify({
                "error": "No recipe parts extracted from transcript",
                "user_message": "We couldn't extract a recipe from this link. Please try another or add the recipe manually.",
            }), 500

        recipe = _merge_recipe_parts(parts) if len(parts) > 1 else parts[0]
        ingredients = recipe.get("ingredients") or []
        instructions = recipe.get("instructions") or []
        if ingredients and isinstance(ingredients[0], str):
            ingredients = [{"name": ing, "quantity": ""} for ing in ingredients]
        if not isinstance(instructions, list):
            instructions = [str(instructions)]
        recipe["ingredients"] = ingredients
        recipe["instructions"] = instructions

        # If transcript yielded empty instructions/ingredients, use already-computed vision result
        if not instructions:
            print("🎬 Instructions empty from transcript; running frame+vision fallback...")
            _run_vision()
            fallback_recipe = vision_result.get("recipe")
            fallback_source = vision_result.get("source")
            extraction_method = vision_result.get("method")
            if fallback_recipe and fallback_source and (
                fallback_recipe.get("instructions") or fallback_recipe.get("ingredients")
            ):
                lang = _finalize_recipe_for_response(fallback_recipe, meta)
                tags = extract_recipe_tags(fallback_recipe)
                _result = {"source": fallback_source, "recipe": fallback_recipe, "tags": tags, "transcript": None, "extraction": {"method": extraction_method, "confidence": 0.5}, "meta": meta}
                return _return_video_extract_result(_result, url_key, video_url, meta, language=lang)
        elif not ingredients:
            print("🎬 Ingredients empty from transcript; running frame+vision fallback...")
            _run_vision()
            fallback_recipe = vision_result.get("recipe")
            fallback_source = vision_result.get("source")
            extraction_method = vision_result.get("method")
            if fallback_recipe and fallback_source:
                lang = _finalize_recipe_for_response(fallback_recipe, meta)
                tags = extract_recipe_tags(fallback_recipe)
                _result = {"source": fallback_source, "recipe": fallback_recipe, "tags": tags, "transcript": None, "extraction": {"method": extraction_method, "confidence": 0.5}, "meta": meta}
                return _return_video_extract_result(_result, url_key, video_url, meta, language=lang)

        source_type = determine_source_type(video_url)
        lang = _finalize_recipe_for_response(recipe, meta)
        tags = extract_recipe_tags(recipe)
        source = {
            "type": "video",
            "url": video_url,
            "provider": meta.get("provider", ""),
            "title": meta.get("title", "") or recipe.get("name", ""),
            "image": meta.get("thumbnail"),
            "source_type": source_type,
        }
        _result = {
            "source": source,
            "recipe": recipe,
            "tags": tags,
            "transcript": transcript_text,
            "extraction": {"method": transcript_extraction_method, "confidence": 0.55},
            "meta": meta,
        }
        return _return_video_extract_result(_result, url_key, video_url, meta, language=lang)
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)



    ## video and webpage code ends here#####
   

if __name__ == "__main__":
    # Disable Flask's built-in debugger when running in VS Code debugger
    # Set FLASK_DEBUG environment variable to enable Flask debug mode separately
    import os
    flask_debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=flask_debug, host='0.0.0.0', port=5003, use_reloader=False)