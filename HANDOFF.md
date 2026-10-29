# RecipeVault extractor API — handoff

Backend for RecipeVault/MealMap recipe import. Repo `unkitaagarwal/NUTRITION-RAG-CHATBOT`,
branch **`harsh_changes_oct_26`**, base `a3da10b`. Deployed on Render as
`nutrition-rag-chatbot` (srv-d2qd9bn5r7bs73ailt20), gunicorn `workers=2, threads=8`.
Single file: `app.py`, ~11k lines. Python **3.12** (3.14 cannot install the pinned deps).

---

## 1. State of the branch

Four commits on top of `a3da10b`:

| commit | contents |
|---|---|
| `e2f13b1` | clipping fix, share-link resolution, YouTube clients, anti-bot headers, test harness |
| `01e0360` | **two NameError boot crashes**, bare `/share/<id>/` form, pin.it homepage guard |
| `5957434` | `VIDEO_CLIP_MARGIN` |
| `0e4b97c` | `batch_tests/AB_RESULTS.md` |

**Check `git log origin/harsh_changes_oct_26..HEAD` first.** As of 2026-10-29 the last
three were unpushed, and what sits on origin (`e2f13b1` alone) **does not boot** — it
contains both NameErrors. If origin is still at `e2f13b1`, push before anything else.

---

## 2. What was changed and why

### 2.1 Video clipping — the biggest fix (`_make_download_ranges`, app.py:119)

**Pre-existing bug:** the code set `ydl_opts["download_sections"]`. That is a yt-dlp
**CLI flag**; the Python API never reads it. All four existing assignments were dead
code — videos always downloaded in full.

The real API parameter is `download_ranges`, a **callable** run after metadata:

```python
def _ranges(info_dict, ydl=None):
    if explicit: return explicit
    duration = (info_dict or {}).get("duration") or 0
    window = ...
    if window and duration and duration > window * VIDEO_CLIP_MARGIN:
        return [{"start_time": 0.0, "end_time": window}]
    return [{}]          # whole video — identical to yt-dlp's default
```

**Critical:** returning `[]` makes yt-dlp **skip the download entirely**. `[{}]` means
whole video. Never return an empty list.

Config (app.py:93-100):
- `MAX_VIDEO_SECONDS` 900 → **300** (clip window)
- `VIDEO_CLIP_OVER_CAP=1` — over-cap videos are clipped, not rejected with 413
- `MAX_VIDEO_SECONDS_HARD=7200` — the outright-reject bound
- `VIDEO_CLIP_MARGIN=1.2` — only clip above 360s. A 322s video measured *bigger and
  slower* clipped (2.2MB/2s whole vs 3.9MB/5s clipped) because of the keyframe re-cut.
- `_youtube_shorts_max_seconds()` follows the same policy instead of a hardcoded 180s.

Wired in at app.py:3351 (`ytdlp_base_opts`) and app.py:7849 (`_download_video_to_file`),
both with `force_keyframes_at_cuts: True`.
`_download_single_profile_video` (profile-video tool, not recipe import) is deliberately
untouched.

Verified with real downloads: 635s → 300.0s file; 19s → 19.0s file (byte-identical path).

### 2.2 Facebook / share links (app.py:332-470)

**Bug:** 70 of 200 Facebook errors were `Unsupported URL: https://www.facebook.com/login/`.
The guard `if "facebook.com" not in fhost` **accepts** the login wall, which then reached
the extractor as if it were the post.

- `_is_dead_end_url()` (app.py:380) rejects `/login`, `/checkpoint`, `/consent`,
  `/privacy`, `/recover`, `/help`, `/policies`, `/terms`, `/unsupportedbrowser`,
  `/device-based`. Verified it does **not** fire on `/reel/` or `/loginhelper/`.
- `_SHARE_PATH_RE = re.compile(r"^/share(?:/|$)", re.I)` — FB's share wrapper is a
  **path**, not a host, so a host-only check never matched. Covers both
  `/share/r/<id>/` and the bare `/share/<id>/` (7 of 135 observed links).
- `_SHARE_LINK_HOSTS` 4 → 12: added `l.facebook.com`, `lm.facebook.com`, `pin.it`,
  `share.google`, `apple.news`.
- Homepage guard: an expired `pin.it` redirects to `pinterest.com`; returning that would
  make the extractor scrape the homepage and "succeed" with nothing. Original is kept.
- `_TRACKING_PARAMS` strips utm/fbclid/igsh etc. **`ref` and `refsrc` are deliberately
  NOT stripped — they are load-bearing on Facebook.**

**Unverified in production.** In the A/B, 0 of 138 Facebook URLs worked on either
version because the test box had no FB cookies. Logic is unit-tested; end-to-end is not.

### 2.3 YouTube

`android_vr` was removed upstream 2026-10-20, degrading the client list.
`YT_PLAYER_CLIENTS` (app.py:594) is now env-overridable, default
`default,ios,tv,web_safari`. `ios` needs no PO token and clears the bot challenge —
confirmed by downloading a video with no cookies and no proxy.

`YT_USE_PROXY=1` was saved in Render but **not deployed**. Until a deploy happens, every
YouTube request logs `Direct attempt blocked ... Sign in to confirm you're not a bot;
retrying via proxy`. After deploy, that line should disappear — use it as the success metric.

### 2.4 Blogs / anti-bot (app.py:4163-4190)

- `_BROWSER_HEADERS`: added `Referer: https://www.google.com/`, `Sec-Fetch-*`,
  `Upgrade-Insecure-Requests`.
- `_BLOCK_STATUSES`: added 401, 402, 451, 520-526 so Cloudflare-class responses escalate
  to the next fetch tier instead of being parsed as page content.
- `fetch_html` ladder is: direct (curl_cffi Chrome impersonation) → residential proxy
  (Decodo) → ScraperAPI (no-render, then render + optional `ultra_premium`).

### 2.5 Startup diagnostics

Two banners now print which tiers are actually armed:

```
[startup] proxies: YT_PROXY=set (PROXY-FIRST via YT_USE_PROXY=1), SOCIAL_PROXY=..., clients=default,ios,tv,web_safari, proxy_blackout=120s
[startup] fetch_html tiers: direct=on  proxy=ON  scraperapi=ON
```

Read these first when debugging — most "code" problems were unset env vars.

---

## 3. Gotchas that cost real time

1. **`py_compile` does not catch NameError.** Two startup banners referenced names
   defined 80-139 lines later and killed the process at import. **Always boot the app,
   never just compile it.**
2. **`download_sections` vs `download_ranges`** — see 2.1. CLI flag vs API parameter.
3. `[]` from the ranges callable = download skipped entirely.
4. Python 3.14 cannot install the pinned deps (`numpy==2.3.2` needs ≤3.13). Use the
   3.12 venv. `ModuleNotFoundError: flask` almost always means system python.
5. Clipping can make a file **larger** — hence `VIDEO_CLIP_MARGIN`.
6. `_is_dead_end_url(parsed)` takes a **parsed urlparse object**, not a string.

---

## 4. Running it

```bash
source venv/bin/activate      # Python 3.12
python app.py                 # dev server, port 5003
# or: gunicorn -c gunicorn.conf.py app:app
```

`.env` is gitignored and stubbed with the vars that matter. Fill from Render →
Environment. Minimum `OPENAI_API_KEY`; add `YTDLP_COOKIES_B64`, `SOCIAL_PROXY`,
`SCRAPER_API_KEY` or TikTok/Instagram (≈77% of real traffic) will fail for reasons
unrelated to the code.

Note: website extraction via JSON-LD works with **no** OpenAI key, so blog URLs are
testable keyless.

---

## 5. Testing

```bash
cd batch_tests
python3 pull_recent_imports.py --days 2     # needs Firestore creds; auto-finds
                                            # ../MealMap/scripts/serviceAccountKey.json
python3 ab_compare.py --baseline <A> --candidate <B> --workers 4
```

- `pull_recent_imports.py` pulls **all** import attempts (not just failures) with their
  production outcome — the regression net.
- `ab_compare.py` is stdlib-only, resumable via `.jsonl`, sends `no_cache` on both sides
  and alternates call order per URL so neither side gets a warm-CDN edge.
- **Gate: `REGRESSIONS (worked -> broken)` must be 0.**

**Do not A/B localhost against Render** — a home IP passes direct fetches a datacenter IP
gets blocked on, so you measure the network. Either deploy the branch as a second Render
service, or compare **old code vs new code on the same box** (that is what
`AB_RESULTS.md` did: 0 regressions across 228 URLs).

Last production baseline: **92.6% success, 463 of 500** over 2 days
(tiktok 273, instagram 113, website 106, youtube 8).

---

## 6. Open items

**Backend**
- Push the branch if origin is behind (see §1).
- **Rotate `SCRAPER_API_KEY`** — it was exposed in a screenshot and is still live.
- Set `SCRAPER_API_ULTRA=true` — stage 2 sends `render=true` but never `ultra_premium`;
  likely fix for the remaining ~131 blog 403s.
- Deploy to pick up `YT_USE_PROXY=1`.
- Consider `PROXY_DISABLE_SECONDS=120` (prod default 600). Decodo threw 4
  `PROXY CONFIG ERROR` (`gate.decodo.com:10001`, tunnel 522/502) in 7 days, each
  disabling proxy fallback for 600s.
- **Biggest remaining win: audio in `_gapfill_vision_worker`.** 444 of 1229 broken
  imports were "no steps in sampled frames" — the steps are spoken, not on screen.
- `_gapfill_jobs` and `_PROXY_DISABLED_UNTIL` are **per-process globals** with 2 gunicorn
  workers, so the workers cannot see each other's state.

**Client (separate repo, MealMap)**
- Server sends `retry_after_sec: 10` (app.py:10582) and **neither client reads it** —
  591 of 1229 broken imports were gap-fill timeouts, recoverable with no extraction work.
- `stepsGenerated` is in the response and unread, so LLM-written steps are invisible.
- `clipped_to_seconds` is unread.
- Share-extension timeout is 45s (`ShareViewController.swift:620`); 117 of 118
  "network dropped" errors hit a 46s ceiling.

---

## 7. Corrections to earlier claims (do not repeat them)

- `share.google` and `pin.it` resolution was called "a clean win". Tested against old
  code, **both already worked**. Not a regression, not a win.
- The 413s were **not** payload size — they were `Video too long (1277s). Max allowed is 900s`.
- "Network dropped" was **not** connectivity — it was the 45s client timeout.
- "No redirect resolution anywhere" was wrong — `_resolve_share_url` existed, scoped to 4 hosts.
