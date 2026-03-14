# `/extract-recipe` API — Tech Stack & Flow

## Tech Stack

| Layer | Technology |
|--------|------------|
| **API** | Flask (Python) |
| **Recipe LLM (text/images)** | OpenAI API — `gpt-4o-mini` (vision for images; chat for transcript/webpage) |
| **Transcription** | OpenAI Whisper (`whisper-1`) |
| **Video download** | yt-dlp (YouTube, TikTok, Instagram, etc.) |
| **Audio extraction from video** | ffmpeg (libmp3lame) |
| **Frame extraction from video** | ffmpeg (JPEG frames at timestamps) |
| **Webpage fetch** | `requests` + `BeautifulSoup` (lxml) |
| **Structured data (webpage)** | JSON-LD parsing from HTML, with LLM fallback on cleaned body text |
| **Tags** | In-app logic (Vegetarian, Vegan, High Protein, Quick/Easy/Medium/Hard, etc.) |
| **Validation** | SSRF-safe URL validation (hostname/IP checks) |

**Key env vars:** `OPENAI_API_KEY`, `RECIPE_LLM_MODEL` (default `gpt-4o-mini`), `RECIPE_LLM_TIMEOUT` (default `120` s), `MAX_VIDEO_SECONDS`, `NUM_VIDEO_FRAMES_RECIPE` (default `5`; fewer = faster), `YTDLP_COOKIES_FILE` / `YTDLP_COOKIES_B64`, `YT_PROXY` (YouTube only).

**System dependency:** `ffmpeg` (for video→audio and video→frames).

---

## API Contract

- **Method:** `POST`
- **Route:** `/extract-recipe`
- **Input (one of):**
  - **Images:** multipart `image` (one or many files) **or** JSON `imageBase64` / `images[]` (+ optional `imageFormat`).
  - **URL:** JSON `url` or `videoUrl` or `recipeUrl` (+ optional `mode`: `"auto"` | `"video"` | `"webpage"`).
- **Output (success):** JSON with:
  - `source` — `type`, `url`, `provider`, `title`, `image`, `source_type`
  - `recipe` — `name`, `ingredients[]`, `instructions[]`, `servings`, `prep_time`, `cook_time`, `total_time`, `notes[]`, `meal_type`, `cuisine`, `diet_flags`
  - `meal_type` — one of: Breakfast, Lunch, Dinner, Snack
  - `cuisine` — one of: Italian, Mexican, American, Asian, Mediterranean, Indian, Chinese, Japanese, Thai, Middle Eastern
  - `diet_flags` — subset of: High Protein, High Fiber, Low Carb, Keto, Vegetarian, Vegan, Gluten Free, Balanced
  - `tags` — e.g. High Protein, Vegetarian, Vegan, Quick, Easy
  - `transcript` — only for video when transcript path is used; else `null`
  - `extraction` — `method`, `confidence`
  - `meta` — (video only) duration, title, thumbnail, etc.

---

## High-Level Flow

```
                    POST /extract-recipe
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        [Images?]        [URL?]          [Neither]
              │               │               │
              ▼               │               └──► 400 "url or image required"
    extract_recipe_from_      │
    images_llm (vision)       ▼
              │         [Video URL?] ──► extract_recipe_from_video_internal
              │               │
              │               └──► [Webpage URL] ──► fetch HTML
              │                         │              ├─► JSON-LD recipe → normalize
              │                         │              └─► else clean text → webpage LLM
              ▼                         ▼
    source + recipe + tags      source + recipe + tags
    (method: image_vision)       (method: jsonld | html_llm | transcript_llm | video_frames_vision)
```

---

## Flow 1: Image(s)

1. Parse request: multipart `image` (list) or JSON `imageBase64` / `images[]`.
2. Build list of image data URLs (base64 JPEG/PNG/etc.).
3. **OpenAI vision** (`extract_recipe_from_images_llm`): one or more images → single recipe JSON (name, ingredients, instructions, times, notes). Multi-image = “merge into one recipe.”
4. `extract_recipe_tags(recipe)`.
5. Return `source` (type `"image"`), `recipe`, `tags`, `transcript: null`, `extraction: { method: "image_vision" }`.

---

## Flow 2: Webpage URL

1. Validate URL (SSRF).
2. `fetch_html(url)` → raw HTML.
3. `extract_jsonld_recipes(html)` (BeautifulSoup + JSON-LD in `<script type="application/ld+json">`).
4. **If JSON-LD recipe found:** `normalize_recipe_from_jsonld` + OG image; `method = "jsonld"`.
5. **Else:** `clean_page_text(html)` → `extract_recipe_from_webpage_llm(page_text)` (OpenAI chat, JSON output); `method = "html_llm"`; image from `extract_og_image(soup)`.
6. `determine_source_type(url)`, `extract_recipe_tags(recipe)`.
7. Return `source`, `recipe`, `tags`, `transcript: null`, `extraction: { method, confidence }`.

---

## Flow 3: Video URL (single download, then audio or frame+vision)

**Principle:** Download the **video file once**; use it for both audio extraction and, when needed, frame+vision fallback.

1. **Validate** video URL (SSRF).
2. **Download video once:** `_download_video_to_file(video_url)` → `temp_dir`, `video_path`, `meta` (yt-dlp; cookies/proxy as configured). Temp dir is cleared in a `finally` at the end.
3. **Extract audio from file:** `_extract_audio_from_video_file(video_path)` (ffmpeg → MP3 bytes).
   - **If no audio:** run **frame+vision fallback** (see below) using `video_path`; return or 500.
4. **Whisper:** `client.audio.transcriptions.create(whisper-1, audio_bytes)` → `transcript_text`.
5. **If no usable transcript** (empty or `_is_likely_non_speech`): run **frame+vision fallback** from `video_path`; return or 500.
6. **Transcript path:**
   - `_chunk_text(transcript_text, 6000)`.
   - If no chunks → frame+vision fallback from `video_path`; return or 500.
   - For each chunk: `_extract_recipe_chunk(chunk)` (OpenAI chat, JSON). On exception → frame+vision fallback from `video_path`; return or 500.
   - `_merge_recipe_parts(parts)` → one recipe; normalize `ingredients` / `instructions`.
7. **If transcript recipe has empty `instructions` (or empty `ingredients`):** run **frame+vision fallback** from `video_path`; if it returns a better recipe, return it (`method: "video_frames_vision"`), else keep transcript recipe.
8. **Else:** return transcript-based recipe with `transcript`, `extraction: { method: "transcript_llm" }`.
9. **Finally:** `shutil.rmtree(temp_dir)`.

**Frame+vision fallback (reuses existing `video_path`):**

- `_run_frame_vision_fallback_from_path(video_path, video_url, meta)`:
  - `_extract_frame_data_urls_from_video(video_path, duration, num_frames=8)` — ffmpeg samples evenly spaced JPEG frames → list of data URLs.
  - `extract_recipe_from_images_llm(frame_urls)` — same OpenAI vision call as image flow; one recipe from multiple frames.
  - Normalize recipe; build `source`; return `(recipe, source, "video_frames_vision")`.

No second video download: the same file is used for audio and for frames.

---

## Extraction Methods (response field)

| `extraction.method` | When |
|----------------------|------|
| `image_vision` | Recipe from uploaded/pasted image(s) via OpenAI vision |
| `jsonld` | Recipe from webpage JSON-LD |
| `html_llm` | Recipe from webpage body text via LLM |
| `transcript_llm` | Recipe from video transcript (Whisper + chunked LLM) |
| `video_frames_vision` | Recipe from video when transcript path not used or yielded empty instructions; frames + OpenAI vision |

---

## Performance (response time)

- **Video frame extraction:** Frames are extracted in **parallel** (ThreadPoolExecutor, up to 4 workers) so multiple ffmpeg runs don’t block each other.
- **Video transcript:** Recipe extraction from transcript chunks runs in **parallel** (one LLM call per chunk, up to 4 concurrent).
- **Default video frames:** Default is **5** frames (`NUM_VIDEO_FRAMES_RECIPE`); set to 8 for more accuracy at the cost of a bit more time.
- **Timeouts:** All recipe-related OpenAI calls use `RECIPE_LLM_TIMEOUT` (default 120 s) so slow API responses don’t hang the request indefinitely.

## Summary

- **Images:** OpenAI vision only; single or multiple images → one recipe.
- **Webpage:** Fetch HTML → JSON-LD or cleaned text → recipe (+ tags/source).
- **Video:** One yt-dlp download → ffmpeg audio → Whisper; if transcript is usable, LLM recipe from transcript (chunks in parallel); if not (no/music-only transcript, empty instructions, or chunk failure), reuse same video file for ffmpeg frames (extracted in parallel) + OpenAI vision to get the recipe.
