# Gunicorn config for the nutrition / extract-recipe API (Flask WSGI app).
#
#   Run:  gunicorn -c gunicorn.conf.py app:app
#
# Why threaded workers: the request pipeline is I/O-bound (yt-dlp downloads,
# Whisper, OpenAI vision/LLM calls, Firebase Storage). Threads let one worker
# absorb many concurrent blocking calls. We keep the number of *processes* low
# so the in-process _recipe_cache stays effective within each worker, and let
# threads handle concurrency.
import os

bind = f"0.0.0.0:{os.getenv('PORT', '5003')}"

worker_class = "gthread"
workers = int(os.getenv("GUNICORN_WORKERS", "2"))
threads = int(os.getenv("GUNICORN_THREADS", "8"))

# A single extraction can take 10-20s (download + Whisper + vision). Don't let
# gunicorn kill long-but-legitimate requests. Generous, but bounded.
timeout = int(os.getenv("GUNICORN_TIMEOUT", "180"))
graceful_timeout = 30
keepalive = 15

# Recycle workers periodically to bound any slow memory growth (yt-dlp/ffmpeg).
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "500"))
max_requests_jitter = 50

# IMPORTANT: do NOT enable preload_app. Firebase/gRPC clients are initialized at
# import time and do not survive fork; each worker must initialize its own.
preload_app = False

accesslog = "-"
errorlog = "-"
loglevel = os.getenv("GUNICORN_LOGLEVEL", "info")


def on_starting(server):
    """Purge the download temp root on fresh boot.

    Workers killed mid-request (timeout SIGKILL, max_requests recycle past
    graceful_timeout, OOM) never run their `finally` cleanup, and /tmp persists
    across worker restarts — orphaned video downloads accumulate until the
    platform's 2GB /tmp limit kills the instance. At master start nothing can
    be mid-download, so wiping the whole root is safe. app.py's tmp janitor
    handles dirs orphaned later, while the master keeps running.
    """
    import shutil
    import tempfile

    root = os.getenv(
        "DOWNLOAD_TMP_ROOT", os.path.join(tempfile.gettempdir(), "mealmap-dl")
    )
    shutil.rmtree(root, ignore_errors=True)
    server.log.info("Purged download temp root: %s", root)
