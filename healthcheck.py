"""
healthcheck.py — calls /recipeVaultHealth at 5 AM and 5 PM EST every day.
Sends a WhatsApp message via CallMeBot on both success AND failure.

Setup:
  1. Activate CallMeBot (one-time):
       - Save +34 644 59 77 59 as a WhatsApp contact
       - Send: "I allow callmebot to send me messages"
       - You'll receive your APIKEY in the reply

  2. Fill in .env  (copy from .env.example)

  3. Install dependencies:
       pip install apscheduler requests python-dotenv

  4. Run:
       python healthcheck.py                           # foreground
       nohup python healthcheck.py > healthcheck.log 2>&1 &   # background on a server
"""

import os
import logging
import requests
from datetime import datetime
from urllib.parse import quote
from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
HEALTH_URL      = os.getenv(
    "HEALTH_URL",
    "https://nutrition-rag-chatbot.onrender.com/recipeVaultHealth",
)
REQUEST_TIMEOUT = int(os.getenv("HEALTH_TIMEOUT_SECS", "30"))  # recipe fetch can take ~5s

WHATSAPP_PHONE  = os.getenv("WHATSAPP_PHONE")   # country code + number, no + or spaces
WHATSAPP_APIKEY = os.getenv("WHATSAPP_APIKEY")  # key sent by CallMeBot

TIMEZONE = "America/New_York"   # handles EST (UTC-5) and EDT (UTC-4) automatically

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("healthcheck")


# ── WhatsApp ─────────────────────────────────────────────────────────────────
def send_whatsapp(message: str):
    if not WHATSAPP_PHONE or not WHATSAPP_APIKEY:
        log.warning("WhatsApp not configured — set WHATSAPP_PHONE and WHATSAPP_APIKEY in .env")
        return
    url = (
        f"https://api.callmebot.com/whatsapp.php"
        f"?phone={WHATSAPP_PHONE}"
        f"&apikey={WHATSAPP_APIKEY}"
        f"&text={quote(message)}"
    )
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            log.info("WhatsApp sent to %s", WHATSAPP_PHONE)
        else:
            log.error("CallMeBot HTTP %d: %s", r.status_code, r.text[:200])
    except Exception as e:
        log.error("WhatsApp send failed: %s", e)


# ── Health check ──────────────────────────────────────────────────────────────
def run_check():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    log.info("Running health check → %s", HEALTH_URL)

    try:
        r = requests.get(HEALTH_URL, timeout=REQUEST_TIMEOUT)
        data = {}
        try:
            data = r.json()
        except Exception:
            pass

        if r.status_code == 200:
            recipe_name = data.get("recipe_name", "unknown")
            method      = data.get("method", "unknown")
            elapsed     = data.get("elapsed_s", "?")
            log.info("✅ Healthy — %s (method: %s, %.2fs)", recipe_name, method, float(elapsed) if elapsed != "?" else 0)

            send_whatsapp(
                f"✅ RecipeVault Health Check PASSED\n"
                f"Time   : {now} EST\n"
                f"Recipe : {recipe_name}\n"
                f"Method : {method}\n"
                f"Elapsed: {elapsed}s\n"
                f"URL    : {HEALTH_URL}"
            )

        else:
            error = data.get("error", r.text[:200])
            elapsed = data.get("elapsed_s", "?")
            log.warning("❌ Unhealthy — HTTP %d: %s", r.status_code, error)

            send_whatsapp(
                f"🚨 RecipeVault Health Check FAILED\n"
                f"Time   : {now} EST\n"
                f"Status : HTTP {r.status_code}\n"
                f"Error  : {error}\n"
                f"Elapsed: {elapsed}s\n"
                f"URL    : {HEALTH_URL}"
            )

    except requests.exceptions.Timeout:
        log.warning("❌ Timed out after %ds", REQUEST_TIMEOUT)
        send_whatsapp(
            f"🚨 RecipeVault Health Check FAILED\n"
            f"Time  : {now} EST\n"
            f"Error : Timed out after {REQUEST_TIMEOUT}s\n"
            f"URL   : {HEALTH_URL}"
        )

    except requests.exceptions.ConnectionError:
        log.warning("❌ Connection refused / server unreachable")
        send_whatsapp(
            f"🚨 RecipeVault Health Check FAILED\n"
            f"Time  : {now} EST\n"
            f"Error : Server unreachable (connection refused)\n"
            f"URL   : {HEALTH_URL}"
        )

    except Exception as e:
        log.error("❌ Unexpected error: %s", e)
        send_whatsapp(
            f"🚨 RecipeVault Health Check FAILED\n"
            f"Time  : {now} EST\n"
            f"Error : {str(e)[:200]}\n"
            f"URL   : {HEALTH_URL}"
        )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("RecipeVault health monitor starting")
    log.info("Endpoint : %s", HEALTH_URL)
    log.info("Schedule : 5:00 AM and 5:00 PM EST (America/New_York)")
    log.info("WhatsApp : %s", WHATSAPP_PHONE or "NOT CONFIGURED")

    # Run once immediately on startup so you can confirm everything works
    run_check()

    scheduler = BlockingScheduler(timezone=TIMEZONE)

    # 5:00 AM EST / EDT every day
    scheduler.add_job(
        run_check,
        CronTrigger(hour=5, minute=0, timezone=TIMEZONE),
        name="5am-check",
    )

    # 5:00 PM EST / EDT every day
    scheduler.add_job(
        run_check,
        CronTrigger(hour=17, minute=0, timezone=TIMEZONE),
        name="5pm-check",
    )

    log.info("Scheduler started — next runs at 05:00 and 17:00 America/New_York")
    try:
        scheduler.start()
    except KeyboardInterrupt:
        log.info("Stopped.")
