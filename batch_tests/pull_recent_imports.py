#!/usr/bin/env python3
"""Pull the last N days of REAL import attempts into a CSV for A/B testing.

Everything that was tried, not just what failed — the point of the A/B run is to
prove nothing that worked stopped working.

  python3 pull_recent_imports.py --days 2
  python3 pull_recent_imports.py --days 2 --include-dupes --out mine.csv

Auth: scripts/serviceAccountKey.json, or gcloud application-default login.
"""
import argparse, csv, os, sys
from collections import Counter
from datetime import datetime, timedelta, timezone

from google.cloud import firestore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CANDIDATE_KEYS = [
    os.path.join(ROOT, "scripts", "serviceAccountKey.json"),
    os.path.join(os.path.dirname(ROOT), "MealMap", "scripts", "serviceAccountKey.json"),
    os.environ.get("FIREBASE_KEY") or "",
]
KEY = next((k for k in _CANDIDATE_KEYS if k and os.path.exists(k)), "")
if KEY:
    from google.oauth2 import service_account
    db = firestore.Client(project="meal-plan-chef",
                          credentials=service_account.Credentials.from_service_account_file(KEY))
else:
    db = firestore.Client(project="meal-plan-chef")


def outcome_of(d):
    """How production actually answered: ok | silent | failed | cancelled."""
    o = d.get("outcome")
    if o == "cancelled_by_user":
        return "cancelled"
    if o == "failure":
        return "failed"
    if (d.get("stepCount") or 0) == 0 or (d.get("ingredientCount") or 0) == 0:
        return "silent"
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=2.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "recent_imports.csv"))
    ap.add_argument("--include-dupes", action="store_true",
                    help="keep every attempt; default keeps one row per URL")
    ap.add_argument("--include-cancelled", action="store_true")
    args = ap.parse_args()

    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    rows, seen = [], set()
    for doc in db.collection_group("imports").stream():
        d = doc.to_dict() or {}
        ts = d.get("createdAt")
        if not isinstance(ts, datetime):
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts < since:
            continue
        url = (d.get("sourceUrl") or "").strip()
        if not url.startswith("http"):
            continue
        oc = outcome_of(d)
        if oc == "cancelled" and not args.include_cancelled:
            continue
        if not args.include_dupes:
            if url in seen:
                continue
            seen.add(url)
        rows.append({
            "url": url,
            "domain": d.get("sourceDomain") or "",
            "source_type": d.get("sourceType") or "",
            "surface": d.get("importSurface") or "in_app",
            "prod_outcome": oc,
            "prod_reason": d.get("reason") or "",
            "prod_http": d.get("httpStatus") if d.get("httpStatus") is not None else "",
            "prod_ingredients": d.get("ingredientCount") or 0,
            "prod_steps": d.get("stepCount") or 0,
            "prod_seconds": round(float(d.get("timetoextract") or 0), 1),
            "imported_at": ts.isoformat()[:19],
        })

    rows.sort(key=lambda r: r["imported_at"], reverse=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["url"])
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} rows -> {args.out}")
    print("production baseline:", dict(Counter(r["prod_outcome"] for r in rows)))
    print("by source          :", dict(Counter(r["source_type"] for r in rows).most_common(8)))
    ok = sum(1 for r in rows if r["prod_outcome"] == "ok")
    if rows:
        print(f"prod success rate  : {100*ok/len(rows):.1f}%  "
              f"(these are what MUST NOT regress)")


if __name__ == "__main__":
    sys.exit(main())
