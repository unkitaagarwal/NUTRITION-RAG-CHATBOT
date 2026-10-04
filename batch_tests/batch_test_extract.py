#!/usr/bin/env python3
"""Batch-test /extract-recipe against every URL in a CSV.

Appends two columns to the CSV:
  test_response  compact JSON summary of the API response (recipe name, counts,
                 method, missing, warnings, error, seconds, first steps/ingredients)
  test_status    "success" when the response has >=1 ingredient AND >=1 step,
                 otherwise "failure"

Stdlib only — no venv needed:
  python3 batch_test_extract.py                       # all rows, 4 parallel
  python3 batch_test_extract.py --limit 20            # quick smoke test
  python3 batch_test_extract.py --workers 6 --no-cache

Resumable: every finished URL is appended to <output>.jsonl; re-running skips
URLs already there (use --fresh to start over). The output CSV is rewritten
every 25 results and at the end, so you can open it while it runs.
"""
import argparse
import csv
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = os.path.join(HERE, "broken_imports_FULL.csv")
DEFAULT_ENDPOINT = "http://10.0.0.229:5003/extract-recipe"


def call_endpoint(endpoint: str, url: str, timeout: float, no_cache: bool) -> tuple[int, dict | None, str, float]:
    body = {"url": url, "mode": "auto"}
    if no_cache:
        body["no_cache"] = True
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            code = resp.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        code = e.code
    except Exception as e:  # timeout, connection refused, ...
        return 0, None, f"{type(e).__name__}: {e}", time.time() - t0
    elapsed = time.time() - t0
    try:
        return code, json.loads(raw), "", elapsed
    except ValueError:
        return code, None, raw[:300], elapsed


def summarize(code: int, data: dict | None, err: str, elapsed: float) -> tuple[dict, str]:
    s = {"http": code, "seconds": round(elapsed, 1)}
    if data is None:
        s["error"] = err or "non-JSON response"
        return s, "failure"
    recipe = data.get("recipe") or {}
    ings = [
        i for i in (recipe.get("ingredients") or [])
        if (isinstance(i, dict) and str(i.get("name") or "").strip()) or (isinstance(i, str) and i.strip())
    ]
    steps = [
        st for st in (recipe.get("instructions") or [])
        if str(st.get("instruction", "") if isinstance(st, dict) else st).strip()
    ]
    s.update({
        "name": recipe.get("name") or "",
        "n_ingredients": len(ings),
        "n_steps": len(steps),
        "method": (data.get("extraction") or {}).get("method"),
        "cached": bool(data.get("cached")),
    })
    if recipe.get("missing"):
        s["missing"] = recipe["missing"]
    if data.get("warnings"):
        s["warnings"] = data["warnings"]
    if data.get("error"):
        s["error"] = data.get("error")
        s["user_message"] = data.get("user_message")
        if data.get("details"):
            s["details"] = str(data["details"])[:300]
    s["sample_ingredients"] = [
        (f"{i.get('quantity', '')} {i.get('name', '')}".strip() if isinstance(i, dict) else i)
        for i in ings[:3]
    ]
    s["sample_steps"] = [str(x)[:120] for x in steps[:2]]
    ok = code == 200 and len(ings) >= 1 and len(steps) >= 1
    if not ok and "error" not in s:
        gaps = [k for k, n in (("ingredients", len(ings)), ("steps", len(steps))) if n == 0]
        s["reason"] = "no " + " and no ".join(gaps) if gaps else f"HTTP {code}"
    return s, ("success" if ok else "failure")


def write_csv(in_path: str, out_path: str, results: dict) -> tuple[int, int, int]:
    with open(in_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    url_i = header.index("url")
    succ = fail = pending = 0
    tmp = out_path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header + ["test_response", "test_status"])
        for r in body:
            res = results.get(r[url_i].strip())
            if res is None:
                pending += 1
                w.writerow(r + ["", ""])
                continue
            succ += res["status"] == "success"
            fail += res["status"] == "failure"
            w.writerow(r + [json.dumps(res["summary"], ensure_ascii=False), res["status"]])
    os.replace(tmp, out_path)
    return succ, fail, pending


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_IN)
    ap.add_argument("--output", default=None, help="default: <input>_tested.csv")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--limit", type=int, default=0, help="only test the first N URLs")
    ap.add_argument("--no-cache", action="store_true", help="send no_cache=true (forces fresh extraction)")
    ap.add_argument("--fresh", action="store_true", help="ignore previous progress")
    a = ap.parse_args()

    out = a.output or os.path.splitext(a.input)[0] + "_tested.csv"
    progress = out + ".jsonl"
    if a.fresh and os.path.exists(progress):
        os.replace(progress, progress + ".bak")

    results: dict = {}
    if os.path.exists(progress):
        with open(progress, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    results[rec["url"]] = rec
                except Exception:
                    pass

    with open(a.input, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        urls = []
        seen = set()
        for row in rd:
            u = (row.get("url") or "").strip()
            if u and u not in seen:
                seen.add(u)
                urls.append(u)
    if a.limit:
        urls = urls[: a.limit]
    todo = [u for u in urls if u not in results]
    print(f"{len(urls)} URLs, {len(urls) - len(todo)} already done, {len(todo)} to run "
          f"→ {a.endpoint} ({a.workers} parallel)")

    lock = threading.Lock()
    done = 0
    t_start = time.time()

    def work(u):
        code, data, err, el = call_endpoint(a.endpoint, u, a.timeout, a.no_cache)
        summary, status = summarize(code, data, err, el)
        return {"url": u, "status": status, "summary": summary, "full_response": data}

    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(progress, "a", encoding="utf-8") as pf:
        futs = {ex.submit(work, u): u for u in todo}
        for fut in as_completed(futs):
            rec = fut.result()
            with lock:
                results[rec["url"]] = rec
                pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                pf.flush()
                done += 1
                s = rec["summary"]
                rate = (time.time() - t_start) / done
                print(f"[{done}/{len(todo)}] {rec['status']:7} {s.get('seconds', 0):5.1f}s "
                      f"ing={s.get('n_ingredients', '-')} steps={s.get('n_steps', '-')} "
                      f"{rec['url'][:70]}  (eta {rate * (len(todo) - done) / 60:.0f} min)")
                if done % 25 == 0:
                    write_csv(a.input, out, results)

    succ, fail, pending = write_csv(a.input, out, results)
    print(f"\nDone. success={succ} failure={fail} untested={pending}\nCSV: {out}\nFull responses: {progress}")


if __name__ == "__main__":
    sys.exit(main())
