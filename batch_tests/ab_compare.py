#!/usr/bin/env python3
"""A/B the extractor: same URLs, two endpoints, side-by-side.

Answers one question — did anything that worked stop working — and then whether
the changes helped on success, time and completeness.

  python3 ab_compare.py --baseline https://nutrition-rag-chatbot.onrender.com \\
                        --candidate http://localhost:5003 \\
                        --in recent_imports.csv --workers 4

Stdlib only. Resumable: finished URLs append to <out>.jsonl, re-runs skip them
(--fresh to start over). Both sides always run with no_cache so you are
comparing extraction, not cache state. Call order alternates per URL so neither
side systematically benefits from a warm upstream CDN.
"""
import argparse, csv, json, os, random, statistics, sys, threading, time
import urllib.error, urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
_print_lock = threading.Lock()


def call(endpoint: str, url: str, timeout: float):
    body = json.dumps({"url": url, "mode": "auto", "no_cache": True}).encode()
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/extract-recipe", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw, code = r.read().decode("utf-8", "replace"), r.status
    except urllib.error.HTTPError as e:
        raw, code = e.read().decode("utf-8", "replace"), e.code
    except Exception as e:
        return {"ok": False, "http": 0, "seconds": round(time.time()-t0, 1),
                "ingredients": 0, "steps": 0, "method": "", "error": f"{type(e).__name__}: {e}"[:200],
                "generated": False, "retry_after": None, "clipped": None}
    secs = round(time.time()-t0, 1)
    try:
        d = json.loads(raw)
    except ValueError:
        return {"ok": False, "http": code, "seconds": secs, "ingredients": 0, "steps": 0,
                "method": "", "error": raw[:200], "generated": False,
                "retry_after": None, "clipped": None}
    rec = d.get("recipe") or {}
    ing = rec.get("ingredients") or []
    steps = rec.get("instructions") or rec.get("steps") or []
    src = d.get("source") or {}
    return {
        "ok": code == 200 and len(ing) > 0 and len(steps) > 0,
        "http": code,
        "seconds": secs,
        "ingredients": len(ing) if isinstance(ing, list) else 0,
        "steps": len(steps) if isinstance(steps, list) else 0,
        "method": (d.get("extraction") or {}).get("method", ""),
        "error": (d.get("error") or "")[:200],
        "generated": bool(rec.get("instructions_generated")),
        "retry_after": d.get("retry_after_sec"),
        "clipped": (src.get("clipped_to_seconds") or d.get("clipped_to_seconds")),
    }


def verdict(a, b):
    if a["ok"] and not b["ok"]:
        return "REGRESSION"
    if not a["ok"] and b["ok"]:
        return "FIXED"
    return "both_ok" if a["ok"] else "both_broken"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="current API base URL")
    ap.add_argument("--candidate", required=True, help="changed API base URL")
    ap.add_argument("--in", dest="infile", default=os.path.join(HERE, "recent_imports.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "ab_results.csv"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.infile, encoding="utf-8")))
    if args.limit:
        rows = rows[:args.limit]

    jl = args.out + ".jsonl"
    done = {}
    if os.path.exists(jl) and not args.fresh:
        for line in open(jl, encoding="utf-8"):
            try:
                r = json.loads(line); done[r["url"]] = r
            except ValueError:
                pass
    todo = [r for r in rows if r["url"] not in done]
    print(f"{len(rows)} urls | {len(done)} already done | {len(todo)} to run | {args.workers} workers")

    jlf = open(jl, "a", encoding="utf-8")
    counter = {"n": 0}

    def work(row):
        url = row["url"]
        # alternate which side goes first so neither gets the warm-cache edge
        if random.random() < 0.5:
            a = call(args.baseline, url, args.timeout); b = call(args.candidate, url, args.timeout)
        else:
            b = call(args.candidate, url, args.timeout); a = call(args.baseline, url, args.timeout)
        out = {"url": url, "domain": row.get("domain", ""), "source_type": row.get("source_type", ""),
               "prod_outcome": row.get("prod_outcome", ""), "A": a, "B": b, "verdict": verdict(a, b)}
        with _print_lock:
            jlf.write(json.dumps(out) + "\n"); jlf.flush()
            counter["n"] += 1
            mark = {"REGRESSION": "!!", "FIXED": "++", "both_ok": "  ", "both_broken": " ."}[out["verdict"]]
            print(f"{mark} [{counter['n']}/{len(todo)}] {out['verdict']:<12} "
                  f"A={a['seconds']:>5.1f}s/{a['ingredients']}i{a['steps']}s  "
                  f"B={b['seconds']:>5.1f}s/{b['ingredients']}i{b['steps']}s  {url[:66]}")
        return out

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(work, r) for r in todo]):
            try:
                res = f.result(); done[res["url"]] = res
            except Exception as e:
                print("worker error:", e)
    jlf.close()

    results = list(done.values())
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["verdict", "url", "domain", "source_type", "prod_outcome",
                    "A_ok", "A_http", "A_secs", "A_ing", "A_steps", "A_method", "A_error",
                    "B_ok", "B_http", "B_secs", "B_ing", "B_steps", "B_method", "B_error",
                    "B_generated_steps", "B_retry_after", "B_clipped_to"])
        for r in sorted(results, key=lambda r: {"REGRESSION": 0, "FIXED": 1, "both_broken": 2, "both_ok": 3}[r["verdict"]]):
            a, b = r["A"], r["B"]
            w.writerow([r["verdict"], r["url"], r["domain"], r["source_type"], r["prod_outcome"],
                        a["ok"], a["http"], a["seconds"], a["ingredients"], a["steps"], a["method"], a["error"],
                        b["ok"], b["http"], b["seconds"], b["ingredients"], b["steps"], b["method"], b["error"],
                        b["generated"], b["retry_after"], b["clipped"]])

    # ── report ───────────────────────────────────────────────────────────────
    n = len(results)
    if not n:
        return 0
    v = Counter(r["verdict"] for r in results)
    a_ok = sum(1 for r in results if r["A"]["ok"]); b_ok = sum(1 for r in results if r["B"]["ok"])
    at = [r["A"]["seconds"] for r in results]; bt = [r["B"]["seconds"] for r in results]
    med = lambda x: statistics.median(x) if x else 0
    p90 = lambda x: sorted(x)[int(len(x)*0.9)] if x else 0

    print("\n" + "="*72)
    print(f"{n} URLs   baseline={args.baseline}   candidate={args.candidate}")
    print("="*72)
    print(f"  REGRESSIONS (worked -> broken) : {v['REGRESSION']:>5}   <-- must be 0")
    print(f"  FIXED       (broken -> worked) : {v['FIXED']:>5}")
    print(f"  both ok                        : {v['both_ok']:>5}")
    print(f"  both broken                    : {v['both_broken']:>5}")
    print(f"\n  success   A {a_ok:>4}/{n} ({100*a_ok/n:5.1f}%)   B {b_ok:>4}/{n} ({100*b_ok/n:5.1f}%)   "
          f"delta {100*(b_ok-a_ok)/n:+.1f} pts")
    print(f"  time      A median {med(at):5.1f}s p90 {p90(at):5.1f}s   "
          f"B median {med(bt):5.1f}s p90 {p90(bt):5.1f}s")
    print(f"  steps>0   A {sum(1 for r in results if r['A']['steps']>0):>4}   "
          f"B {sum(1 for r in results if r['B']['steps']>0):>4}")
    print(f"  B used generated steps         : {sum(1 for r in results if r['B']['generated'])}")
    print(f"  B returned retry_after_sec     : {sum(1 for r in results if r['B']['retry_after'])}")
    print(f"  B clipped a long video         : {sum(1 for r in results if r['B']['clipped'])}")

    if v["REGRESSION"]:
        print("\n  REGRESSIONS — investigate every one:")
        for r in results:
            if r["verdict"] == "REGRESSION":
                print(f"    {r['source_type']:<10} {r['url'][:70]}")
                print(f"       A: http={r['A']['http']} {r['A']['ingredients']}i/{r['A']['steps']}s "
                      f"method={r['A']['method']}")
                print(f"       B: http={r['B']['http']} {r['B']['ingredients']}i/{r['B']['steps']}s "
                      f"err={r['B']['error'][:90]}")

    by = defaultdict(lambda: [0, 0, 0, 0])
    for r in results:
        k = r["source_type"] or "other"
        by[k][0] += 1; by[k][1] += r["A"]["ok"]; by[k][2] += r["B"]["ok"]
        by[k][3] += r["verdict"] == "REGRESSION"
    print(f"\n  {'source':<12}{'n':>5}{'A ok':>7}{'B ok':>7}{'regress':>9}")
    for k, (t, ao, bo, rg) in sorted(by.items(), key=lambda kv: -kv[1][0]):
        print(f"  {k:<12}{t:>5}{ao:>7}{bo:>7}{rg:>9}")
    print(f"\nrows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
