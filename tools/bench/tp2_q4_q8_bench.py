#!/usr/bin/env python3
"""Q4 vs Q8 TP2-over-RDMA spot check.

Drives the tp2-merged-swap.sh wrapper (rank 0 on this host, rank 1 over ssh),
measures with the server's own reported per-request metrics, and prints a
side-by-side comparison. Nothing is written to the repo's artifacts/ dir.

UNQUALIFIED: REPS=2 spot numbers, no warm-up or repetition control, no matched
single-host baseline. Results are recorded in
docs/models/qwen3.8-flash-next/TP2-Q4-vs-Q8.md. For a publishable comparison
raise REPS, add a warm-up request per depth, and follow
 docs/models/qwen3.8-flash-next/QUALITY.md.

Rank 0 must be the local host and rank 1 reachable over ssh. Never binds 8080.

Usage:  python3 tools/bench/tp2_q4_q8_bench.py
"""
from __future__ import annotations

import json
import hashlib
import statistics
import subprocess
import sys
import time
import urllib.request

PORT = 8318                      # never 8080: the user's other model owns it
CLI = "/home/ugo/opt/llama.cpp-cli/tp2-merged-swap.sh"
MODEL = {
    "q8": "gufo-qwen3.8-flash-next-tp2-012",
    "q4": "gufo-qwen3.8-flash-next-tp2-012-q4",
}
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
# (label, target prompt tokens, output tokens)
DEPTHS = [("short", 200, 256), ("medium", 2000, 256), ("long", 8000, 128)]
REPS = 5
WARMUP = 1                 # unrecorded requests per depth before timing
CONCURRENCY = 4


def sh(*cmd: str, quiet: bool = True) -> None:
    subprocess.run(cmd, check=False,
                   stdout=subprocess.DEVNULL if quiet else None,
                   stderr=subprocess.DEVNULL if quiet else None)


def health(timeout: float = 240.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def start(variant: str) -> bool:
    print(f"  starting {variant} pair on :{PORT} ...", flush=True)
    subprocess.Popen([CLI, "run", str(PORT), variant],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return health()


def stop() -> None:
    sh(CLI, "stop")
    time.sleep(4)


def prompt_for(target_tokens: int, salt: str = "") -> str:
    # Fill the prompt with words whose content is derived from `salt`, so no
    # two requests share a prefix or long substring. Repetitive filler is not
    # safe here: gufo's prompt cache will recognise it and report prefill=0,
    # silently halving the measured rate. The server reports the real
    # prompt_tokens and that is what every rate uses.
    words = ("distributed tensor parallel inference splits large neural network "
             "tensors across accelerators each rank computes a share of the "
             "layer partial results summed over a high speed link").split()
    out = []
    i = 0
    while len(out) < target_tokens:
        seed = hashlib.blake2s(f"{salt}:{i}".encode(), digest_size=4).digest()
        out.append(words[int.from_bytes(seed, "big") % len(words)])
        i += 1
    return " ".join(out)


def ask(text: str, max_tokens: int) -> dict | None:
    body = json.dumps({
        "model": MODEL[current_variant],
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)
    except Exception as exc:  # noqa: BLE001
        print(f"    request failed: {exc}", flush=True)
        return None


def sample(text: str, max_tokens: int) -> dict | None:
    d = ask(text, max_tokens)
    if not d:
        return None
    u = d.get("usage", {})
    return {
        "prompt_tokens": u.get("prompt_tokens", 0),
        "completion_tokens": u.get("completion_tokens", 0),
        "prefill_tps": u.get("prompt_tokens_per_second", 0.0),
        "decode_tps": u.get("completion_tokens_per_second", 0.0),
        "ttft_ms": u.get("gufo", {}).get("ttft_ms", 0.0),
        "cached_tokens": u.get("prompt_tokens_details", {}).get("cached_tokens", 0),
        "drafted": u.get("draft_tokens", 0),
        "accepted": u.get("draft_tokens_accepted", 0),
    }


current_variant = "q8"


def run_variant(variant: str) -> dict:
    global current_variant
    current_variant = variant
    out: dict[str, list[dict]] = {}
    if not start(variant):
        print(f"  !! pair failed to come up for {variant}", flush=True)
        stop()
        return out
    print("  pair ready", flush=True)
    for label, ptok, mtok in DEPTHS:
        rows = []
        for i in range(WARMUP):          # keep cold-start out of the sample
            ask(prompt_for(ptok, f"warmup-{i}-"), mtok)
        for i in range(REPS):
            s = sample(prompt_for(ptok, f"rep-{i}-"), mtok)
            if s:
                rows.append(s)
        out[label] = rows
        if rows:
            bad = [r for r in rows if r["cached_tokens"]]
            if bad:
                print(f"    !! {label}: {len(bad)} cached request(s) "
                      f"polluted prefill — treat these rates as invalid",
                      flush=True)
            pf = statistics.median(r["prefill_tps"] for r in rows)
            ck = statistics.median(r["cached_tokens"] for r in rows)
            dc = statistics.median(r["decode_tps"] for r in rows)
            tt = statistics.median(r["ttft_ms"] for r in rows)
            pt = statistics.median(r["prompt_tokens"] for r in rows)
            ds = [r["decode_tps"] for r in rows]
            print(f"    {label:<7} prompt={pt:>6.0f} tok  "
                  f"prefill={pf:>7.1f} tok/s  decode={dc:>5.1f} "
                  f"[{min(ds):.1f}-{max(ds):.1f}]  "
                  f"ttft={tt:>7.1f} ms  cached={ck:.0f}  n={len(rows)}",
                  flush=True)
    # concurrency spot check at the medium depth
    t0 = time.time()
    procs = []
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(CONCURRENCY) as ex:
        futs = [ex.submit(sample, prompt_for(2000, f"conc-{i}-"), 256)
                for i in range(CONCURRENCY)]
        rows = [f.result() for f in futs if f.result()]
    wall = time.time() - t0
    if rows:
        total = sum(r["completion_tokens"] for r in rows)
        out["conc"] = rows
        print(f"    concurrency={CONCURRENCY}  {len(rows)} ok  "
              f"aggregate decode={total / wall:5.1f} tok/s  "
              f"wall={wall:.1f}s", flush=True)
    stop()
    return out


def main() -> int:
    results = {}
    for variant in ("q8", "q4"):
        print(f"\n=== variant {variant} ({MODEL[variant]}) ===", flush=True)
        results[variant] = run_variant(variant)

    print("\n" + "=" * 74)
    print("Q4 vs Q8  —  TP2 over RDMA (usb4_rdma0, RoCE v2, write striping)")
    print("=" * 74)
    hdr = (f"{'depth':<9}{'metric':<12}{'Q8_0':>12}{'Q4_K_XL':>12}{'Q4/Q8':>10}")
    print(hdr)
    print("-" * 74)
    for label, _, _ in DEPTHS:
        a = results.get("q8", {}).get(label) or []
        b = results.get("q4", {}).get(label) or []
        if not a or not b:
            print(f"{label:<9}{'(missing)':<12}")
            continue
        for key, fmt in (("prefill_tps", "{:.1f}"),
                         ("decode_tps", "{:.1f}"),
                         ("ttft_ms", "{:.1f}")):
            va = statistics.median(r[key] for r in a)
            vb = statistics.median(r[key] for r in b)
            ratio = f"{vb / va:+.2f}x" if va else "—"
            print(f"{label:<9}{key:<12}{fmt.format(va):>12}"
                  f"{fmt.format(vb):>12}{ratio:>10}")
        print()
    ca = results.get("q8", {}).get("conc") or []
    cb = results.get("q4", {}).get("conc") or []
    if ca and cb:
        print(f"{'conc'+str(CONCURRENCY):<9}{'decode':<12}"
              f"{len(ca)} streams vs {len(cb)} streams  (see lines above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
