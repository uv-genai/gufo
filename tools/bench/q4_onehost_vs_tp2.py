#!/usr/bin/env python3
"""Q4 standalone (one host) against Q4 TP=2 over RDMA (two hosts).

Answers: is TP2 worth it for Q4, given that Q4 fits one host?

Both configurations run the same build, the same model, the same `--context`
and the same `--sessions`, on the same primary host (strix-halo-1), so the only
difference is the parallelism. `--context 32768` is required on one host: four
full-context sessions exceed the request-state claim (TP2 splits state, so it
would cope with more, but is pinned to the same value here for a matched pair).
Measured depths top out well below 32k, so the cap does not affect per-token
throughput.

Methodology matches tools/bench/tp2_q4_q8_bench.py: warm-up request per depth,
REPS recorded requests, every prompt derived from a unique salt so no request
can hit a previous one's prompt cache, and `cached_tokens` checked on every
sample.

Run this ON strix-halo-1. Rank 1 is started over ssh for the TP2 case.

Usage:  python3 tools/bench/q4_onehost_vs_tp2.py
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import statistics
import subprocess
import sys
import time
import urllib.request

PORT = 8320                      # never 8080: the user's other model owns it
GUFO = "/home/ugo/projects/gufo/build/release/gufo"
MODEL_DIR = "/home/ugo/.cache/models/Qwen3.8-Flash-Next"
MODEL = f"{MODEL_DIR}/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
MTP = f"{MODEL_DIR}/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
MMPROJ = f"{MODEL_DIR}/mmproj-BF16.gguf"
RANK1_HOST = "strix-halo-2.local"
TOKEN = "92ffa404804d336980d95f224d6bd678"
SESSIONS = 4
CONTEXT = 32768
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
# (label, target prompt words, output tokens)
DEPTHS = [("short", 200, 256), ("medium", 2000, 256), ("long", 8000, 128)]
REPS = 5
WARMUP = 1
CONCURRENCY = 4
WORDS = ("distributed tensor parallel inference splits large neural network "
         "tensors across accelerators each rank computes a share of the "
         "layer partial results summed over a high speed link").split()


def prompt_for(n_words: int, salt: str) -> str:
    out = []
    i = 0
    while len(out) < n_words:
        seed = hashlib.blake2s(f"{salt}:{i}".encode(), digest_size=4).digest()
        out.append(WORDS[int.from_bytes(seed, "big") % len(WORDS)])
        i += 1
    return " ".join(out)


def sh(*cmd: str, **kw) -> None:
    subprocess.run(cmd, check=False, **kw)


def rdev() -> str:
    """The same device the rank scripts pick: lowest node GUID, its NAME.

    `ibv_devices` prints `name  guid`; taking the last field gives the GUID,
    which gufo rejects.
    """
    out = subprocess.run(["ibv_devices"], capture_output=True, text=True).stdout
    rows = [ln.split() for ln in out.splitlines() if "usb4_rdma" in ln]
    if not rows:
        return ""
    rows.sort(key=lambda r: r[1])   # lowest GUID, as the rank scripts do
    return rows[0][0]               # the device name


def common_flags(name: str) -> list[str]:
    return [
        "--served-model-name", name,
        "--model", MODEL,
        "--speculative", "mtp",
        "--mtp-model", MTP,
        "--mmproj", MMPROJ,
        "--sessions", str(SESSIONS),
        "--context", str(CONTEXT),
    ]


def health(timeout: float = 300.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health",
                                        timeout=2) as r:
                return r.status == 200
        except Exception:
            pass
        time.sleep(1)
    return False


def start_onehost(name: str) -> None:
    cmd = [GUFO, "serve", "--host", "0.0.0.0", "--port", str(PORT),
           "--max-request-bytes", "268435456", "llm"] + common_flags(name)
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def start_tp2(name: str) -> None:
    dev = rdev()
    rank1 = ([GUFO, "serve", "llm"] + common_flags(name) + [
        "--tp-world-size", "2", "--tp-rank", "1",
        "--tp-bootstrap-host", "192.168.1.22",
        "--tp-bootstrap-port", "18515",
        "--tp-control-port", "18516",
        "--tp-control-token", TOKEN,
        "--tp-rdma-device", dev,
    ])
    sh("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", RANK1_HOST,
       "nohup " + " ".join(rank1) + " >/home/ugo/q4-vs-tp2-rank1.log 2>&1 &",
       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    rank0 = ([GUFO, "serve", "--host", "0.0.0.0", "--port", str(PORT),
              "--max-request-bytes", "268435456", "llm"] + common_flags(name) + [
        "--tp-world-size", "2", "--tp-rank", "0",
        "--tp-bootstrap-port", "18515",
        "--tp-control-port", "18516",
        "--tp-control-token", TOKEN,
        "--tp-rdma-device", dev,
    ])
    subprocess.Popen(rank0, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)


def stop_all() -> None:
    sh("pkill", "-x", "gufo", stdout=subprocess.DEVNULL,
       stderr=subprocess.DEVNULL)
    sh("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", RANK1_HOST,
       "pkill -x gufo", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(6)


def ask(text: str, max_tokens: int, name: str) -> dict | None:
    body = json.dumps({"model": name, "messages": [{"role": "user", "content": text}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.load(r)
    except Exception as exc:  # noqa: BLE001
        print(f"    request failed: {exc}", flush=True)
        return None


def sample(text: str, max_tokens: int, name: str) -> dict | None:
    d = ask(text, max_tokens, name)
    if not d or "usage" not in d:
        return None
    u = d["usage"]
    return {
        "prompt_tokens": u.get("prompt_tokens", 0),
        "completion_tokens": u.get("completion_tokens", 0),
        "prefill_tps": u.get("prompt_tokens_per_second", 0.0),
        "decode_tps": u.get("completion_tokens_per_second", 0.0),
        "ttft_ms": u.get("gufo", {}).get("ttft_ms", 0.0),
        "cached_tokens": u.get("prompt_tokens_details", {}).get("cached_tokens", 0),
    }


def run_config(label: str, name: str, starter) -> dict[str, list[dict]]:
    print(f"\n=== {label} ===", flush=True)
    print(f"  starting ...", flush=True)
    starter(name)
    if not health():
        print("  !! failed to come up", flush=True)
        stop_all()
        return {}
    print("  ready", flush=True)
    out: dict[str, list[dict]] = {}
    for dlabel, nwords, mtok in DEPTHS:
        for i in range(WARMUP):
            ask(prompt_for(nwords, f"warm-{dlabel}-{i}-"), mtok, name)
        rows = []
        for i in range(REPS):
            s = sample(prompt_for(nwords, f"{label}-{dlabel}-{i}-"), mtok, name)
            if s:
                rows.append(s)
        out[dlabel] = rows
        if rows:
            bad = sum(1 for r in rows if r["cached_tokens"])
            pf = statistics.median(r["prefill_tps"] for r in rows)
            dc = statistics.median(r["decode_tps"] for r in rows)
            tt = statistics.median(r["ttft_ms"] for r in rows)
            pt = statistics.median(r["prompt_tokens"] for r in rows)
            ds = [r["decode_tps"] for r in rows]
            warn = f"  !! {bad} CACHED" if bad else ""
            print(f"    {dlabel:<7} prompt={pt:>6.0f}  prefill={pf:>7.1f}  "
                  f"decode={dc:>5.1f} [{min(ds):.1f}-{max(ds):.1f}]  "
                  f"ttft={tt:>7.1f}ms  cached={bad}{warn}", flush=True)
    t0 = time.time()
    with cf.ThreadPoolExecutor(CONCURRENCY) as ex:
        futs = [ex.submit(sample, prompt_for(200, f"{label}-conc-{i}-"), 256, name)
                for i in range(CONCURRENCY)]
        rows = [f.result() for f in futs if f.result()]
    wall = time.time() - t0
    if rows:
        total = sum(r["completion_tokens"] for r in rows)
        bad = sum(1 for r in rows if r["cached_tokens"])
        out["conc"] = rows
        print(f"    conc{CONCURRENCY}  aggregate={total / wall:5.1f} tok/s  "
              f"wall={wall:.1f}s  cached={bad}", flush=True)
    stop_all()
    return out


def main() -> int:
    if "strix-halo-1" not in subprocess.run(
            ["hostname"], capture_output=True, text=True).stdout:
        print("run this on strix-halo-1 (it launches rank 1 over ssh)")
        return 2
    results = {
        "onehost": run_config("onehost Q4", "q4-onehost", start_onehost),
        "tp2": run_config("TP2 Q4 (RDMA)", "q4-tp2", start_tp2),
    }
    a, b = results["onehost"], results["tp2"]
    print("\n" + "=" * 72)
    print("Q4 standalone  vs  Q4 TP=2 over RDMA   (same build, context 32768)")
    print("=" * 72)
    print(f"{'depth':<9}{'metric':<12}{'onehost':>12}{'TP2 RDMA':>12}{'TP2/1h':>10}")
    print("-" * 72)
    for label, _, _ in DEPTHS:
        ra, rb = a.get(label) or [], b.get(label) or []
        if not ra or not rb:
            print(f"{label:<9}{'(missing)':<12}")
            continue
        for key, fmt in (("prefill_tps", "{:.1f}"), ("decode_tps", "{:.1f}"),
                         ("ttft_ms", "{:.1f}")):
            va = statistics.median(r[key] for r in ra)
            vb = statistics.median(r[key] for r in rb)
            print(f"{label:<9}{key:<12}{fmt.format(va):>12}"
                  f"{fmt.format(vb):>12}{(vb / va if va else 0):>+9.2f}x")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
