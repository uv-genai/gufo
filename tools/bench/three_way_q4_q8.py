#!/usr/bin/env python3
"""Matched three-way spot bench: standalone Q4, Q4-TP2, Q8-TP2.

Run ON strix-halo-1 (it launches rank 1 over ssh for the TP2 cases).
Same build, same --sessions/--context/--think off, same prompt generator
(unique salt per request — no prompt-cache hits), warm-up then REPS recorded
samples per (config, mode, depth). Server-reported prompt/decode tok/s only.

Configs:
  onehost-q4  standalone UD-Q4_K_XL (1 host)
  tp2-q4      UD-Q4_K_XL over RDMA (2 hosts)
  tp2-q8      Q8_0 over RDMA (Q8 does not fit one host)

Modes: ar (speculative off) and mtp (shared Q8 MTP draft).

Writes JSON under docs/models/qwen3.8-flash-next/artifacts/ and prints a
side-by-side table. Unqualified as a published model-bench table: spot depths,
REPS medians; use tools/bench/model-bench.py for the published cards.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PORT = 8321  # never 8080
GUFO = "/home/ugo/projects/gufo/build/release/gufo"
MODEL_DIR = Path("/home/ugo/.cache/models/Qwen3.8-Flash-Next")
Q4 = str(MODEL_DIR / "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")
Q8 = str(MODEL_DIR / "Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf")
MTP = str(MODEL_DIR / "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf")
MMPROJ = str(MODEL_DIR / "mmproj-BF16.gguf")
RANK1_HOST = "strix-halo-2.local"
TOKEN = "92ffa404804d336980d95f224d6bd678"
SESSIONS = 4
CONTEXT = 32768
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
# (label, target prompt words, output tokens)
DEPTHS = [("short", 200, 256), ("medium", 2000, 256), ("long", 8000, 128)]
MODES = ("ar", "mtp")
CONFIGS = ("onehost-q4", "tp2-q4", "tp2-q8")
REPS = 5
WARMUP = 1
CONCURRENCY = 4
WORDS = (
    "distributed tensor parallel inference splits large neural network "
    "tensors across accelerators each rank computes a share of the "
    "layer partial results summed over a high speed link"
).split()
OUT_DIR = Path(
    "/home/ugo/projects/gufo/docs/models/qwen3.8-flash-next/artifacts"
)


def prompt_for(n_words: int, salt: str) -> str:
    out: list[str] = []
    i = 0
    while len(out) < n_words:
        seed = hashlib.blake2s(f"{salt}:{i}".encode(), digest_size=4).digest()
        out.append(WORDS[int.from_bytes(seed, "big") % len(WORDS)])
        i += 1
    return " ".join(out)


def sh(*cmd: str) -> None:
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


def rdev() -> str:
    out = subprocess.run(["ibv_devices"], capture_output=True, text=True).stdout
    rows = [ln.split() for ln in out.splitlines() if "usb4_rdma" in ln]
    if not rows:
        return ""
    rows.sort(key=lambda r: r[1])
    return rows[0][0]


def model_for(config: str) -> str:
    return Q8 if config == "tp2-q8" else Q4


def flags_for(config: str, mode: str) -> list[str]:
    flags = [
        "--model", model_for(config),
        "--served-model-name", "bench-three-way",
        "--mmproj", MMPROJ,
        "--sessions", str(SESSIONS),
        "--context", str(CONTEXT),
        "--think", "off",
        "--max-pending-per-client", "8",
    ]
    if mode == "mtp":
        flags += ["--speculative", "mtp", "--mtp-model", MTP]
    else:
        flags += ["--speculative", "off"]
    return flags


def health(timeout: float = 360.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/ready",
                                        timeout=2) as r:
                return r.status == 200
        except Exception:
            pass
        time.sleep(1)
    return False


def stop_all() -> None:
    sh("pkill", "-x", "gufo")
    sh("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", RANK1_HOST,
       "pkill -x gufo")
    for _ in range(30):
        local = subprocess.run(["pgrep", "-x", "gufo"], capture_output=True).returncode == 0
        remote = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
             RANK1_HOST, "pgrep -x gufo >/dev/null"],
            capture_output=True).returncode == 0
        if not local and not remote:
            break
        time.sleep(1)
    time.sleep(3)


def start_onehost(config: str, mode: str) -> None:
    cmd = [
        GUFO, "serve", "--host", "0.0.0.0", "--port", str(PORT),
        "--max-request-bytes", "268435456", "llm", *flags_for(config, mode),
    ]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def start_tp2(config: str, mode: str) -> None:
    dev = rdev()
    if not dev:
        raise RuntimeError("no usb4_rdma device")
    common = flags_for(config, mode)
    rank1 = [
        GUFO, "serve", "llm", *common,
        "--tp-world-size", "2", "--tp-rank", "1",
        "--tp-bootstrap-host", "192.168.1.22",
        "--tp-bootstrap-port", "18515",
        "--tp-control-port", "18516",
        "--tp-control-token", TOKEN,
        "--tp-rdma-device", dev,
    ]
    sh("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", RANK1_HOST,
       "nohup " + " ".join(rank1) + " >/home/ugo/three-way-rank1.log 2>&1 &")
    time.sleep(2)
    rank0 = [
        GUFO, "serve", "--host", "0.0.0.0", "--port", str(PORT),
        "--max-request-bytes", "268435456", "llm", *common,
        "--tp-world-size", "2", "--tp-rank", "0",
        "--tp-bootstrap-port", "18515",
        "--tp-control-port", "18516",
        "--tp-control-token", TOKEN,
        "--tp-rdma-device", dev,
    ]
    subprocess.Popen(rank0, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def start(config: str, mode: str) -> bool:
    if config == "onehost-q4":
        start_onehost(config, mode)
    else:
        start_tp2(config, mode)
    return health()


def ask(text: str, max_tokens: int) -> dict | None:
    body = json.dumps({
        "model": "bench-three-way",
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 1,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.load(r)
    except Exception as exc:  # noqa: BLE001
        print(f"    request failed: {exc}", flush=True)
        return None


def sample(text: str, max_tokens: int) -> dict | None:
    d = ask(text, max_tokens)
    if not d or "usage" not in d:
        return None
    u = d["usage"]
    g = u.get("gufo") or {}
    return {
        "prompt_tokens": u.get("prompt_tokens", 0),
        "completion_tokens": u.get("completion_tokens", 0),
        "prefill_tps": u.get("prompt_tokens_per_second") or g.get("prefill_tps") or 0.0,
        "decode_tps": u.get("completion_tokens_per_second") or g.get("decode_tps") or 0.0,
        "ttft_ms": g.get("ttft_ms") or 0.0,
        "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "draft_accepted": g.get("draft_tokens_accepted"),
        "draft_proposed": g.get("draft_tokens"),
    }


def median(rows: list[dict], key: str) -> float:
    return statistics.median(r[key] for r in rows)


def run_cell(config: str, mode: str) -> dict:
    print(f"\n=== {config} / {mode} ===", flush=True)
    stop_all()
    if not start(config, mode):
        print("  !! failed to come up", flush=True)
        stop_all()
        return {}
    print("  ready", flush=True)
    out: dict[str, list[dict]] = {}
    for dlabel, nwords, mtok in DEPTHS:
        for i in range(WARMUP):
            ask(prompt_for(nwords, f"warm-{config}-{mode}-{dlabel}-{i}-"), mtok)
        rows: list[dict] = []
        for i in range(REPS):
            s = sample(prompt_for(nwords, f"{config}-{mode}-{dlabel}-{i}-"), mtok)
            if s and s["completion_tokens"] >= mtok * 0.9:
                rows.append(s)
            elif s:
                print(f"    !! {dlabel} rep{i} short out={s['completion_tokens']}",
                      flush=True)
        out[dlabel] = rows
        if rows:
            bad = sum(1 for r in rows if r["cached_tokens"])
            pf = median(rows, "prefill_tps")
            dc = median(rows, "decode_tps")
            pt = median(rows, "prompt_tokens")
            ds = [r["decode_tps"] for r in rows]
            print(
                f"    {dlabel:<7} prompt={pt:>6.0f}  prefill={pf:>7.1f}  "
                f"decode={dc:>5.1f} [{min(ds):.1f}-{max(ds):.1f}]  "
                f"cached={bad}",
                flush=True,
            )
    # C=4 aggregate (short prompts)
    import concurrent.futures as cf
    t0 = time.time()
    with cf.ThreadPoolExecutor(CONCURRENCY) as ex:
        futs = [
            ex.submit(sample, prompt_for(200, f"{config}-{mode}-conc-{i}-"), 256)
            for i in range(CONCURRENCY)
        ]
        rows = [f.result() for f in futs if f.result()]
    wall = time.time() - t0
    if rows:
        total = sum(r["completion_tokens"] for r in rows)
        out["conc4"] = rows
        print(f"    conc4  aggregate={total / wall:5.1f} tok/s  wall={wall:.1f}s",
              flush=True)
    stop_all()
    return out


def main() -> int:
    host = subprocess.run(["hostname"], capture_output=True, text=True).stdout
    if "strix-halo-1" not in host:
        print("run this on strix-halo-1")
        return 2
    sha = subprocess.run(["sha256sum", GUFO], capture_output=True,
                         text=True).stdout.split()[0]
    results: dict[str, dict] = {}
    t_start = time.time()
    for config in CONFIGS:
        for mode in MODES:
            results[f"{config}/{mode}"] = run_cell(config, mode)
    elapsed = time.time() - t_start

    # Summary table
    print("\n" + "=" * 96)
    print(f"Three-way spot bench  sha={sha[:12]}  REPS={REPS}  context={CONTEXT}  "
          f"elapsed={elapsed / 60:.1f} min")
    print("=" * 96)
    hdr = f"{'depth':<8}{'metric':<10}"
    for c in CONFIGS:
        for m in MODES:
            hdr += f"{c[:6]}/{m:>3}"
    # simpler layout: rows = config/mode, cols = depth prefill/decode
    for dlabel, _, _ in DEPTHS:
        print(f"\n--- {dlabel} (median of {REPS}) ---")
        print(f"{'config/mode':<20}{'prompt':>8}{'prefill':>10}{'decode':>10}"
              f"{'ttft_ms':>10}{'cached':>8}")
        for key, cell in results.items():
            rows = cell.get(dlabel) or []
            if not rows:
                print(f"{key:<20}{'(missing)':>8}")
                continue
            print(
                f"{key:<20}{median(rows, 'prompt_tokens'):>8.0f}"
                f"{median(rows, 'prefill_tps'):>10.1f}"
                f"{median(rows, 'decode_tps'):>10.1f}"
                f"{median(rows, 'ttft_ms'):>10.1f}"
                f"{sum(1 for r in rows if r['cached_tokens']):>8}"
            )
    print("\n--- conc4 aggregate (short) ---")
    for key, cell in results.items():
        rows = cell.get("conc4") or []
        if not rows:
            continue
        # recompute would need wall; store was printed live
        print(f"{key:<20} n={len(rows)}")

    payload = {
        "schema": "gufo-three-way-spot/1",
        "measuredOn": time.strftime("%Y-%m-%d"),
        "binarySha256": sha,
        "context": CONTEXT,
        "sessions": SESSIONS,
        "reps": REPS,
        "warmup": WARMUP,
        "depths": [d[0] for d in DEPTHS],
        "modes": list(MODES),
        "configs": list(CONFIGS),
        "elapsedSec": elapsed,
        "note": "Spot bench on v0.26.0 TP2 merge; not model-bench published tables.",
        "results": results,
    }
    out_path = OUT_DIR / f"three-way-{payload['measuredOn']}.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {out_path}")
    stop_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
