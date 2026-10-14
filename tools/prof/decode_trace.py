#!/usr/bin/env python3
"""Analyse a rocprofv3 --kernel-trace CSV for decode-phase behaviour.

prof.py's show needs the rocpd database, which does not build on this host
(schema version mismatch), so this reads the raw kernel trace directly.

It answers the three questions behind the decode optimisation work:
  * where does a decode step actually spend GPU time (per-kernel rollup);
  * how much of the wall clock is the GPU idle (occupancy), and what dispatch
    precedes the largest idle gaps - that is what exposes launch-bound and
    host-bound behaviour;
  * how the passes split (warm-up prefill, timed prefill, decode);
  * --shapes groups one kernel by grid dimension, which is how separate
    dispatches of the same templated kernel are told apart into distinct
    projections when their row widths differ.

Usage: tools/prof/decode_trace.py TRACE.csv [--gap-ms 0.5] [--top 20]
                                   [--shapes NAME_FRAGMENT] [--tokens N]
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("Kind") != "KERNEL_DISPATCH":
                continue
            try:
                start = int(r["Start_Timestamp"])
                end = int(r["End_Timestamp"])
            except (KeyError, ValueError):
                continue
            rows.append(
                {
                    "name": r["Kernel_Name"],
                    "start": start,
                    "end": end,
                    "dur": end - start,
                    "grid": int(r.get("Grid_Size_X", 0) or 0),
                    "wg": int(r.get("Workgroup_Size_X", 0) or 0),
                    "dispatch": int(r.get("Dispatch_Id", 0)),
                }
            )
    rows.sort(key=lambda x: x["start"])
    return rows


def shapes(rows, fragment, tokens, top):
    """Group one kernel by grid shape, to separate projections sharing a name."""
    # Decode is the final continuous stretch; take the tail of the trace.
    tail = rows[-1]["start"] - 2_500_000_000
    dec = [r for r in rows if r["start"] >= tail]
    agg = {}
    for r in dec:
        if fragment not in r["name"]:
            continue
        key = (r["grid"], r["wg"])
        e = agg.setdefault(key, [0, 0])
        e[0] += 1
        e[1] += r["dur"]
    if not agg:
        print(f"no decode dispatch matches {fragment!r}", file=sys.stderr)
        return 1
    print(f"grid shapes for {fragment!r} in the decode tail:")
    print(f"  {'gridX':>9}{'wg':>6}{'count':>7}{'per tok':>9}{'avg us':>9}{'total ms':>10}")
    for (gx, wg), (count, total) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:top]:
        print(
            f"  {gx:>9}{wg:>6}{count:>7}{count / tokens:>9.2f}"
            f"{total / count / 1e3:>9.1f}{total / 1e6:>10.1f}"
        )
    print(f"\ndistinct shapes: {len(agg)}")
    return 0


def order(rows, fragment, top):
    """Grid-shape transitions, to find projections issued back to back.

    A transition that always goes to the same successor identifies a fixed
    dispatch pair - the shape a fusion would collapse.
    """
    tail = rows[-1]["start"] - 2_500_000_000
    dec = [r for r in rows if r["start"] >= tail]
    seq = [r["grid"] for r in dec if fragment in r["name"]]
    if len(seq) < 2:
        print("not enough dispatches", file=sys.stderr)
        return 1
    succ = defaultdict(Counter)
    for a, b in zip(seq, seq[1:]):
        succ[a][b] += 1
    print(f"grid-shape transitions for {fragment!r} ({len(seq)} dispatches):")
    for a in sorted(succ, key=lambda k: -sum(succ[k].values()))[:top]:
        total = sum(succ[a].values())
        for b, c in succ[a].most_common(4):
            share = 100.0 * c / total
            mark = "  <- always" if share > 99.5 else ""
            print(
                f"  {a:>9} -> {b:>9}   {c:6d}   {share:5.1f}%{mark}"
            )
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--gap-ms", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument(
        "--shapes",
        metavar="NAME_FRAGMENT",
        help="group this kernel by grid shape instead of printing the rollup",
    )
    ap.add_argument(
        "--tokens",
        type=int,
        default=64,
        help="tokens generated, used to normalise per-token counts",
    )
    ap.add_argument(
        "--order",
        metavar="NAME_FRAGMENT",
        help="print grid-shape transition counts, to find fixed dispatch pairs",
    )
    args = ap.parse_args()

    rows = load(args.trace)
    if not rows:
        print("no kernel dispatch rows found", file=sys.stderr)
        return 1

    if args.shapes:
        return shapes(rows, args.shapes, args.tokens, args.top)
    if args.order:
        return order(rows, args.order, args.top)

    t0 = rows[0]["start"]
    t1 = max(r["end"] for r in rows)
    wall = (t1 - t0) / 1e6  # ms
    busy = sum(r["dur"] for r in rows) / 1e6

    print(f"dispatches: {len(rows)}")
    print(f"wall span : {wall:.2f} ms")
    print(f"gpu busy  : {busy:.2f} ms")
    print(f"occupancy : {100.0 * busy / wall:.2f}%")

    # Split into passes wherever the GPU idles longer than the threshold.
    gap_ns = int(args.gap_ms * 1e6)
    passes = []
    cur = [rows[0]]
    for prev, nxt in zip(rows, rows[1:]):
        if nxt["start"] - prev["end"] > gap_ns:
            passes.append(cur)
            cur = [nxt]
        else:
            cur.append(nxt)
    passes.append(cur)

    print(f"\npasses (split at >{args.gap_ms} ms idle): {len(passes)}")
    for i, p in enumerate(passes):
        s = p[0]["start"]
        e = max(r["end"] for r in p)
        b = sum(r["dur"] for r in p) / 1e6
        span = (e - s) / 1e6
        print(
            f"  pass {i}: {len(p):6d} dispatches  span {span:9.2f} ms  "
            f"busy {b:9.2f} ms  occ {100.0 * b / span:5.1f}%"
        )

    # The decode pass is the last one; roll it up per kernel.
    dec = passes[-1]
    agg = defaultdict(lambda: [0, 0])
    for r in dec:
        a = agg[r["name"]]
        a[0] += r["dur"]
        a[1] += 1
    d_busy = sum(v[0] for v in agg.values()) / 1e6
    d_span = (max(r["end"] for r in dec) - dec[0]["start"]) / 1e6
    print(
        f"\ndecode pass (last): span {d_span:.2f} ms, busy {d_busy:.2f} ms, "
        f"occupancy {100.0 * d_busy / d_span:.2f}%"
    )
    print(f"\ntop kernels by total time in decode (of {len(agg)} distinct):")
    print(f"  {'kernel':<62} {'total ms':>10} {'count':>8} {'avg us':>9} {'%':>6}")
    for name, (total, count) in sorted(
        agg.items(), key=lambda kv: -kv[1][0]
    )[: args.top]:
        print(
            f"  {name[:62]:<62} {total / 1e6:10.3f} {count:8d} "
            f"{total / count / 1e3:9.2f} {100.0 * total / (d_busy * 1e6):6.1f}"
        )

    # Idle gaps inside the decode pass: what serialises the step.
    dec_sorted = sorted(dec, key=lambda r: r["start"])
    gaps = []
    for prev, nxt in zip(dec_sorted, dec_sorted[1:]):
        gap = nxt["start"] - prev["end"]
        if gap > 0:
            gaps.append((gap, prev["name"], nxt["name"]))
    gaps.sort(key=lambda g: -g[0])
    total_gap = sum(g[0] for g in gaps) / 1e6
    print(
        f"\nidle between dispatches inside decode: {total_gap:.2f} ms "
        f"({100.0 * total_gap / d_span:.2f}% of the pass)"
    )
    print(f"largest gaps (ms) and the kernels either side:")
    for gap, before, after in gaps[:10]:
        print(f"  {gap / 1e6:8.3f}  after {before[:44]:<44} -> {after[:44]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
