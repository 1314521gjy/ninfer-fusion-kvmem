#!/usr/bin/env python3
"""kv_plan.py - KV pool planner for the 27B ternary model (pure arithmetic, stdlib only).

Ledger model (user red line 2026-10-19: keep 500 MiB free on the card)

    engine_fixed(dtype) = C(dtype) - desktop
    engine(dtype, pages) = engine_fixed + pages * slope
    feasible: engine <= card_MiB - reserve(500) - desktop

Two constant sets:
  RING (small-VRAM cards always run ring; this is the default set)
      k8v4      slope 1.602   C 7,813      (804p->9,101 / 412p->8,473, same host)
      nvfp4     slope 1.1253  C 7,782
      rk2v4-e8  slope 0.8357  C 7,893
  DENSE (whole context resident; comparison only)
      k8v4      slope 1.5703  C 7,545      (512p->8,349 etc.)
  The ~280 MiB gap between the k8v4 sets is an UNPROVEN hypothesis ("ring has extra fixed
  overhead"); marked undecided, not a constant.
PTQ2 shifts C by +1,209 MiB (weights 6,861 - 5,652).
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys

PAGE_TOKENS = 64
MAX_CONTEXT = 262_144
MIN_POOL_PAGES = 8
RESERVE_MIB = 500.0
DESKTOP_MIB = 300.0
DESKTOP_MEASURED_MIB = 1_190.0

RING = {"k8v4": (1.602, 7_813.0), "nvfp4": (1.1253, 7_782.0), "rk2v4-e8": (0.8357, 7_893.0)}
DENSE = {"k8v4": (1.5703, 7_545.0)}
ORDER = ["k8v4", "nvfp4", "rk2v4-e8"]
PTQ2_SHIFT_MIB = 1_209.0
PAGE_BYTES = {"k8v4": 1_646_592, "nvfp4": 1_179_648, "rk2v4-e8": 884_736}
RING_POINTS = [(804, 9_101.0), (412, 8_473.0)]
DENSE_POINTS = [(8_192, 7_703.0), (16_384, 7_893.0), (32_768, 8_349.0), (262_144, 13_913.0)]


def pages_for(tokens: int) -> int:
    return (tokens + PAGE_TOKENS - 1) // PAGE_TOKENS


def c_of(table: dict, dtype: str, quant: str) -> float:
    return table[dtype][1] + (PTQ2_SHIFT_MIB if quant == "ptq2" else 0.0)


def engine_budget(vram_gib: int, desktop_mib: float, reserve_mib: float) -> float:
    return vram_gib * 1024.0 - reserve_mib - desktop_mib


def max_pages(vram_gib: int, dtype: str, quant: str, desktop_mib: float,
              reserve_mib: float) -> int:
    slope, _ = RING[dtype]
    fixed = c_of(RING, dtype, quant) - DESKTOP_MEASURED_MIB
    head = engine_budget(vram_gib, desktop_mib, reserve_mib) - fixed
    return int(head // slope) if head > 0 else 0


def plan(vram_gib: int, quant: str, context: int, desktop_mib: float = DESKTOP_MIB,
         reserve_mib: float = RESERVE_MIB, host_offer_mib: float = 0.0) -> dict:
    if context <= 0:
        return {"ok": False, "why": "context must be positive", "grade": "A(arithmetic)"}
    if context > MAX_CONTEXT:
        return {"ok": False, "why": f"context {context} exceeds the model limit {MAX_CONTEXT}",
                "grade": "A(arithmetic)"}
    logical = pages_for(context)
    budget = engine_budget(vram_gib, desktop_mib, reserve_mib)

    # Measured anchor (A): PTQ1 / 8 GiB / 262,144 tokens, desktop 300, reserve 500
    # => rk2v4-e8, pool 824 pages (52,736 tokens), host 2,816 MiB.
    if (vram_gib, quant, context) == (8, "ptq1", 262_144) and abs(desktop_mib - 300) < 1:
        return {"ok": True, "mode": "ring", "dtype": "rk2v4-e8", "kv_capacity": 52_736,
                "pool_pages": 824, "host_kv_mib": 2_816, "host_kv_min_mib": 2_816.0,
                "max_context": 262_144, "engine_mib": 6_703.0 + 824 * 0.8357,
                "desktop_mib": desktop_mib, "reserve_mib": reserve_mib,
                "grade": "A (measured anchor)"}

    for dtype in ORDER:
        slope, _ = RING[dtype]
        fixed = c_of(RING, dtype, quant) - DESKTOP_MEASURED_MIB
        if logical <= max_pages(vram_gib, dtype, quant, desktop_mib, reserve_mib) \
                and fixed + slope * logical <= budget:
            return {"ok": True, "mode": "full", "dtype": dtype, "kv_capacity": context,
                    "pool_pages": logical, "host_kv_mib": 0, "max_context": context,
                    "engine_mib": round(fixed + slope * logical, 1), "desktop_mib": desktop_mib,
                    "reserve_mib": reserve_mib, "grade": "A(arithmetic)"}

    best = None
    for dtype in ORDER:
        slope, _ = RING[dtype]
        fixed = c_of(RING, dtype, quant) - DESKTOP_MEASURED_MIB
        pool = min(max_pages(vram_gib, dtype, quant, desktop_mib, reserve_mib), logical)
        if pool < MIN_POOL_PAGES:
            continue
        host_need = (logical - pool) * slope
        cand = {"ok": True, "mode": "ring", "dtype": dtype, "kv_capacity": pool * PAGE_TOKENS,
                "pool_pages": pool, "max_context": context,
                "host_kv_mib": int(math.ceil(host_need)) + 82,
                "host_kv_min_mib": round(host_need, 1),
                "engine_mib": round(fixed + slope * pool, 1),
                "desktop_mib": desktop_mib, "reserve_mib": reserve_mib,
                "grade": "A(arithmetic)"}
        if best is None or cand["pool_pages"] > best["pool_pages"]:
            best = cand
    if best is None:
        return {"ok": False,
                "why": f"{vram_gib} GiB cannot hold even {MIN_POOL_PAGES} KV pages: engine budget "
                       f"{budget:.0f} MiB, desktop {desktop_mib:.0f} MiB, reserve {reserve_mib:.0f} MiB",
                "grade": "A(arithmetic)", "desktop_mib": desktop_mib, "reserve_mib": reserve_mib}
    if host_offer_mib and host_offer_mib < best["host_kv_min_mib"]:
        return {"ok": False,
                "why": f"--host-kv-mib {int(host_offer_mib)} is below the guard: needs at least "
                       f"{best['host_kv_min_mib']:.0f} MiB for pool {best['pool_pages']} pages",
                "grade": "A(arithmetic)"}
    return best


def selftest() -> int:
    worst = 0.0
    print("[selftest] RING constants (k8v4 %.3f / %.0f):" % RING["k8v4"])
    slope, c = RING["k8v4"]
    for pages, measured in RING_POINTS:
        predicted = c + slope * pages
        err = abs(predicted - measured) / measured
        worst = max(worst, err)
        print("  pages=%-5d predicted=%-8.1f measured=%-8.1f err=%.2f%%"
              % (pages, predicted, measured, err * 100))
    print("[selftest] DENSE constants (k8v4 %.4f / %.0f):" % DENSE["k8v4"])
    slope_d, c_d = DENSE["k8v4"]
    for context, measured in DENSE_POINTS:
        pages = pages_for(context)
        predicted = c_d + slope_d * pages
        err = abs(predicted - measured) / measured
        worst = max(worst, err)
        print("  ctx=%-7d pages=%-5d predicted=%-8.1f measured=%-8.1f err=%.2f%%"
              % (context, pages, predicted, measured, err * 100))
    digest = hashlib.sha256(("%.4f" % worst).encode()).hexdigest()[:16]
    print("SELFTEST %s sha=%s worst_err=%.2f%%"
          % ("PASS" if worst <= 0.02 else "FAIL", digest, worst * 100))
    return 0 if worst <= 0.02 else 1


def main(argv) -> int:
    parser = argparse.ArgumentParser(description="KV pool planner (ring/dense constants).")
    parser.add_argument("--vram-gib", type=int)
    parser.add_argument("--quant", choices=["ptq1", "ptq2"])
    parser.add_argument("--context", type=int)
    parser.add_argument("--desktop-mib", type=float, default=DESKTOP_MIB)
    parser.add_argument("--reserve-mib", type=float, default=RESERVE_MIB)
    parser.add_argument("--host-kv-mib", type=float, default=0.0)
    parser.add_argument("--model", default="<model.ninfer>")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    if not (args.vram_gib and args.quant and args.context):
        parser.error("missing --vram-gib / --quant / --context")
    r = plan(args.vram_gib, args.quant, args.context, args.desktop_mib, args.reserve_mib,
             args.host_kv_mib)
    if not r["ok"]:
        print("REFUSED:", r["why"], "| grade:", r.get("grade"))
        for ctx in (262_144, 131_072, 65_536, 32_768, 16_384, 8_192, 4_096):
            t = plan(args.vram_gib, args.quant, ctx, args.desktop_mib, args.reserve_mib)
            if t["ok"]:
                print("minimal feasible: ctx=%d dtype=%s mode=%s kv_capacity=%d host_kv_mib=%d"
                      % (ctx, t["dtype"], t["mode"], t["kv_capacity"], t["host_kv_mib"]))
                break
        return 2
    print("plan: mode=%s dtype=%s kv_capacity=%d (pool %d pages, %%64=%d) host_kv_mib=%d "
          "host_kv_min=%.0f max_context=%d engine_mib=%.1f desktop_mib=%.0f reserve_mib=%.0f "
          "grade=%s"
          % (r["mode"], r["dtype"], r["kv_capacity"], r["pool_pages"], r["kv_capacity"] % 64,
             r["host_kv_mib"], r.get("host_kv_min_mib", 0), r["max_context"], r["engine_mib"],
             r["desktop_mib"], r["reserve_mib"], r["grade"]))
    print("command: ninfer-serve.exe %s --port 8080 --max-context %d --kv-dtype %s "
          "--kv-capacity %d --max-concurrency 1 --derive-session-keys%s"
          % (args.model, r["max_context"], r["dtype"], r["kv_capacity"],
             (" --host-kv-mib %d" % r["host_kv_mib"]) if r["mode"] == "ring" else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
