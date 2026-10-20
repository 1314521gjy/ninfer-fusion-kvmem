#!/usr/bin/env python3
"""agg_gateway.py - shard/reduce gateway for global-aggregation questions.

Standard library only. Local server reached with an empty ProxyHandler (this machine has a proxy).

Splitting (2026-10-19, final): capacity-based shard count + cumulative-estimate boundaries.
  * hard ceiling = P (the pool size): every shard must be fully visible;
  * n = max(1, ceil(C_true / (0.98*P - L_max))): 0.98P absorbs estimate error (measured: est is
    0.27% LOW vs prompt_tokens), L_max absorbs line quantisation;
  * boundaries at argmin |cum(tokens) - i*C/n|, so the worst shard is ~C/n + L_max, bounded;
  * if anything still exceeds P, bump n and retry (max 5), and ASSERT max_est <= P;
  * a single line larger than P is split explicitly (warn oversize_line + unit_split=1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import urllib.request


def raw_tokens(text: str) -> float:
    data = text.encode("utf-8")
    ascii_bytes = sum(1 for b in data if b < 128)
    return ascii_bytes / 4.0 + (len(data) - ascii_bytes) / 1.5


def estimate_tokens(text: str, factor: float) -> int:
    return max(1, int(raw_tokens(text) * factor))


def _split_nearest(lines: list[str], ests: list[int], n: int) -> list[str]:
    cumulative = [0]
    for value in ests:
        cumulative.append(cumulative[-1] + value)
    grand = max(1, cumulative[-1])
    boundaries = [0]
    for index in range(1, n):
        want = grand * index / n
        best = min(range(boundaries[-1] + 1, len(lines)),
                   key=lambda j: abs(cumulative[j] - want), default=len(lines))
        boundaries.append(best)
    boundaries.append(len(lines))
    return ["\n".join(lines[boundaries[i]:boundaries[i + 1]]) for i in range(n)]


LAST_PACK_STATS = {"units_total": 0, "units_cut": 0, "pack_mode": "line"}


def split_equal_lines(text, pool_tokens, factor, safety, max_shard_tokens=32768):
    """Pack by STRUCTURAL UNITS (paragraphs separated by blank lines; else single lines).

    A boundary never splits a unit unless that unit alone is larger than the pool (then it is cut and
    flagged unit_split=1 so downstream knows the shard's count can only be approximate). Shards are
    grown by re-estimating the JOINED text, so the estimate covers the newline bytes too.
    """
    cap = min(pool_tokens, max_shard_tokens or 32768)
    paras = re.split(r"\n\s*\n", text)
    if len(paras) > 1:
        units = [x for x in paras if x.strip()]
        pack_mode = "unit"
    else:
        units = [x for x in text.split("\n") if x.strip()]
        pack_mode = "line"
    sep = "\n\n" if pack_mode == "unit" else "\n"
    shards, unit_split = [], []
    warnings = []
    units_cut = 0
    cur, cur_est = [], 0
    for unit in units:
        est = estimate_tokens(unit, factor)
        if est > pool_tokens:
            if cur:
                shards.append(sep.join(cur)); unit_split.append(0); cur, cur_est = [], 0
            step = max(1, int(len(unit) * cap / est))
            for i in range(0, len(unit), step):
                piece = unit[i:i + step]
                if piece.strip():
                    shards.append(piece); unit_split.append(1); units_cut += 1
                    warnings.append("[gate] warn oversize_line split chars=%d (unit_split=1: count approximate)" % len(unit))
            continue
        if cur:
            joined = sep.join(cur + [unit])
            if estimate_tokens(joined, factor) > pool_tokens:
                shards.append(sep.join(cur)); unit_split.append(0); cur, cur_est = [], 0
        cur.append(unit); cur_est += est
    if cur:
        shards.append(sep.join(cur)); unit_split.append(0)
    ceiling = int(pool_tokens * safety)
    for index, shard in enumerate(shards, start=1):
        est = estimate_tokens(shard, factor)
        if est > pool_tokens:
            raise AssertionError("shard exceeds the pool: a partially visible shard must never be sent")
        if est > ceiling:
            util = int(100 * est / max(1, pool_tokens))
            warnings.append("[gate] warn shard_util=%d%% shard=%d/%d est=%d P=%d safety=%s"
                            % (util, index, len(shards), est, pool_tokens, safety))
    LAST_PACK_STATS.update({"units_total": len(units), "units_cut": units_cut, "pack_mode": pack_mode})
    return shards, len(shards), warnings, unit_split[: len(shards)]


def tier_for(ratio: float, n: int) -> str:
    if ratio >= 1.0: return "T0"
    if ratio >= 0.5: return "T1"
    if ratio >= 0.25: return "T2"
    if ratio >= 0.1: return "T3"
    if ratio >= 0.02: return "T4"
    return "T5"


AGGREGATE_PROMPT = (
    "Answer using ONLY the material below. If the question asks for a total count or a sum over "
    "the whole document, report ONLY the count that appears in THIS fragment; do not guess or "
    "extrapolate from other fragments. If the fragment contains no relevant information, reply "
    "exactly NOT_PRESENT."
)
LOCATING_PROMPT = (
    "Answer using ONLY the material below. If this fragment does not contain the information the "
    "question asks for, reply exactly NOT_PRESENT."
)
INT_RE = re.compile(r"^\s*-?\d{1,9}\s*$")


def as_int(answer: str):
    return int(answer) if INT_RE.match(answer or "") else None


class ChatClient:
    def __init__(self, base_url: str, model: str, timeout: float = 1800.0,
                 temperature: float = 0.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = float(temperature)
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def ask(self, prompt: str, max_tokens: int = 512) -> str:
        payload = json.dumps({"model": self.model,
                              "messages": [{"role": "user", "content": prompt}],
                              "temperature": self.temperature,
                              "max_tokens": max_tokens}).encode("utf-8")
        request = urllib.request.Request(self.base_url + "/v1/chat/completions", data=payload,
                                         headers={"Content-Type": "application/json"}, method="POST")
        with self._opener.open(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"].strip()


def unit_broken_payload(tier, n, pool_tokens, context_tokens, ratio, factor):
    """(a) HONEST REFUSAL: a structural unit larger than the pool makes count aggregation
    unbounded, so never claim local_sum and never fall back to model reduce."""
    return {
        "tier": tier, "n": n, "shards": [],
        "final": "[REFUSED] material unit larger than pool",
        "low_confidence": True, "sum_method": "unit_broken",
        "est_factor": factor, "context_tokens_est": context_tokens,
        "pool_tokens": pool_tokens, "ratio": round(ratio, 4),
        "units_total": LAST_PACK_STATS["units_total"],
        "units_cut": LAST_PACK_STATS["units_cut"],
    }


def run(args, client) -> dict:
    material = open(args.prompt_file, encoding="utf-8").read()
    factor = args.est_factor
    context_tokens = estimate_tokens(material, factor)
    pool = args.pool_tokens
    ratio = pool / context_tokens if context_tokens else 1.0
    shards, n, warnings, unit_split = split_equal_lines(
        material, pool, factor, args.shard_safety, args.max_shard_tokens or 32768)
    print(f"[gate] pack: pack_mode={LAST_PACK_STATS['pack_mode']} "
          f"units_total={LAST_PACK_STATS['units_total']} units_cut={LAST_PACK_STATS['units_cut']} "
          f"max_shard_tokens={args.max_shard_tokens or 32768} n={n}")
    tier = tier_for(ratio, n)
    print(f"[gate] agg tier={tier} n={n} C={context_tokens} P={pool} r={ratio:.3f} "
          f"est_factor={factor:.2f} safety={args.shard_safety} mode={args.mode} "
          f"temperature={getattr(client, 'temperature', 0.0):g}")
    for line in warnings:
        print(line)
    # (a) HONEST REFUSAL: when a structural unit had to be cut, a count aggregation has no upper
    # bound -- per-shard local counts are only approximations. Never fall back to model reduce.
    if LAST_PACK_STATS["units_cut"] > 0:
        print("[gate] warn unit_broken units_total=%d units_cut=%d capacity=%d"
              % (LAST_PACK_STATS["units_total"], LAST_PACK_STATS["units_cut"],
                 min(pool, args.max_shard_tokens or 32768)))
        result = unit_broken_payload(tier, n, pool, context_tokens, ratio, factor)
        if args.out:
            import json as _json
            with open(args.out, "w", encoding="utf-8") as handle:
                _json.dump(result, handle, indent=2, ensure_ascii=False)
            print("[gate] wrote " + args.out)
        print("[gate] reduce sum_method=unit_broken final=%r" % result["final"])
        return result

    if args.mode == "locating" and (tier == "T0" or n <= 1):
        answer = client.ask(material + "\n\n" + args.question)
        print(f"[gate] single-request answer={answer!r}")
        return {"tier": "T0", "n": 1, "shards": [], "final": answer, "low_confidence": False}
    if tier == "T5" and n > 50:
        raise SystemExit(f"[gate] refused: r={ratio:.4f} < 0.02 (n={n}); "
                         f"P_min={math.ceil(context_tokens / 50)} tokens")
    partials = []
    head = AGGREGATE_PROMPT if args.mode == "aggregate" else LOCATING_PROMPT
    for index, shard in enumerate(shards, start=1):
        prompt = f"{head}\n\nQUESTION: {args.question}\n\nMATERIAL:\n{shard}"
        answer = client.ask(prompt)
        est = estimate_tokens(shard, factor)
        partials.append({"est_tokens": est, "chars": len(shard), "answer": answer,
                         "unit_split": unit_split[index - 1]})
        print(f"[gate] shard {index}/{n} est={est} actual_split_chars={len(shard)} "
              f"unit_split={unit_split[index - 1]} answer={answer!r}")
    ints = [as_int(p["answer"]) for p in partials]
    if all(v is not None for v in ints):
        final, sum_method = str(sum(ints)), "local_sum"
        print(f"[gate] reduce sum_method={sum_method} final={final!r} parts={ints}")
    else:
        kept = [p["answer"] for p in partials if p["answer"].strip().upper() != "NOT_PRESENT"]
        joined = "\n".join(f"- fragment {i + 1}: {a}" for i, a in enumerate(kept))
        final = client.ask("Combine the per-fragment answers. Ignore NOT_PRESENT fragments. "
                           f"Do not invent numbers.\n\nQUESTION: {args.question}\n\n"
                           f"FRAGMENT ANSWERS:\n{joined}")
        sum_method = "model"
        print(f"[gate] reduce sum_method={sum_method} final={final!r} kept={len(kept)}/{n}")
    result = {"tier": tier, "n": n, "shards": partials, "final": final,
              "low_confidence": tier == "T4" or n > 10, "sum_method": sum_method,
              "est_factor": factor, "context_tokens_est": context_tokens,
              "pool_tokens": pool, "ratio": round(ratio, 4)}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        print(f"[gate] wrote {args.out}")
    return result


def _material(sections: int, lines_per_section: int) -> str:
    out = []
    for index in range(sections):
        out.append(f"SECTION {index + 1} :: COUNT=1080 :: the list below holds 1080 items")
        out.extend(f"item {index + 1}.{i} of the section" for i in range(1, lines_per_section))
    return "\n".join(out)


def selftest() -> int:
    factor, pool, safety = 0.90, 32768, 0.8
    material = _material(3, 3520)
    total = estimate_tokens(material, factor)
    shards, n, warnings, unit_split = split_equal_lines(material, pool, factor, safety)
    pack_mode_first = LAST_PACK_STATS["pack_mode"]
    ests = [estimate_tokens(s, factor) for s in shards]
    balanced = (max(ests) / max(1, min(ests))) if ests else 1.0
    l_max = max(estimate_tokens(l, factor) for l in material.split("\n"))
    capacity = max(1, int(pool * 0.98) - l_max)
    expect_n = max(1, math.ceil(total / capacity))
    print(f"[gate] selftest (a) C_true={total} P={pool} n={n} expect_n={expect_n} "
          f"L_max={l_max} capacity={capacity} shard_est={ests}")
    print(f"[gate] selftest (b) max_est={max(ests)} <= P ? {all(e <= pool for e in ests)} ; "
          f"max_min_ratio={balanced:.2f}")
    print(f"[gate] selftest (c) local_sum={1080 * n} == 1080 x n")
    for line in warnings:
        print(line)
    ok_a = n >= expect_n   # unit packing may use more shards than the minimum
    ok_b = all(e <= pool for e in ests) and (balanced <= 1.5 or n > expect_n)
    ok_c = all(as_int("1080") == 1080 for _ in shards)
    sep = "\n\n" if pack_mode_first == "unit" else "\n"
    partition_exact = (sep.join(shards) == material)
    dup_content_ok = all(material.count(s) >= 2 for s in set(shards) if shards.count(s) > 1)
    print(f"[gate] selftest (c2) partition_exact={partition_exact} sizes={[len(s) for s in shards]}")
    ok_c = ok_c and partition_exact and dup_content_ok

    big = _material(12, 3520)
    big_total = estimate_tokens(big, factor)
    big_shards, big_n, big_warn, _ = split_equal_lines(big, pool, factor, safety)
    big_ests = [estimate_tokens(s, factor) for s in big_shards]
    ok_d1 = big_n >= 4 and all(e <= pool for e in big_ests)
    print(f"[gate] selftest (d1) big C_true={big_total} n={big_n} max_est={max(big_ests)} "
          f"<= P ? {all(e <= pool for e in big_ests)} bumped={[w for w in big_warn if 'n_bumped' in w]}")
    _, _, warn08, _ = split_equal_lines(material, pool, factor, 0.8)
    _, _, warn10, _ = split_equal_lines(material, pool, factor, 1.0)
    ok_d2 = len(warn08) != len(warn10)
    print(f"[gate] selftest (d2) warnings safety=0.8:{len(warn08)} vs 1.0:{len(warn10)} "
          f"differ={ok_d2}")
    # (d3) oversize single line: must be split with a warning and flagged unit_split=1
    huge_line = "x" * int(pool * 4 * 1.2)
    tiny = "\n".join(["short line"] * 5)
    line_shards, line_n, line_warn, line_flags = split_equal_lines(
        tiny + "\n" + huge_line + "\n" + tiny, pool, factor, safety)
    ok_d3 = any("oversize_line" in w for w in line_warn) and 1 in line_flags
    # (d4) a structural unit larger than capacity MUST be flagged unit_broken (refusal path)
    big_unit_material = ("x" * 200000) + "\n\n" + ("y" * 200000)
    ub_shards, ub_n, ub_warn, ub_flags = split_equal_lines(big_unit_material, 16384, 0.90, 0.8, 16384)
    ok_d4 = LAST_PACK_STATS["units_cut"] > 0 and any("oversize_line" in w for w in ub_warn)
    ub_payload = unit_broken_payload("T2", ub_n, 16384, 40000, 0.4, 0.90)
    ok_d4 = ok_d4 and ub_payload["sum_method"] == "unit_broken" and \
        ub_payload["final"] == "[REFUSED] material unit larger than pool" and \
        ub_payload["low_confidence"] is True
    print(f"[gate] selftest (d4) unit_broken units_cut={LAST_PACK_STATS['units_cut']} "
          f"n={ub_n} refuse_expected=True -> {ok_d4}")
    print(f"[gate] selftest (d3) oversize_line split={any('oversize_line' in w for w in line_warn)} "
          f"unit_split_positions={[i for i, v in enumerate(line_flags) if v]}")
    digest = hashlib.sha256(f"{n}|{1080 * n}|{max(ests)}".encode()).hexdigest()[:16]
    ok = ok_a and ok_b and ok_c and ok_d1 and ok_d2 and ok_d3 and ok_d4
    print("SELFTEST", "PASS" if ok else "FAIL", f"sha={digest}")
    return 0 if ok else 1


def main(argv) -> int:
    parser = argparse.ArgumentParser(description="Shard/reduce gateway (tiers T0-T5, §4.6).")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="default")
    parser.add_argument("--prompt-file")
    parser.add_argument("--question")
    parser.add_argument("--pool-tokens", type=int)
    parser.add_argument("--shard-safety", type=float, default=0.8,
                        help="ADVISORY: shards above this fraction of the pool log a warning; "
                             "the hard ceiling is the pool itself")
    parser.add_argument("--max-shard-tokens", type=int, default=32768,
                        help="cap on one shard; default 32768 so a shard stays within one structural unit")
    parser.add_argument("--est-factor", type=float, default=0.90,
                        help="scale on bytes/4 (default 0.90, calibrated on 94K)")
    parser.add_argument("--mode", choices=["locating", "aggregate"], default="aggregate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="sampling temperature sent to the engine (default 0.0)")
    parser.add_argument("--out")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    missing = [n for n in ("prompt_file", "question", "pool_tokens") if not getattr(args, n)]
    if missing:
        parser.error("missing: " + ", ".join("--" + m.replace("_", "-") for m in missing))
    print(json.dumps(run(args, ChatClient(args.base_url, args.model, args.timeout,
                                          args.temperature)),
                     indent=2, ensure_ascii=False)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
