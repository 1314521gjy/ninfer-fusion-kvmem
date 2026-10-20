#!/usr/bin/env python3
"""accept_8g.py - end-to-end acceptance for the 8 GiB profile.

Six streaming cases go through the chat gateway (default :8081) and the script
judges the run against evidence collected from the engine log, the gateway log,
GPU memory sampling and host free memory.

Exit code 0 = GREEN, 1 = RED.  Nothing is retried and no parameter is changed.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import re
import subprocess
import sys
import threading
import time
import urllib.request

FILLER = None


def load_filler(path: str) -> list[str]:
    global FILLER
    with open(path, encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    FILLER = lines
    return lines


def material(chars: int) -> str:
    """Repeat filler lines (ASCII) until roughly `chars` characters."""
    assert FILLER, "filler not loaded"
    out: list[str] = []
    n = 0
    i = 0
    while n < chars:
        line = FILLER[i % len(FILLER)]
        out.append(line)
        n += len(line) + 1
        i += 1
    return "\n".join(out)


def stream_request(url: str, body: dict, timeout: float = 600.0) -> dict:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    t0 = time.time()
    chunks: list[str] = []
    code = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = int(resp.status)
            for raw in resp:
                chunks.append(raw.decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:  # type: ignore[attr-defined]
        code = int(exc.code)
        chunks.append(exc.read().decode("utf-8", "replace"))
    except Exception as exc:  # noqa: BLE001 - report, never retry
        chunks.append(f"EXC:{exc}")
    return {"code": code, "text": "".join(chunks), "secs": round(time.time() - t0, 1)}


def tail(path: str, start: int) -> list[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except FileNotFoundError:
        return []
    return lines[start:]


def count_lines(path: str) -> int:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return len(fh.read().splitlines())
    except FileNotFoundError:
        return 0


def engine_prompts(lines: list[str]) -> list[int]:
    out = []
    for ln in lines:
        m = re.search(r"req#\d+ done .*?prompt ([\d,]+)", ln)
        if m:
            out.append(int(m.group(1).replace(",", "")))
    return out


def gateway_levels(lines: list[str]) -> list[tuple[str, str]]:
    out = []
    for ln in lines:
        m = re.search(r"\d\d:\d\d:\d\d est=(\d+) pool_usable=(\d+) ratio=([\d.]+) level=(\S+) est_after=(\d+)(.*)", ln)
        if m:
            out.append((m.group(4), ln.strip()))
    return out


def gpu_used() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip().splitlines()[0]
        return int(out.strip())
    except Exception:  # noqa: BLE001
        return -1


class _MemStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def host_free_gib() -> float:
    stat = _MemStatus()
    stat.dwLength = ctypes.sizeof(_MemStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
    return round(stat.ullAvailPhys / (1024 ** 3), 2)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default="http://127.0.0.1:8081")
    ap.add_argument("--engine-log", default=r"E:\infer-build\kv-stream\accept8g.engine.log")
    ap.add_argument("--gateway-log", default=r"E:\infer-build\kv-stream\accept8g.gateway.log")
    ap.add_argument("--filler", default=r"E:\infer-build\kv-stream\long-prompt-27b.txt")
    ap.add_argument("--pool", type=int, default=15616, help="engine --kv-capacity")
    ap.add_argument("--idle-mib", type=int, default=-1)
    ap.add_argument("--host-min-gib", type=float, default=20.0)
    ap.add_argument("--engine-max-mib", type=int, default=7442)
    args = ap.parse_args(argv)

    load_filler(args.filler)
    if args.idle_mib < 0:
        args.idle_mib = gpu_used()

    peak = {"mib": 0}
    stop = threading.Event()

    def sampler() -> None:
        while not stop.is_set():
            u = gpu_used()
            if u > peak["mib"]:
                peak["mib"] = u
            time.sleep(0.4)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    tools62 = [
        {
            "type": "function",
            "function": {
                "name": f"tool_{i:02d}",
                "description": f"Tool number {i} used by the acceptance harness.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
                    "required": ["path"],
                },
            },
        }
        for i in range(62)
    ]

    cases: list[tuple[str, dict]] = []
    # A: short question
    cases.append(("A_short", {
        "model": "j27b",
        "messages": [{"role": "user", "content": "你好，1+1 等于几？只回答数字。"}],
        "max_tokens": 64, "stream": True,
    }))
    # B: 30 messages multi-turn
    msgs_b = []
    for i in range(15):
        msgs_b.append({"role": "user", "content": f"第 {i + 1} 轮：请记住编号 {i + 1}。"})
        msgs_b.append({"role": "assistant", "content": f"已记住编号 {i + 1}。"})
    cases.append(("B_30msgs", {"model": "j27b", "messages": msgs_b,
                                "max_tokens": 64, "stream": True}))
    # C: 62 tools + 4 messages
    cases.append(("C_62tools", {
        "model": "j27b",
        "messages": [
            {"role": "system", "content": "You are a tool-using agent."},
            {"role": "user", "content": "先列出有哪些工具，再说明你会先调用哪一个。"},
            {"role": "assistant", "content": "好的。"},
            {"role": "user", "content": "只回答工具数量加一行的下一步。"},
        ],
        "tools": tools62, "max_tokens": 64, "stream": True,
    }))
    # D: single 47K-token long text
    long_d = material(47_000 * 4)
    cases.append(("D_47kTok", {
        "model": "j27b",
        "messages": [{"role": "user", "content": long_d + "\n\n材料最后一行以什么词开头？只回答该词。"}],
        "max_tokens": 64, "stream": True,
    }))
    # E: 64K-token history + tools
    hist = []
    for i in range(16):
        hist.append({"role": "user", "content": material(4_000 * 4)})
        hist.append({"role": "assistant", "content": f"已读第 {i + 1} 段。"})
    hist.append({"role": "user", "content": "基于以上历史，回答：你一共读过几段？只回答数字。"})
    cases.append(("E_64kHist", {"model": "j27b", "messages": hist, "tools": tools62[:8],
                                "max_tokens": 64, "stream": True}))
    # F: 130K-char system prompt
    sys_f = material(130_000)
    cases.append(("F_130kSys", {
        "model": "j27b",
        "messages": [
            {"role": "system", "content": sys_f},
            {"role": "user", "content": "系统提示里的材料是什么风格？一句话回答。"},
        ],
        "max_tokens": 64, "stream": True,
    }))

    e_pos = count_lines(args.engine_log)
    g_pos = count_lines(args.gateway_log)
    rows = []
    for name, body in cases:
        res = stream_request(args.gateway, body)
        elines = tail(args.engine_log, e_pos)
        glines = tail(args.gateway_log, g_pos)
        e_pos += len(elines)
        g_pos += len(glines)
        prompts = engine_prompts(elines)
        lv = gateway_levels(glines)
        has_finish = "finish_reason" in res["text"]
        has_done = "[DONE]" in res["text"]
        rows.append({
            "case": name,
            "code": res["code"],
            "secs": res["secs"],
            "finish": has_finish,
            "done": has_done,
            "prompts": prompts,
            "levels": [x[0] for x in lv],
            "level_lines": [x[1] for x in lv],
        })
        print(f"case {name}: http={res['code']} finish={has_finish} DONE={has_done} "
              f"secs={res['secs']} engine_prompt={prompts} gateway_level={[x[0] for x in lv]}")

    time.sleep(1.0)
    stop.set()
    th.join(timeout=3)
    elines_all = tail(args.engine_log, 0)
    glines_all = tail(args.gateway_log, 0)

    all_elines = tail(args.engine_log, e_pos - len(elines_all) if False else 0)
    prompts_all = engine_prompts(elines_all)
    exceeds = sum(1 for ln in elines_all if "exceeds the resident" in ln)
    bad = {
        "crash": sum(1 for ln in elines_all if "worker crash" in ln),
        "FATAL": sum(1 for ln in elines_all if "FATAL" in ln),
        "cannot_free": sum(1 for ln in elines_all if "cannot free Device KV pages" in ln),
        "ownership": sum(1 for ln in elines_all if "ownership" in ln),
        "monotonic": sum(1 for ln in elines_all if "not monotonic" in ln),
    }
    levels: dict[str, int] = {}
    for lv, _ in gateway_levels(glines_all):
        levels[lv] = levels.get(lv, 0) + 1
    segs = []
    for ln in glines_all:
        m = re.search(r"L2s segments=(\d+)", ln)
        if m:
            segs.append(int(m.group(1)))

    peak_mib = peak["mib"]
    engine_peak = peak_mib - args.idle_mib
    host_free = host_free_gib()

    problems: list[str] = []
    for r in rows:
        if r["code"] != 200:
            problems.append(f"{r['case']}: http={r['code']}")
        if not r["finish"]:
            problems.append(f"{r['case']}: finish_reason missing")
        if not r["done"]:
            problems.append(f"{r['case']}: [DONE] missing")
        if not r["prompts"]:
            problems.append(f"{r['case']}: no engine prompt line")
        for p in r["prompts"]:
            if p > args.pool:
                problems.append(f"{r['case']}: engine prompt {p} > pool {args.pool}")
        if not r["levels"]:
            problems.append(f"{r['case']}: no gateway level line")
    if not prompts_all:
        problems.append("engine log: no prompt lines at all")
    if exceeds:
        problems.append(f"engine 'exceeds the resident' lines = {exceeds}")
    for k, v in bad.items():
        if v:
            problems.append(f"engine {k} lines = {v}")
    if not any(lv in levels for lv in ("L1", "L2", "L2s")):
        problems.append(f"gateway levels only {levels}")
    if engine_peak > args.engine_max_mib:
        problems.append(f"engine_peak {engine_peak} MiB > {args.engine_max_mib}")
    if host_free < args.host_min_gib:
        problems.append(f"host_free {host_free} GiB < {args.host_min_gib}")

    verdict = "GREEN" if not problems else "RED(" + "; ".join(problems) + ")"

    print()
    print("EVIDENCE:")
    print(f"  engine_lines={len(elines_all)} prompt_tokens_all={prompts_all}")
    print(f"  exceeds_the_resident={exceeds} bad={bad}")
    print(f"  gateway_levels={levels} L2s_segments={segs}")
    print(f"  idle_gpu_mib={args.idle_mib} peak_gpu_mib={peak_mib} engine_peak={engine_peak} "
          f"(limit {args.engine_max_mib}) host_free_gib={host_free}")
    for r in rows:
        for ln in r["level_lines"]:
            print(f"  gw[{r['case']}] {ln}")
    print()
    print("ACCEPT=" + verdict)
    return 0 if verdict == "GREEN" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
