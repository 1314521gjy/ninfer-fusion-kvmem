#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""8091 `--thorough-admission-search` A/B 判定脚本。

为什么不能只看"重启后输出变长了"：
    刚重启时设备池是空的 → 输出租约本来就能扩 → 假阳性。
    必须**先把池灌满到 99%**，再看输出租约还能不能扩。这才是真判别。

本脚本做四件事：
  1. 确认引擎已起，并从 requests.jsonl 的 server_start.argv 里**确认新旗标真的生效**；
  2. 记录初始池占用；
  3. 灌池：连发大题面请求，把设备池吃到接近满；
  4. 施压测量：发一条"逼它长输出"的请求，回报输出上限；
     并从日志读回该请求的 materialization（search_granted_ns / search_stop_phase / stop_reason）。

对照基线（2026-10-07 改旗标前，实例 serve-33372，池 65536）：
    池占用            kv_cache_tokens 65024 / 65536  (99.22%, 空闲 ~512)
    施压测量          completion_tokens 4056 / finish_reason length   ← 硬顶
    评估器            search_granted_ns 5,000,000 / search_stop_phase expansion
                      / stop_reason insufficient_expected_gain

判读：
    * 若 search_granted_ns 明显 > 5e6（到千万级/亿级）→ 旗标生效。
    * 若施压测量仍卡在 ~4056          → 旗标生效但救不了（该试 --recency-eviction）。
    * 若施压测量 > 8k                → 成功，输出墙被打开。
"""
import json
import subprocess
import sys
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8091"
BASE = f"http://127.0.0.1:{PORT}"
KEY = "123456"
LOG = r"C:\Bonsai_NInfer_KVMem\logs\requests.jsonl"
BIG = [r"C:\Bonsai_NInfer_KVMem\README-NInfer-KVMem.md",
       r"C:\Bonsai_NInfer_KVMem\docs\4070Ti-SUPER-16G-测试报告-20261002.md",
       r"C:\Bonsai_NInfer_KVMem\docs\他人日志-可调项清单.md",
       r"C:\Bonsai_NInfer_KVMem\docs\回执-20261002-4070TiS-16G-超池正中针.md"]
ASK = "\n\n=== 任务 ===\nPrint the integers from 1 to 4000, space separated, nothing else."


def get(path):
    r = urllib.request.Request(BASE + path, headers={"Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(r, timeout=30) as f:
        return f.read().decode("utf-8", "replace")


def post(payload, timeout=900):
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json",
                                        "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read().decode("utf-8", "replace"))


def metrics():
    d = {}
    try:
        for l in get("/metrics").splitlines():
            if l.startswith("llamacpp:kv_cache_usage_ratio"):
                d["ratio"] = float(l.split()[-1])
            if l.startswith("llamacpp:kv_cache_tokens"):
                d["tokens"] = int(float(l.split()[-1]))
    except Exception as e:
        d["err"] = str(e)
    return d


def tail_events(fname_pred, n=1):
    out = []
    try:
        with open(LOG, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                if fname_pred(j):
                    out.append(j)
    except Exception:
        pass
    return out[-n:]


def wait_up(limit=20):
    t0 = time.time()
    while time.time() - t0 < limit:
        try:
            get("/health")
            return True
        except Exception:
            time.sleep(3)
    return False


def main():
    print("=" * 74, flush=True)
    if not wait_up():
        print(f"!! {PORT} 还没起来。请先双击 start-kvmem-8091.cmd（我不代你启动）。", flush=True)
        print("   —— 引擎起来后重跑本脚本即可。", flush=True)
        return
    print(f"{PORT} 已就绪", flush=True)

    starts = tail_events(lambda j: j.get("event") == "server_start", 1)
    if not starts:
        print("!! 读不到 server_start，跳过旗标确认")
    else:
        argv = starts[0].get("argv", [])
        has = "--thorough-admission-search" in argv
        print(f"[1] 旗标已生效？ {'✅ 是' if has else '❌ 否 —— 请检查启动器是否被回退/改错'}")
        print("    argv 片段:", " ".join(a for a in argv if "search" in a or "lease" in a) or "(无)")
        if not has:
            print("    => 旗标没生效，后面的测量无意义。停。")
            return

    m0 = metrics()
    print(f"[2] 初始池占用: {m0.get('tokens')} token, 比例 {m0.get('ratio')}")

    # 灌池必须把池顶到 >=95%，否则"输出当然扩得动"，测量无效（第一版就栽在这）。
    # 做法：题面按 1x/2x/4x 递增，每次只给 64 token 输出（省时间），直到占用率达标。
    base = "".join(open(p, encoding="utf-8", errors="replace").read() + "\n\n" for p in BIG)
    print(f"[3] 灌池中（基准题面 {len(base)} 字符，向上级数复用以顶满池）...")
    for mult in (1, 2, 4):
        txt = base * mult
        j = post({"model": "Bonsai 27B KVMem", "messages": [{"role": "user", "content": txt}],
                  "max_tokens": 64, "reasoning_effort": "none"})
        u = j.get("usage", {})
        mm = metrics()
        print(f"    {mult}x: prompt={u.get('prompt_tokens')} pool={mm.get('tokens')} "
              f"比例={mm.get('ratio'):.4f}")
        if (mm.get("ratio") or 0) >= 0.95:
            break
    m1 = metrics()
    print(f"[4] 灌池后池占用: {m1.get('tokens')} token, 比例 {m1.get('ratio'):.4f}")
    if (m1.get("ratio") or 0) < 0.95:
        print("    ⚠️ 还没到 95%，本次施压结果仅供参考（池没满，输出本来就能扩）。")

    print("[5] 施压测量（题面极小 + 要求 16384 输出，看租约能不能扩）...")
    j = post({"model": "Bonsai 27B KVMem",
              "messages": [{"role": "user", "content": "Print the integers from 1 to 4000, space separated, nothing else."}],
              "max_tokens": 16384, "reasoning_effort": "none"})
    u, ch = j.get("usage", {}), j["choices"][0]
    ct, fr = u.get("completion_tokens"), ch.get("finish_reason")
    print(f"    prompt={u.get('prompt_tokens')}  completion={ct}  finish_reason={fr}")

    rec = tail_events(lambda x: x.get("event") == "request_done", 1)
    if rec:
        m = rec[0].get("materialization", {})
        print(f"[6] 评估器: search_granted_ns={m.get('search_granted_ns')} "
              f"search_elapsed_ns={m.get('search_elapsed_ns')} "
              f"stop_phase={m.get('search_stop_phase')} stop_reason={m.get('stop_reason')} "
              f"degradation_units={m.get('selected_degradation_units')}")

    print("-" * 74)
    if ct and ct > 8192:
        print(f"判定: ✅ 成功 —— 池满时输出仍扩到 {ct}。输出墙被打开。")
    elif ct:
        print(f"判定: ⚠️ 仍卡在 {ct}。旗标生效但不足以腾出空间。")
        print("      下一步候选: --recency-eviction / --value-aware-demote（把常驻 KV 降级到 Host）。")
    print("=" * 74)


if __name__ == "__main__":
    main()
