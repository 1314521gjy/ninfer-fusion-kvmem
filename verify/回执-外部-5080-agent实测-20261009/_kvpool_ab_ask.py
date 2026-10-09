"""池子 A/B 的"题面 + 发问"部分。

做法（对应今天 14:00 那次失败的真实条件）：
  - 造一段 ~20.5K token 的长提示词（中段埋一句不常见的"密语"），**请求 1** 发出去；
  - **紧接着再发第二个请求**（同样带这段长题面，模拟"继续/追问"）——这才会走到 host-restore；
  - 记录：HTTP 状态、耗时、引擎日志里新增的 `materialize-one FAIL` / `capacity` / `adopt host-backed` 行。

用法：
  python _kvpool_ab_ask.py --port 8090 --tokens 20500 --tag B300 [--prompt-file 复用同一段题面]
"""
import argparse
import json
import os
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TD = r"C:\Users\wo739\Documents\ai\tools_dashboard"
NEEDLE = "紫电青霜-4729-鲸落"

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--tokens", type=int, default=20500, help="题面目标长度（按 ~2.2 字符/token 估算中文）")
ap.add_argument("--tag", default="ab")
ap.add_argument("--rounds", type=int, default=2)
args = ap.parse_args()

# ── 造题面：填充段落 + 中段密语 ───────────────────────────────────────────
FILLER = ("下面是某一批哈尔滨文创产品的工艺记录，逐条列出。"
          "产品编号与工序说明仅供参考，不需要你做任何统计，只要读完。\n")
def para(i):
    return ("记录 %04d：产品为金属冰箱贴，材质锌合金，表面电镀镍+烤漆，"
            "色号参照 PANTONE %dC，工艺顺序为冲压→抛光→电镀→喷漆→丝印，"
            "质检要点为边角无毛刺、色差 ΔE00 ≤ 2。\n" % (i, 200 + (i * 7) % 500))

chars_per_token = 2.2
target_chars = int(args.tokens * chars_per_token)
blocks, total = [], 0
i = 1
needle_at = int(target_chars * 0.6)
while total < target_chars:
    p = para(i)
    if total < needle_at <= total + len(p):
        p = p + "\n【要点】本批次的内部口令是：%s（这一句是唯一要点，其余都是背景）。\n\n" % NEEDLE
    blocks.append(p)
    total += len(p)
    i += 1
prompt = FILLER + "".join(blocks) + "\n读完后请只回答一句话：本批次的内部口令是什么？（不要复述其它内容）"

print("题面字符数 =", len(prompt), "≈ token", int(len(prompt) / chars_per_token), "｜密语埋点 ≈60% 处")

payload = {
    "model": "ninfer-ptq1",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 256,
    "stream": False,
}
body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

def ask(n):
    url = "http://127.0.0.1:%d/v1/chat/completions" % args.port
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            raw = r.read().decode("utf-8", "replace")
            dt = time.time() - t0
            try:
                j = json.loads(raw)
                txt = (j.get("choices") or [{}])[0].get("message", {}).get("content", "")
                usage = j.get("usage")
            except Exception:
                txt, usage = raw[:300], None
            hit = NEEDLE in (txt or "")
            print("  请求%d: HTTP %s | %.1fs | 命中密语=%s | usage=%s" % (n, r.status, dt, hit, usage))
            print("        正文前 200 字:", (txt or "")[:200].replace("\n", " "))
            return r.status, hit
    except urllib.error.HTTPError as e:
        dt = time.time() - t0
        detail = e.read().decode("utf-8", "replace")[:300]
        print("  请求%d: HTTP %s | %.1fs | 错误体: %s" % (n, e.code, dt, detail.replace("\n", " ")))
        return e.code, False
    except Exception as e:
        print("  请求%d: 异常 %s" % (n, e))
        return 0, False

log = os.path.join(TD, "infer-%s.out.log" % args.tag)
before = os.path.getsize(log) if os.path.exists(log) else 0

results = []
for n in range(1, args.rounds + 1):
    results.append(ask(n))

after = os.path.getsize(log) if os.path.exists(log) else 0
print("\n=== 引擎日志新增片段关键行（%s，%d → %d B）===" % (log, before, after))
if os.path.exists(log):
    with open(log, "r", encoding="utf-8", errors="replace") as f:
        f.seek(before)
        chunk = f.read()
    keys = ("capacity |", "adopt host-backed", "materialize-one FAIL", "HTTP 429", "HTTP 500", "worker crash",
            "req#", "exceeds the resident", "ERROR", "FATAL")
    shown = 0
    for line in chunk.splitlines():
        if any(k in line for k in keys):
            print("  ", line[:200])
            shown += 1
            if shown > 25:
                print("   ...（更多见日志）")
                break
print("\n结论:", "两轮都成功" if all(s == 200 for s, _ in results) else "有失败（见上）",
      "｜密语命中:", [h for _, h in results])
