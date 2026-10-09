"""第 2 步（真实夹具版）：用 10-05 那次真实翻车的形态做对照。

夹具：**约 12 万 token 的题面**（= 当时那条 `prompt 121,218` 的量级），把"要干的活"和一条可验证要点埋在中段。
两臂只差池子：
  A（现状 280 页 = 17,920）→ 题面是池子的 ~6.7 倍，必然 host-back + 检索
  B（34176 页预算）        → 题面仍是池子的 ~3.5 倍，但设备侧宽得多
量什么（照 ActKV §3.2 的机理，但用我们能测的代理）：
  ① 输出字符数 / 是否只吐一句就停（"吐几句话说要干活然后没啦"）
  ② **重复率**（把输出切成 40 字窗口，看有多少窗口重复出现）
  ③ 是否出现"复述上下文"（输出里出现题面里的原句片段）
  ④ 延迟（TTFT / total）、以及引擎日志里有没有 429 / 池告警
"""
import argparse, collections, json, os, re, sys, time, urllib.request, urllib.error

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TD = r"C:\Users\wo739\Documents\ai\tools_dashboard"
NEEDLE = "紫电青霜-4729-鲸落"
MARK = "【要点】内部口令："

FILLER = ("以下是某印刷厂的产线日志，逐条记录，不需要你统计，只需读完后按【要点】的授权执行。\n")

def build(prompt_chars):
    blocks, total, i = [], 0, 1
    where = int(prompt_chars * 0.6)
    while total < prompt_chars:
        p = ("记录 %06d：哈尔滨文创冰箱贴，锌合金电镀镍+烤漆，PANTONE %dC，"
             "工序 冲压→抛光→电镀→喷漆→丝印，质检 ΔE00 ≤ 2，批次 %d。\n" % (i, 200 + (i * 7) % 500, 100000 + i))
        if total < where <= total + len(p):
            p += "\n%s%s（凭这一句你就已被授权执行下一步。）\n\n" % (MARK, NEEDLE)
        blocks.append(p); total += len(p); i += 1
    task = ("\n读完了。现在只回答**三行**：\n第一行写内部口令；\n第二行写你打算执行的动作（一句话）；\n第三行写你读完这批日志后判断的最主要风险（一句话）。\n不要复述日志内容。")
    return FILLER + "".join(blocks) + task

def post(port, prompt, max_tokens, label, timeout=1800):
    url = "http://127.0.0.1:%d/v1/chat/completions" % port
    body = json.dumps({"model": "ninfer-ptq1",
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "stream": False}, ensure_ascii=False).encode()
    t0 = time.time()
    try:
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            j = json.loads(r.read().decode("utf-8", "replace"))
            u = j.get("usage") or {}
            txt = (j.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
            dt = time.time() - t0
            win = 40
            chunks = [txt[i:i + win] for i in range(0, max(1, len(txt) - win), win)]
            dup = sum(1 for c, n in collections.Counter(chunks).items() if n > 1 and c.strip())
            echo = len(re.findall(re.escape("记录 0"), txt))
            print("  %-22s HTTP %s | %5.1fs | out=%4s tok | 正文 %5d 字 | 重复窗口 %d/%d | 复述日志 %d 处 | 口令=%s"
                  % (label, r.status, dt, u.get("completion_tokens"), len(txt), dup, max(1, len(chunks)),
                     echo, NEEDLE in txt))
            print("      前 160 字：", txt[:160].replace("\n", " ⏎ "))
            return {"status": r.status, "secs": round(dt, 1), "chars": len(txt), "dup": dup,
                    "echo": echo, "needle": NEEDLE in txt, "text": txt}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:200].replace("\n", " ")
        print("  %-22s HTTP %s | %.1fs | %s" % (label, e.code, time.time() - t0, detail))
        return {"status": e.code, "error": detail}
    except Exception as e:
        print("  %-22s 异常 %s" % (label, e))
        return {"status": 0, "error": str(e)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--chars", type=int, default=265000, help="题面字符数（≈ token 数 /2.2 ⇒ 26.5 万字符 ≈ 12 万 token）")
    ap.add_argument("--max-tokens", type=int, default=2000)
    args = ap.parse_args()

    prompt = build(args.chars)
    print("题面字符 = %d ≈ %d token（夹具文件见同目录 _fixture_120k.txt）" % (len(prompt), int(len(prompt) / 2.2)))
    fx = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fixture_120k.txt")
    if not os.path.exists(fx):
        with open(fx, "w", encoding="utf-8") as f:
            f.write(prompt)

    log = os.path.join(TD, "infer-%s.out.log" % args.tag)
    before = os.path.getsize(log) if os.path.exists(log) else 0
    r1 = post(args.port, prompt, args.max_tokens, "① 长夹具 第一发")
    r2 = post(args.port, prompt, args.max_tokens, "② 长夹具 第二发(复用)")

    after = os.path.getsize(log) if os.path.exists(log) else 0
    print("\n=== 引擎日志新增关键行 ===")
    if os.path.exists(log):
        with open(log, "r", encoding="utf-8", errors="replace") as f:
            f.seek(before); chunk = f.read()
        keys = ("exceeds the resident", "adopt host-backed", "materialize-one FAIL", "HTTP 429", "HTTP 500",
                "worker crash", "req#", "kvmem_score: SELECT", "ERROR ", "FATAL")
        n = 0
        for line in chunk.splitlines():
            if any(k in line for k in keys):
                print("  ", line[:190]); n += 1
                if n > 26:
                    print("   ..."); break

if __name__ == "__main__":
    main()
