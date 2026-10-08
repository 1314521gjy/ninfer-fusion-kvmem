#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""8091 KVMem 输出窗口探针 —— 量出「当前实例单次回答能出多少 token」。

用法：
    python verify-output-window.py            # 默认 http://127.0.0.1:8091
    python verify-output-window.py 8091

判定口径：
  * finish_reason=length 且 completion_tokens ~= 4000  =>  输出租约卡在初始 4096 窗口（扩不动）
  * finish_reason=length 且 completion_tokens 明显 > 4096  =>  租约扩过了（继续到 max_tokens 上限）
  * finish_reason=stop    =>  自然收尾（没撞任何上限，本次不构成判定，可加大题面重试）

配合 /props 的 n_predict 与 /metrics 的 kv_cache_usage_ratio 一起看，
即可验证「prompt + output <= --kv-capacity 才扩得动」这条分界线。
"""
import json
import sys
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8091"
BASE = f"http://127.0.0.1:{PORT}"
KEY = "123456"


def _get(path):
    req = urllib.request.Request(BASE + path, headers={"Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def _post(payload):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def main():
    print("=" * 72)
    try:
        props = json.loads(_get("/props"))
        gen = props.get("default_generation_settings", {})
        print(f"model            : {props.get('model_alias')}")
        print(f"n_ctx            : {gen.get('n_ctx')}")
        print(f"n_predict(默认)   : {gen.get('params', {}).get('n_predict')}")
    except Exception as e:
        print(f"!! /props 读不到：{e}")

    pool, used = None, None
    try:
        for line in _get("/metrics").splitlines():
            if line.startswith("llamacpp:kv_cache_usage_ratio"):
                ratio = float(line.split()[-1])
            elif line.startswith("llamacpp:kv_cache_tokens"):
                used = int(float(line.split()[-1]))
        print(f"设备 KV 占用      : {used} token   ({ratio:.3%})")
        pool = int(round(used / ratio)) if ratio > 0 else None
        if pool:
            print(f"推算设备池容量    : ~{pool} token")
            print(f"当前空闲池        : ~{pool - used} token   <== 这就是输出租约能扩的上限")
    except Exception as e:
        print(f"!! /metrics 读不到：{e}")

    print("-" * 72)
    print("发一条「逼它长输出」的请求（关思维链，题面极小）：")
    payload = {
        "model": "Bonsai 27B KVMem",
        "messages": [{"role": "user",
                      "content": "Print the integers from 1 to 4000, space separated, nothing else."}],
        "reasoning_effort": "none",
        "max_tokens": 16384,
        "stream": False,
    }
    j = _post(payload)
    ch = j["choices"][0]
    u = j.get("usage", {})
    ct = u.get("completion_tokens")
    fr = ch.get("finish_reason")
    print(f"  finish_reason     : {fr}")
    print(f"  completion_tokens : {ct}")
    print(f"  prompt_tokens     : {u.get('prompt_tokens')}")
    print("-" * 72)
    if fr == "length" and ct is not None and ct > 8192:
        print("判定：租约扩到了 >8k —— 输出窗口正常。")
    elif fr == "length" and ct is not None:
        print(f"判定：卡在 {ct} token —— 输出租约只在初始窗口内（扩不动）。")
        print("      => 根因是 prompt + output 超过了 --kv-capacity。")
    else:
        print("判定：自然收尾，本次未撞上限（可重跑或加大题面）。")
    print("=" * 72)


if __name__ == "__main__":
    main()
