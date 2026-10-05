#!/usr/bin/env python3
"""Synthetic HTTP regression for non-ring, shared-KV admission and prefix reuse.

Run against an otherwise idle server with --max-context 262144 --kv-capacity
262144 --max-concurrency 4 and KVMem disabled. No real conversations are needed.
Credentials are read from a file and are never included in the report.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request


def document(lane, rows):
    codes = [f"MOCK{lane}A713", f"MOCK{lane}B829", f"MOCK{lane}C947"]
    lines = [f"Item {i:04d}: ordinary sample data without a relevant key." for i in range(rows)]
    for slot, fraction in enumerate((0.27, 0.57, 0.87)):
        lines[int(rows * fraction)] = f"The required access code for slot {slot + 1} is {codes[slot]}."
    return (f"Synthetic document {lane}. Read the document.\n" + "\n".join(lines) +
            "\nReturn only the three required access codes in slot order, separated by commas."), codes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--long-rows", type=int, default=15700)
    parser.add_argument("--concurrent-rows", type=int, default=6800)
    parser.add_argument("--phase", choices=("all", "long", "concurrent"), default="all")
    args = parser.parse_args()
    key = args.api_key_file.read_text().strip() if args.api_key_file else ""
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key

    def request(messages, max_tokens, barrier=None):
        payload = {"model": args.model, "messages": messages, "max_tokens": max_tokens,
                   "temperature": 0, "reasoning_effort": "none", "stream": True,
                   "stream_options": {"include_usage": True}}
        if barrier:
            barrier.wait(timeout=30)
        start = time.monotonic()
        result = {"max_tokens": max_tokens, "content": "", "done": False}
        req = urllib.request.Request(args.url.rstrip("/") + "/v1/chat/completions",
                                     json.dumps(payload).encode(), headers)
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as response:
                result["status"] = response.status
                for raw in response:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        result["done"] = True
                        break
                    event = json.loads(data)
                    if event.get("error"):
                        result["error"] = event["error"]
                    if event.get("usage"):
                        result["usage"] = event["usage"]
                    if event.get("timings"):
                        result["timings"] = event["timings"]
                    for choice in event.get("choices", []):
                        text = choice.get("delta", {}).get("content") or ""
                        if text and "ttft_s" not in result:
                            result["ttft_s"] = time.monotonic() - start
                        result["content"] += text
                        if choice.get("finish_reason"):
                            result["finish_reason"] = choice["finish_reason"]
        except urllib.error.HTTPError as error:
            # Do not copy arbitrary gateway error bodies into a public report.
            result.update(status=error.code, error="HTTPError")
        except Exception as error:
            result["error"] = type(error).__name__
        result["elapsed_s"] = time.monotonic() - start
        result["passed"] = (result.get("status") == 200 and result["done"] and
                            not result.get("error") and bool(result["content"]))
        return result

    results = {}
    def save(label, value):
        results[label] = value
        args.report.write_text(json.dumps(results, indent=2) + "\n")
        summary = [{k: r.get(k) for k in ("status", "passed", "hits", "usage", "ttft_s")}
                   for r in (value if isinstance(value, list) else [value])]
        print(label, json.dumps(summary), flush=True)

    save("smoke", request([{"role": "user", "content": "Reply with 42 only."}], 32))
    if not results["smoke"]["passed"]:
        raise SystemExit(1)
    if args.phase in ("all", "long"):
        prompt, codes = document(17, args.long_rows)
        answer = request([{"role": "user", "content": prompt}], 256)
        answer["hits"] = sum(code in answer["content"] for code in codes)
        answer["passed"] &= (answer["hits"] == 3 and
                             250000 <= answer.get("usage", {}).get("prompt_tokens", 0) <= 262144)
        save("near_256k", answer)
    if args.phase in ("all", "concurrent"):
        docs = [document(lane, args.concurrent_rows) for lane in range(4)]
        histories = [[{"role": "user", "content": prompt}] for prompt, _ in docs]
        for turn in range(2):
            barrier = threading.Barrier(4)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(request, history, 32768, barrier) for history in histories]
                answers = [future.result() for future in futures]
            for i, answer in enumerate(answers):
                answer["hits"] = sum(code in answer["content"] for code in docs[i][1])
                answer["passed"] &= answer["hits"] == 3
                histories[i] += [{"role": "assistant", "content": answer["content"]},
                                 {"role": "user", "content": "Repeat the three codes in slot order, nothing else."}]
            save(f"concurrent_turn_{turn + 1}", answers)
        # Continue the most recently completed session while its checkpoint is still warm.
        lane = max(range(4), key=lambda i: answers[i]["elapsed_s"])
        answer = request(histories[lane], 32768)
        answer["hits"] = sum(code in answer["content"] for code in docs[lane][1])
        cached = answer.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
        answer["passed"] &= answer["hits"] == 3 and cached > 0
        save("serial_cached_followup", answer)
    all_results = [item for value in results.values()
                   for item in (value if isinstance(value, list) else [value])]
    raise SystemExit(0 if all(item["passed"] for item in all_results) else 1)


if __name__ == "__main__":
    main()
