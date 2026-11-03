#!/usr/bin/env python3
"""Exercise the real gateway CLI against a local HTTP recorder, without a model."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

GATEWAY = Path(__file__).resolve().parents[1] / "tools" / "agg_gateway.py"


class GatewayShardCapTest(unittest.TestCase):
    def run_gateway(self, material, *, mode="aggregate", cap=512, pool=4096):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                requests.append((self.path, json.loads(self.rfile.read(
                    int(self.headers["Content-Length"])).decode("utf-8"))))
                response = json.dumps({"choices": [{"message": {"content": "1"}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, *args):
                pass

        with HTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    source = Path(tmp) / "material.txt"
                    source.write_text(material, encoding="utf-8")
                    command = [sys.executable, str(GATEWAY), "--prompt-file", str(source),
                               "--question", "Count the records", "--pool-tokens", str(pool),
                               "--est-factor", "1", "--mode", mode,
                               "--base-url", "http://127.0.0.1:%d" % server.server_port]
                    if cap is not None:
                        command += ["--max-shard-tokens", str(cap)]
                    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
            finally:
                server.shutdown()
                thread.join(timeout=5)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for path, body in requests:
            self.assertEqual(path, "/v1/chat/completions")
            self.assertEqual(body["max_tokens"], 512)
            # These fixtures leave enough room for the actual question and prompt wrapper.
            # This is an estimator contract test, not a model-tokenizer or GPU guarantee.
            prompt = body["messages"][0]["content"]
            self.assertLessEqual(len(prompt.encode("utf-8")) // 4, pool)
        return result.stdout, requests

    def assert_fragments(self, requests, material, separator, cap):
        fragments = []
        for _, body in requests:
            prompt = body["messages"][0]["content"]
            self.assertIn("\n\nMATERIAL:\n", prompt)
            shard = prompt.split("\n\nMATERIAL:\n", 1)[1]
            self.assertLessEqual(len(shard.encode("utf-8")) // 4, cap)
            fragments.append(shard)
        self.assertEqual(separator.join(fragments), material)

    def test_line_packing_respects_lower_cap(self):
        material = "\n".join(letter * 1200 for letter in "abc")
        _, requests = self.run_gateway(material)
        self.assert_fragments(requests, material, "\n", 512)
        self.assertEqual(len(requests), 3)

    def test_paragraph_packing_respects_lower_cap(self):
        material = "\n\n".join(letter * 1200 for letter in "abc")
        _, requests = self.run_gateway(material)
        self.assert_fragments(requests, material, "\n\n", 512)
        self.assertEqual(len(requests), 3)

    def test_locating_does_not_bypass_multiple_shards_when_material_fits_pool(self):
        material = "\n\n".join(letter * 1200 for letter in "abc")
        _, requests = self.run_gateway(material, mode="locating")
        self.assert_fragments(requests, material, "\n\n", 512)
        self.assertEqual(len(requests), 3)

    def test_oversize_structural_unit_refuses_without_http(self):
        output, requests = self.run_gateway("x" * 2400)
        self.assertIn('"sum_method": "unit_broken"', output)
        self.assertEqual(requests, [])

    def test_cjk_units_are_packed_without_splitting(self):
        units = [letter * 200 for letter in "甲乙丙"]
        _, requests = self.run_gateway("\n\n".join(units))
        actual = [body["messages"][0]["content"].split("\n\nMATERIAL:\n", 1)[1]
                  for _, body in requests]
        # Every 200-character CJK unit is 400 estimated tokens at factor=1.
        self.assertEqual(actual, units)

    def test_mixed_width_oversize_unit_refuses_without_http(self):
        output, requests = self.run_gateway("x" * 1600 + "界" * 400)
        self.assertIn('"sum_method": "unit_broken"', output)
        self.assertEqual(requests, [])

    def test_locating_whitespace_does_not_bypass_cap(self):
        material = "x" * 1200 + "\n\n" + " " * 4000
        _, requests = self.run_gateway(material, mode="locating")
        self.assert_fragments(requests, "x" * 1200, "\n\n", 512)

    def test_exact_cap_is_allowed(self):
        material = "x" * 2048
        _, requests = self.run_gateway(material)
        self.assert_fragments(requests, material, "\n", 512)
        self.assertEqual(len(requests), 1)

    def test_default_cap_is_respected_with_larger_pool(self):
        material = "a" * 80000 + "\n\n" + "b" * 80000
        _, requests = self.run_gateway(material, cap=None, pool=65536)
        self.assert_fragments(requests, material, "\n\n", 32768)
        self.assertEqual(len(requests), 2)

    def test_single_locating_request_stays_direct(self):
        _, requests = self.run_gateway("short material", mode="locating")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0][1]["messages"][0]["content"],
                         "short material\n\nCount the records")

    def test_pool_remains_the_limit_when_below_cap(self):
        material = "a" * 12000 + "\n\n" + "b" * 12000
        _, requests = self.run_gateway(material, cap=8192, pool=4096)
        self.assert_fragments(requests, material, "\n\n", 4096)
        self.assertEqual(len(requests), 2)


if __name__ == "__main__":
    unittest.main()
