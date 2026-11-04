"""CPU-only regressions for the planner's measured-anchor budget guards."""
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "kv_plan.py"
spec = importlib.util.spec_from_file_location("kv_plan", SCRIPT)
kv_plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kv_plan)


class PlannerBudgetTests(unittest.TestCase):
    def cli(self, *args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--vram-gib", "8", "--quant", "ptq1",
             "--context", "262144", *args], text=True, capture_output=True,
        )

    def test_default_anchor_is_preserved(self):
        result = kv_plan.plan(8, "ptq1", 262144)
        self.assertTrue(result["ok"])
        self.assertEqual(result["pool_pages"], 824)
        self.assertEqual(result["host_kv_mib"], 2816)
        self.assertEqual(result["grade"], "A (measured anchor)")

    def test_larger_reserve_recalculates_pool(self):
        result = kv_plan.plan(8, "ptq1", 262144, reserve_mib=600)
        self.assertTrue(result["ok"])
        self.assertLess(result["pool_pages"], 824)
        self.assertLessEqual(result["engine_mib"] + 300 + 600, 8192)

    def test_impossible_reserve_is_refused_by_cli(self):
        result = self.cli("--reserve-mib", "2000")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("REFUSED:", result.stdout)
        self.assertNotIn("command:", result.stdout)

    def test_insufficient_host_offer_is_refused_by_cli(self):
        result = self.cli("--host-kv-mib", "1")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("needs at least 2816 MiB", result.stdout)
        self.assertNotIn("command:", result.stdout)

    def test_host_offer_boundary(self):
        for amount, expected in [(2815, False), (2816, True), (4096, True)]:
            with self.subTest(amount=amount):
                self.assertEqual(kv_plan.plan(8, "ptq1", 262144, host_offer_mib=amount)["ok"], expected)

    def test_default_cli_still_emits_anchor_command(self):
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--kv-capacity 52736", result.stdout)
        self.assertIn("--host-kv-mib 2816", result.stdout)


if __name__ == "__main__":
    unittest.main()
