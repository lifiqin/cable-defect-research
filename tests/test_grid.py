import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cable_research.grid import combinations, main


class GridTests(unittest.TestCase):
    def test_baseline_is_not_duplicated_for_irrelevant_axes(self):
        rows = combinations(["baseline", "hybrid"], ["p3", "p3p4"], [0, 1], [42])
        self.assertEqual(len(rows), 5)
        self.assertEqual(sum(r["mode"] == "baseline" for r in rows), 1)
        self.assertEqual(sum(r["mode"] == "hybrid" for r in rows), 4)
        constrained = combinations(["baseline", "hybrid"], ["p3p4"], [1], [42])
        self.assertEqual([r["mode"] for r in constrained], ["baseline", "hybrid"])
        self.assertEqual((constrained[0]["sites"], constrained[0]["gate"]), ("p3", False))

    def test_duplicate_and_oversized_grids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            combinations(["tensor", "tensor"], ["p3"], [0], [42])
        with self.assertRaisesRegex(ValueError, "1 to 24"):
            combinations(["tensor"], ["p3", "p3p4"], [0, 1], list(range(7)))

    def test_execution_builds_safe_argument_lists_and_private_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            plan, weights = root / "plan.json", root / "weights.pt"
            plan.write_text("{}", encoding="utf-8")
            weights.write_bytes(b"test")
            output = root / "grid"
            args = ["grid", "--dataset", str(dataset), "--fold-plan", str(plan), "--weights", str(weights),
                    "--work", str(output), "--modes", "baseline", "hybrid", "--gates", "0", "1", "--execute"]
            with patch("sys.argv", args), patch("cable_research.grid.subprocess.run") as run, contextlib.redirect_stdout(io.StringIO()):
                main()
            self.assertEqual(run.call_count, 3)
            self.assertTrue((output / "grid_plan.json").is_file())
            commands = [c.args[0] for c in run.call_args_list]
            self.assertTrue(all(isinstance(c, list) and c[1:3] == ["-m", "cable_research.train"] for c in commands))
            self.assertEqual(sum("--channel-gate" in c for c in commands), 1)


if __name__ == "__main__":
    unittest.main()
