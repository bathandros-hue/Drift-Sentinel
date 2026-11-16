import csv
import tempfile
import unittest
from pathlib import Path

from driftsentinel import analyze, compare_column, load_csv, psi, jensen_shannon


class TestDrift(unittest.TestCase):
    def test_identical_distributions(self):
        self.assertAlmostEqual(psi([10, 20, 30], [10, 20, 30]), 0)
        self.assertAlmostEqual(jensen_shannon([1, 2, 3], [1, 2, 3]), 0)

    def test_numeric_shift(self):
        old = [str(i) for i in range(100)]
        new = [str(i + 100) for i in range(100)]
        report = compare_column(old, new, "sensor")
        self.assertEqual(report["kind"], "numeric")
        self.assertIn("distribution_shift", report["reasons"])
        self.assertGreater(report["psi"], 0.2)

    def test_constant_baseline_anomalies(self):
        result = compare_column(["5"] * 20, ["500"] * 20, "voltage")
        self.assertTrue(result["flagged"])

    def test_missing_values(self):
        report = compare_column(["A"] * 10, ["", "NULL", "A", ""] * 3, "category")
        self.assertIn("missingness_shift", report["reasons"])

    def test_unseen_category(self):
        report = compare_column(["A", "A", "B"], ["C", "C", "A"], "site")
        self.assertEqual(report["kind"], "categorical")
        self.assertIn("unseen_categories", report["reasons"])

    def test_invalid_numeric(self):
        report = compare_column(["1.0", "2.0", "3.0"], ["4", "oops", "5"], "mass")
        self.assertIn("invalid_numeric_values", report["reasons"])
        self.assertEqual(report["invalid_numeric_values"], 1)

    def test_csv_integration(self):
        with tempfile.TemporaryDirectory() as tmp:
            old, new = Path(tmp) / "old.csv", Path(tmp) / "new.csv"
            old.write_text("a,b\n1,A\n2,B\n3,A\n", encoding="utf-8")
            new.write_text("a,b\n10,A\n11,C\n12,C\n", encoding="utf-8")
            report = analyze(load_csv(old), load_csv(new))
            self.assertIn("a", report["flagged_columns"])
            self.assertIn("b", report["flagged_columns"])

    def test_schema_mismatch(self):
        with self.assertRaises(ValueError):
            analyze([{"a": "1"}], [{"b": "1"}])
        with self.assertRaises(ValueError):
            psi([1], [1, 2])
        with self.assertRaises(ValueError):
            compare_column([], ["1"], "a")


if __name__ == "__main__":
    unittest.main()
