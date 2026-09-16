"""Regression checks for frozen values, mirror direction and coverage claims."""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))  # comtrade_check lives in Code/data/
import comtrade_check as check

ROOT = Path(os.environ.get("COMTRADE_TEST_ROOT", Path(__file__).resolve().parents[2]))
SUPPLEMENT = Path(os.environ.get("COMTRADE_TEST_SUPPLEMENT", ROOT / "Data/manual/comtrade_supplement_2026-09-15"))


class ComtradeCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = pd.read_csv(ROOT / "Data/manual/comtrade_crude_2709_api.csv")
        cls.fetch = staticmethod(check.make_offline_fetch(supplement=SUPPLEMENT))

    def test_integrated_evidence_and_missing_cells(self):
        status, evidence, summary, _ = check.assess(self.main, self.fetch)
        self.assertEqual(check.exit_code(status), 2)
        self.assertEqual(len(evidence), 156)
        self.assertEqual(evidence.source.eq("reporter").sum(), 111)
        self.assertEqual(evidence.source.eq("mirror_observed").sum(), 43)
        gaps = evidence[evidence.observed_usd.isna()]
        self.assertEqual(set(zip(gaps.iso, gaps.flow, gaps.year)), {("KWT", "M", 2019), ("KWT", "M", 2023)})
        self.assertTrue(summary.sign_matches.all())
        tur = summary.set_index("iso").loc["TUR"]
        self.assertAlmostEqual(tur.observed_exports_usd, 9560085184.354, places=2)
        self.assertAlmostEqual(tur.observed_imports_usd, 28583584200.556, places=2)
        self.assertAlmostEqual(tur.observed_net_exports_usd, -19023499016.202, places=2)

    def test_reporter_priority_prevents_double_counting_and_revision(self):
        _, evidence, _, _ = check.assess(self.main, self.fetch)
        selected = evidence.set_index(["iso", "flow", "year"])
        for cell in [("JPN", "X", 2021), ("JPN", "X", 2022), ("MEX", "M", 2019)]:
            self.assertEqual(selected.loc[cell, "source"], "reporter")
        self.assertAlmostEqual(selected.loc[("JPN", "M", 2023), "observed_usd"], 80884609985.472, places=2)

    def test_numeric_partner_rows_never_establish_full_coverage(self):
        def fetch(flow, partner, year):
            frame = self.fetch(flow, partner, year)
            if frame is not None and frame.empty:
                frame = self.fetch(flow, partner, 2020).iloc[:1].copy()
                frame["refYear"] = year
                frame["primaryValue"] = 0.0
            return frame
        status, evidence, _, _ = check.assess(self.main, fetch)
        self.assertFalse(evidence.observed_usd.isna().any())
        self.assertEqual(check.exit_code(status), 2)

    def test_reported_zero_and_missing_cell_remain_distinct(self):
        row = self.main[self.main.reporterISO.eq("KWT")].iloc[:1].copy()
        row["flowCode"], row["refYear"], row["primaryValue"] = "M", 2019, 0.0
        main = pd.concat([self.main, row], ignore_index=True)
        _, evidence, _, _ = check.assess(main, self.fetch)
        selected = evidence.set_index(["iso", "flow", "year"])
        self.assertEqual(selected.loc[("KWT", "M", 2019), "source"], "reporter")
        self.assertEqual(selected.loc[("KWT", "M", 2019), "observed_usd"], 0.0)
        self.assertTrue(pd.isna(selected.loc[("KWT", "M", 2023), "observed_usd"]))

    def test_duplicate_or_reversed_mirror_rows_are_rejected(self):
        for corruption in ["duplicate", "reverse"]:
            with self.subTest(corruption=corruption):
                def fetch(flow, partner, year):
                    frame = self.fetch(flow, partner, year)
                    if (flow, partner, year) == ("M", "792", 2019):
                        frame = pd.concat([frame, frame.iloc[:1]]) if corruption == "duplicate" else frame.assign(flowCode="X")
                    return frame
                status, _, _, errors = check.assess(self.main, fetch)
                self.assertTrue(status.startswith("INCOMPLETE:"))
                self.assertTrue(errors)

    def test_duplicate_reporter_is_not_replaced_by_mirror(self):
        main = pd.concat([self.main, self.main[self.main.refYear.eq(2019)].iloc[:1]])
        status, evidence, _, _ = check.assess(main, self.fetch)
        self.assertTrue(status.startswith("INCOMPLETE:"))
        self.assertEqual(evidence.source.eq("rejected").sum(), 1)

    def test_invalid_partner_value_stays_partial(self):
        frame = self.fetch("M", "792", 2019)
        frame.loc[frame.index[0], "primaryValue"] = float("inf")
        value, state, _, _ = check.observe(frame, "M", "792", 2019)
        self.assertEqual(state, "mirror_partial")
        self.assertGreater(value, 0)

    def test_wrong_observed_sign_fails(self):
        def fetch(flow, partner, year):
            frame = self.fetch(flow, partner, year)
            if (flow, partner, year) == ("M", "792", 2019):
                frame = frame.copy()
                frame.loc[frame.index[0], "primaryValue"] = 1e12
            return frame
        status, _ = check.run_check(self.main, fetch, out=lambda line: None)
        self.assertTrue(status.startswith("FAIL:"))
        self.assertEqual(check.exit_code(status), 1)

    def test_saved_response_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            meta = json.loads((SUPPLEMENT / "request_manifest.json").read_text())[0]
            (folder / "raw").mkdir()
            (folder / "request_manifest.json").write_text(json.dumps([meta]))
            raw = folder / "raw" / (meta["label"] + ".json")
            shutil.copy2(SUPPLEMENT / "raw" / raw.name, raw)
            raw.write_bytes(raw.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                check.load_supplement(folder)

    def test_live_wrapper_uses_opposite_flow_and_total_dimensions(self):
        calls = []
        def pull(**kwargs):
            calls.append(kwargs)
            return pd.DataFrame()
        fetch = check.make_api_fetch(pull, sleep=0)
        check.run_check(self.main, fetch, out=lambda line: None)
        tur = [row for row in calls if row["partnerCode"] == "792"]
        self.assertEqual(len(tur), 12)
        self.assertEqual({row["flowCode"] for row in tur}, {"M", "X"})
        self.assertTrue(all(row["customsCode"] == "C00" and row["partner2Code"] == "0" and row["motCode"] == "0" for row in calls))


if __name__ == "__main__":
    unittest.main()
