"""Regression checks for Code/event_study/10_prose_checks.py.

Every test builds a throw-away copy of the checker, the shared helper, the frozen inputs
and the four LaTeX files it reads in a temporary folder, runs the script there as a
subprocess (exactly as `python3 Code/event_study/10_prose_checks.py` is run by hand) and
checks the exit code and the report. Nothing in the repository is touched.

The baseline tree is the live thesis plus FIXTURE_FIXES, the corrections of known stale
numbers that are still pending in the text (each applied only if its old string is
present, so the fixture stays valid once Leon pastes the fix). `test_baseline_passes`
therefore states what the checker must say about a thesis whose numbers are right;
every mutation test starts from that baseline and changes one thing.

Run from the repo root:
    python -m unittest discover -s Code/tests -p 'test_prose_checks.py'

Environment overrides:
    PROSE_TEST_ROOT     repo root (default: two levels up from this file)
    PROSE_TEST_SCRIPT   checker under test
                        (default: <root>/Code/event_study/10_prose_checks.py; while the
                        script is still a proposal, point this at
                        .handoff/proposed_code/10_prose_checks.py)
"""
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(os.environ.get("PROSE_TEST_ROOT", Path(__file__).resolve().parents[2]))
SCRIPT = Path(os.environ.get("PROSE_TEST_SCRIPT", ROOT / "Code/event_study/10_prose_checks.py"))
CONTENT = Path("Main/LaTeX Thesis/content")
TEX = ["05_data_methodology.tex", "06_results.tex", "07_discussion.tex",
       "tab_se_sensitivity.tex", "tab_es_dollar_horizon.tex"]
DATA = ["Data/processed/returns_daily.csv", "Data/processed/daily_panel.csv",
        "Data/processed/events.csv", "Data/processed/event_study/cross_event_diff.csv",
        "Code/common/es_common.py"]
N_VALUES, N_PASSAGES = 37, 10
SUCCESS = "RESULT: All selected thesis values match the recomputed results."

# Known stale numbers in the live thesis, corrected in the fixture only (old, new, file).
FIXTURE_FIXES = [
    ("06_results.tex",
     r"with a cumulative abnormal return of $-0.64\%$ ($t=-0.63$) over the twenty trading days",
     r"with a cumulative abnormal return of $-0.63\%$ ($t=-0.63$) over the twenty trading days"),
]


class ProseChecksTests(unittest.TestCase):
    """One pristine baseline tree per class, a fresh copy per test."""

    @classmethod
    def setUpClass(cls):
        cls._base = tempfile.TemporaryDirectory(prefix="prose_checks_base_")
        base = Path(cls._base.name)
        for rel in DATA:
            (base / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / rel, base / rel)
        (base / CONTENT).mkdir(parents=True)
        for f in TEX:
            shutil.copy(ROOT / CONTENT / f, base / CONTENT / f)
        (base / "Code/event_study").mkdir(parents=True)
        shutil.copy(SCRIPT, base / "Code/event_study/10_prose_checks.py")
        for f, old, new in FIXTURE_FIXES:
            p = base / CONTENT / f
            s = p.read_text(encoding="utf-8")
            if old in s:
                assert s.count(old) == 1, (f, old)
                p.write_text(s.replace(old, new), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls._base.cleanup()

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="prose_checks_test_")
        self.tmp = Path(self._tmp.name) / "repo"
        shutil.copytree(self._base.name, self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    # ---- helpers ------------------------------------------------------------------
    def run_script(self):
        r = subprocess.run([sys.executable, "Code/event_study/10_prose_checks.py"],
                           cwd=self.tmp, capture_output=True, text=True,
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        return r.returncode, r.stdout + r.stderr

    def tex(self, name):
        return self.tmp / CONTENT / name

    def edit(self, name, old, new, count=1):
        p = self.tex(name)
        s = p.read_text(encoding="utf-8")
        self.assertEqual(s.count(old), count, f"{name}: anchor {old!r} found {s.count(old)} times")
        p.write_text(s.replace(old, new), encoding="utf-8")

    def rows(self, name="prose_checks.csv"):
        with open(self.tmp / "Output/tables/prose_checks" / name, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    # ---- baseline -----------------------------------------------------------------
    def test_baseline_passes(self):
        code, out = self.run_script()
        self.assertEqual(code, 0, out)
        self.assertIn(SUCCESS, out)
        self.assertIn(f"{N_VALUES} values in {N_PASSAGES} passages", out)
        for name in ("Table 6.4 row 1", "Section 5.1.1 holiday footnote",
                     "Section 6.3 Ukraine anticipation caveat", "Section 6.5 IEEPA disclosure",
                     "Section 7.1 asymmetry sentence", "Section 7.5 strikes sentence",
                     "Table 6.1 note, ceasefire sentence"):
            self.assertIn(name, out)
        rows = self.rows()
        self.assertEqual(len(rows), N_VALUES)
        self.assertTrue(all(r["match"] == "True" for r in rows))
        self.assertEqual(len({r["passage"] for r in rows}), N_PASSAGES)

    def test_thesis_values_come_from_the_tex_not_from_the_script(self):
        src = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("PRINTED", src)          # no list of expected thesis values in the script
        self.run_script()
        rows = self.rows()
        # the thesis column holds the strings as they stand in the LaTeX, e.g. "six", "+1.09"
        thesis = {r["quantity"]: r["thesis"] for r in rows}
        self.assertEqual(thesis["ceasefire_days_on_or_after_onset"], "six")
        self.assertEqual(thesis["ukr_ar0"], "+1.09")
        self.assertEqual(thesis["hormuz_est_end"], "30~January 2026")

    # ---- a changed number fails, correcting it restores agreement -------------------
    def test_changed_number_fails_and_names_the_passage(self):
        self.edit("tab_se_sensitivity.tex", r"& $-7.24$ & $1.76$ & $-4.11$ & $1\%$", r"& $-7.24$ & $1.76$ & $-4.12$ & $1\%$")
        code, out = self.run_script()
        self.assertEqual(code, 1)
        self.assertRegex(out, r"MISMATCH\s+Table 6\.4 row 1 \(estimation-window sd\): t64_t1 printed -4\.12")
        self.assertNotIn(SUCCESS, out)
        bad = [r for r in self.rows() if r["match"] == "False"]
        self.assertEqual([(r["passage"], r["quantity"]) for r in bad],
                         [("Table 6.4 row 1 (estimation-window sd)", "t64_t1")])

    def test_correcting_the_number_restores_agreement(self):
        self.edit("tab_se_sensitivity.tex", r"& $-7.24$ & $1.76$ & $-4.11$ & $1\%$", r"& $-7.24$ & $1.76$ & $-4.12$ & $1\%$")
        self.assertEqual(self.run_script()[0], 1)
        self.edit("tab_se_sensitivity.tex", r"& $-7.24$ & $1.76$ & $-4.12$ & $1\%$", r"& $-7.24$ & $1.76$ & $-4.11$ & $1\%$")
        code, out = self.run_script()
        self.assertEqual(code, 0, out)
        self.assertIn(SUCCESS, out)

    def test_the_known_ukraine_error_is_caught(self):
        # the live thesis printed -0.64 on 2026-09-20; the fixture holds -0.63
        self.edit("06_results.tex", r"cumulative abnormal return of $-0.63\%$ ($t=-0.63$)",
                  r"cumulative abnormal return of $-0.64\%$ ($t=-0.63$)")
        code, out = self.run_script()
        self.assertEqual(code, 1)
        self.assertIn("ukr_car_m20_m1 printed -0.64, recomputed -0.634522", out)

    def test_wrong_rejection_level_fails(self):
        self.edit("tab_se_sensitivity.tex", r"& $2.99$ & $-2.42$ & $5\%$", r"& $2.99$ & $-2.42$ & $1\%$")
        code, out = self.run_script()
        self.assertEqual(code, 1)
        self.assertIn("t64_lvl3 printed 1, recomputed 5", out)

    def test_wrong_date_fails(self):
        self.edit("06_results.tex", "closes on 30~January 2026", "closes on 31~January 2026")
        code, out = self.run_script()
        self.assertEqual(code, 1)
        self.assertIn("hormuz_est_end printed 31~January 2026, recomputed 2026-01-30", out)

    def test_word_numbers(self):
        self.edit("tab_es_dollar_horizon.tex", "contains the first six trading days", "contains the first seven trading days")
        code, out = self.run_script()
        self.assertEqual(code, 1)
        self.assertIn("ceasefire_days_on_or_after_onset printed seven, recomputed 6", out)
        self.edit("tab_es_dollar_horizon.tex", "contains the first seven trading days", "contains the first 6 trading days")
        code, out = self.run_script()
        self.assertEqual(code, 0, out)

    # ---- a missing, duplicated or unreadable passage cannot pass ----------------------
    def test_removed_passage_fails_as_missing(self):
        self.edit("07_discussion.tex",
                  " The estimation window of the first contains 40 of its 120 trading days on or after the 2~March onset of the Hormuz crisis, so its benchmark would be contaminated, and",
                  " And")
        code, out = self.run_script()
        self.assertEqual(code, 1)
        self.assertRegex(out, r"MISSING\s+Section 7\.5 strikes sentence in 07_discussion\.tex")
        self.assertNotIn(SUCCESS, out)
        # the other passages are still reported, so the failure is visible, not silent
        self.assertEqual(len(self.rows()), N_VALUES - 2)

    def test_removed_table_row_fails_as_missing(self):
        self.edit("tab_se_sensitivity.tex", r"Both corrections & $-7.24$ & $3.24$ & $-2.23$ & $5\%$ \\" + "\n", "")
        code, out = self.run_script()
        self.assertEqual(code, 1)
        self.assertRegex(out, r"MISSING\s+Table 6\.4 row 4 \(both corrections\) in tab_se_sensitivity\.tex")

    def test_duplicated_passage_fails_as_ambiguous(self):
        sentence = "The ceasefire's estimation window ends on 9~March and contains the first six trading days of the crisis, so its cells are read with corresponding caution."
        self.edit("tab_es_dollar_horizon.tex", sentence, sentence + " " + sentence)
        code, out = self.run_script()
        self.assertEqual(code, 1)
        self.assertRegex(out, r"AMBIGUOUS \(2 matches\)\s+Table 6\.1 note, ceasefire sentence in tab_es_dollar_horizon\.tex")

    def test_missing_file_fails(self):
        self.tex("tab_se_sensitivity.tex").unlink()
        code, out = self.run_script()
        self.assertEqual(code, 1)
        self.assertIn("UNREADABLE", out)
        self.assertNotIn(SUCCESS, out)

    # ---- only the active LaTeX is read -------------------------------------------------
    def test_commented_out_passage_is_not_read(self):
        p = self.tex("07_discussion.tex")
        s = p.read_text(encoding="utf-8")
        line = next(l for l in s.split("\n") if "contains 40 of its 120 trading days" in l)
        p.write_text(s.replace(line, "% " + line), encoding="utf-8")
        code, out = self.run_script()
        self.assertEqual(code, 1)
        self.assertRegex(out, r"MISSING\s+Section 7\.5 strikes sentence")

    def test_old_number_in_a_comment_line_is_ignored(self):
        p = self.tex("05_data_methodology.tex")
        s = p.read_text(encoding="utf-8")
        line = next(l for l in s.split("\n") if "excluding zero returns" in l)
        stale = "% " + line.replace("to $-4.03$", "to $-9.99$")
        p.write_text(s.replace(line, stale + "\n" + line), encoding="utf-8")
        code, out = self.run_script()
        self.assertEqual(code, 0, out)          # neither ambiguous nor a mismatch

    def test_trailing_comment_is_ignored_and_escaped_percent_is_kept(self):
        # `\%` inside "$-0.63\%$" must not start a comment (the sentence is found), while an
        # unescaped `%` ends the active text of the line (the stale value behind it is not read)
        self.edit("07_discussion.tex", "over six trading days (Section~\\ref{sec:es_robust}).",
                  "over six trading days (Section~\\ref{sec:es_robust}). % old draft: $-0.35\\%$ over six trading days")
        code, out = self.run_script()
        self.assertEqual(code, 0, out)
        self.assertEqual(sum(r["passage"].startswith("Section 6.3") for r in self.rows()), 8)

    # ---- diagnostics are reported but never compared -----------------------------------
    def test_diagnostics_are_separate(self):
        code, out = self.run_script()
        diag = self.rows("prose_checks_diagnostics.csv")
        self.assertGreaterEqual(len(diag), 15)
        self.assertNotIn("match", diag[0])
        self.assertIn("Diagnostics (recomputed, not printed in the thesis, not compared):", out)
        compared = {r["quantity"] for r in self.rows()}
        self.assertNotIn("zero returns in the two (0,+20) event windows", compared)


if __name__ == "__main__":
    unittest.main()
