"""Regression checks for Code/event_study/verify_results.py.

Every test builds a throw-away copy of the verifier, the three event-study
output files and their baselines in a temporary folder, runs the script there
as a subprocess (exactly as `python3 Code/event_study/verify_results.py` is run by hand)
and checks the exit code and the report. Nothing in the repository is touched.

Run from the repo root:
    python -m unittest discover -s Code/tests -p 'test_verify_results.py'

Environment overrides:
    VERIFY_TEST_ROOT    repo root holding Data/processed/event_study (default: two levels up)
    VERIFY_TEST_SCRIPT  verifier under test (default: <root>/Code/event_study/verify_results.py)
    VERIFY_TEST_V2      optional path to the previous verifier; enables the
                        check that its report lines for the two old files are reproduced
"""
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import date
from hashlib import sha256
from pathlib import Path

ROOT = Path(os.environ.get("VERIFY_TEST_ROOT", Path(__file__).resolve().parents[2]))
SCRIPT = Path(os.environ.get("VERIFY_TEST_SCRIPT", ROOT / "Code/event_study/verify_results.py"))
V2 = os.environ.get("VERIFY_TEST_V2")
ES = Path("Data/processed/event_study")
SUMMARY, DIFF, W50 = "car_summary.csv", "cross_event_diff.csv", "car_persistence_w50.csv"
FILES = [SUMMARY, DIFF, W50]
IDENTICAL, DIFFERS = "RESULT: IDENTICAL", "RESULT: DIFFERS"
TODAY = f"{date.today():%Y-%m-%d}"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


class VerifyResultsTests(unittest.TestCase):
    """Fresh tree per test: Code/event_study/verify_results.py, the three current files and
    baselines for all three. The new file's baseline is a copy of its current
    file, i.e. the state right after `--pin car_persistence_w50.csv`."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="verify_results_test_")
        self.tmp = Path(self._tmp.name)
        self.es, self.base = self.tmp / ES, self.tmp / ES / "baseline"
        self.base.mkdir(parents=True)
        (self.tmp / "Code/event_study").mkdir(parents=True)
        shutil.copy(SCRIPT, self.tmp / "Code/event_study/verify_results.py")
        for f in FILES:
            shutil.copy(ROOT / ES / f, self.es / f)
        for f in (SUMMARY, DIFF):
            shutil.copy(ROOT / ES / "baseline" / f, self.base / f)
        shutil.copy(self.es / W50, self.base / W50)
        self.n = {f: len(self.rows(f)) - 1 for f in FILES}      # data rows per file

    def tearDown(self):
        self._tmp.cleanup()

    # ------------------------------------------------------------ helpers
    def run_verifier(self, *args, script=None):
        script = script or self.tmp / "Code/event_study/verify_results.py"
        return subprocess.run([sys.executable, str(script), *args],
                              capture_output=True, text=True, cwd=self.tmp)

    def assert_run(self, proc, exit_code, *fragments):
        out = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, exit_code, out)
        self.assertNotIn("Traceback", out, out)
        for frag in fragments:
            self.assertIn(frag, out, out)

    def rows(self, name, where="current"):
        path = (self.es if where == "current" else self.base) / name
        with open(path, newline="") as fh:
            return list(csv.reader(fh))

    def write(self, name, rows, where="current"):
        path = (self.es if where == "current" else self.base) / name
        with open(path, "w", newline="") as fh:
            csv.writer(fh, lineterminator="\n").writerows(rows)

    def line(self, name, **counts):
        c = {"shared": self.n[name], "changed": 0, "new": 0, "removed": 0}
        c.update(counts)
        return f"{name}: {c['shared']} shared rows | {c['changed']} changed | {c['new']} new | {c['removed']} removed"

    def baseline_digests(self):
        return {p.name: digest(p) for p in self.base.iterdir()}

    # ------------------------------------------------------------ unchanged
    def test_unchanged_tree_is_identical_exit_0(self):
        p = self.run_verifier()
        self.assert_run(p, 0, IDENTICAL, self.line(SUMMARY), self.line(DIFF), self.line(W50))

    # ------------------------------------------------------------ value changes in the new file
    def test_changed_value_fails(self):
        r = self.rows(W50)
        old, r[1][4] = r[1][4], "-9.99"
        self.write(W50, r)
        p = self.run_verifier()
        self.assert_run(p, 1, DIFFERS, self.line(W50, changed=1), old, "-9.99", r[1][0], r[1][1])

    def test_changed_stars_only_fails(self):
        r = self.rows(W50)
        i, j = next((i, j) for i in range(1, len(r)) for j in (2, 3, 4) if "*" in r[i][j])
        r[i][j] = r[i][j].rstrip("*")
        self.write(W50, r)
        p = self.run_verifier()
        self.assert_run(p, 1, DIFFERS, self.line(W50, changed=1))

    def test_cell_blanked_fails(self):
        r = self.rows(W50)
        r[1][2] = ""
        self.write(W50, r)
        p = self.run_verifier()
        self.assert_run(p, 1, DIFFERS, self.line(W50, changed=1))

    def test_starless_column_is_compared_as_text(self):
        # a column without any star would be parsed as floats by a default read;
        # the verifier reads the file as text, so only the starred cells differ
        r = self.rows(W50)
        starred = sum("*" in row[4] for row in r[1:])
        self.assertGreater(starred, 0)
        for row in r[1:]:
            row[4] = row[4].rstrip("*")
        self.write(W50, r)
        p = self.run_verifier()
        self.assert_run(p, 1, DIFFERS, self.line(W50, changed=starred))
        # the same starless file on both sides is identical, no false alarm from number formatting
        self.write(W50, r, where="baseline")
        p = self.run_verifier()
        self.assert_run(p, 0, IDENTICAL, self.line(W50))

    # ------------------------------------------------------------ added and removed rows
    def test_added_row_fails(self):
        r = self.rows(W50) + [["ZZZ_TEST", "hormuz", "0.1", "0.2", "0.3"]]
        self.write(W50, r)
        p = self.run_verifier()
        self.assert_run(p, 1, DIFFERS, self.line(W50, new=1), "new rows (first 5)", "ZZZ_TEST")

    def test_removed_row_fails(self):
        r = self.rows(W50)
        gone = r.pop(1)
        self.write(W50, r)
        p = self.run_verifier()
        self.assert_run(p, 1, DIFFERS, self.line(W50, shared=self.n[W50] - 1, removed=1),
                        "removed rows (first 5)", gone[0])

    def test_added_and_removed_rows_without_value_change_fails(self):
        # the bug fixed in v2: rows that come or go must fail even when every shared row matches
        r = self.rows(W50)
        r.pop(1)
        r.append(["ZZZ_TEST", "ukraine", "0.1", "0.2", "0.3"])
        self.write(W50, r)
        p = self.run_verifier()
        self.assert_run(p, 1, DIFFERS, self.line(W50, shared=self.n[W50] - 1, new=1, removed=1))

    def test_added_and_removed_rows_in_car_summary_still_fail(self):
        r = self.rows(SUMMARY)
        del r[1:6]
        r.append(["zzz_test", "EUR", "const_mean", "(0,1)", "0.01", "1.0", ""])
        self.write(SUMMARY, r)
        p = self.run_verifier()
        self.assert_run(p, 1, DIFFERS, self.line(SUMMARY, shared=self.n[SUMMARY] - 5, new=1, removed=5))

    # ------------------------------------------------------------ missing and unreadable files
    def test_missing_current_file_fails_without_traceback(self):
        (self.es / W50).unlink()
        p = self.run_verifier()
        self.assert_run(p, 1, DIFFERS, f"{W50}: MISSING", self.line(SUMMARY), self.line(DIFF))

    def test_missing_baseline_fails_and_names_the_scoped_pin(self):
        # the state right after installing the extension, before the new file is pinned
        (self.base / W50).unlink()
        p = self.run_verifier()
        self.assert_run(p, 1, DIFFERS, f"{W50}: MISSING", f"--pin {W50}", self.line(SUMMARY), self.line(DIFF))

    def test_missing_baseline_folder_fails(self):
        shutil.rmtree(self.base)
        p = self.run_verifier()
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("No baseline yet", p.stdout + p.stderr)

    def test_empty_file_is_reported_not_a_crash(self):
        (self.es / W50).write_bytes(b"")
        p = self.run_verifier()
        self.assert_run(p, 1, DIFFERS, f"{W50}: UNREADABLE", "EmptyDataError")

    def test_header_only_file_counts_every_row_as_removed(self):
        self.write(W50, self.rows(W50)[:1])
        p = self.run_verifier()
        self.assert_run(p, 1, DIFFERS, self.line(W50, shared=0, removed=self.n[W50]))

    # ------------------------------------------------------------ schema problems
    def test_missing_column_fails(self):
        self.write(W50, [row[:4] for row in self.rows(W50)])
        p = self.run_verifier()
        self.assert_run(p, 1, DIFFERS, f"{W50}: column sets differ", "only in baseline ['(0,50)']")

    def test_extra_column_fails(self):
        r = self.rows(W50)
        r[0].append("extra")
        for row in r[1:]:
            row.append("1")
        self.write(W50, r)
        p = self.run_verifier()
        self.assert_run(p, 1, DIFFERS, f"{W50}: column sets differ", "only in current ['extra']")

    def test_renamed_key_column_fails(self):
        r = self.rows(W50)
        r[0][0] = "Series"
        self.write(W50, r)
        p = self.run_verifier()
        self.assert_run(p, 1, DIFFERS, f"{W50}: column sets differ")

    def test_duplicate_key_fails(self):
        r = self.rows(W50)
        r.append(list(r[1]))
        self.write(W50, r)
        p = self.run_verifier()
        self.assert_run(p, 1, DIFFERS, f"{W50}: duplicated keys — baseline 0, current 1")

    def test_empty_key_cell_fails(self):
        r = self.rows(W50)
        r[1][0] = ""
        self.write(W50, r)
        p = self.run_verifier()
        self.assert_run(p, 1, DIFFERS, f"{W50}: rows with an empty key — baseline 0, current 1")

    # ------------------------------------------------------------ the two old files keep their behaviour
    def test_summary_numeric_change_beyond_tol_fails(self):
        r = self.rows(SUMMARY)
        col = r[0].index("CAR")
        r[1][col] = repr(float(r[1][col]) + 1e-6)
        self.write(SUMMARY, r)
        p = self.run_verifier()
        self.assert_run(p, 1, DIFFERS, self.line(SUMMARY, changed=1))

    def test_summary_numeric_change_within_tol_passes(self):
        r = self.rows(SUMMARY)
        col = r[0].index("CAR")
        r[1][col] = repr(float(r[1][col]) + 1e-12)
        self.write(SUMMARY, r)
        p = self.run_verifier()
        self.assert_run(p, 0, IDENTICAL, self.line(SUMMARY))

    def test_summary_sig_text_change_fails(self):
        r = self.rows(SUMMARY)
        col = r[0].index("sig")
        i = next(i for i in range(1, len(r)) if r[i][col] == "***")
        r[i][col] = "**"
        self.write(SUMMARY, r)
        p = self.run_verifier()
        self.assert_run(p, 1, DIFFERS, self.line(SUMMARY, changed=1))

    def test_diff_file_removed_row_fails(self):
        r = self.rows(DIFF)
        r.pop(1)
        self.write(DIFF, r)
        p = self.run_verifier()
        self.assert_run(p, 1, DIFFERS, self.line(DIFF, shared=self.n[DIFF] - 1, removed=1))

    @unittest.skipUnless(V2, "set VERIFY_TEST_V2 to the previous verify_results.py to enable")
    def test_report_lines_for_the_two_old_files_match_the_previous_verifier(self):
        v2 = self.tmp / "Code/verify_results_previous.py"
        shutil.copy(V2, v2)

        def old_file_lines(out):
            keep = []
            for l in out.splitlines():
                if l.startswith(W50):
                    break
                if l.strip() and not l.startswith("RESULT:"):
                    keep.append(l)
            return keep

        scenarios = {"unchanged": lambda: None}

        def change_summary():
            r = self.rows(SUMMARY)
            r[1][r[0].index("t")] = "9.9"
            self.write(SUMMARY, r)

        def drop_diff_rows():
            r = self.rows(DIFF)
            del r[1:4]
            self.write(DIFF, r)

        scenarios["changed value in car_summary"] = change_summary
        scenarios["removed rows in cross_event_diff"] = drop_diff_rows
        for name, edit in scenarios.items():
            edit()
            a, b = self.run_verifier(script=v2), self.run_verifier()
            self.assertEqual(old_file_lines(a.stdout), old_file_lines(b.stdout), name)
            self.assertEqual(a.returncode, b.returncode, name)

    # ------------------------------------------------------------ pinning
    def test_pin_named_file_touches_only_that_file(self):
        (self.base / W50).unlink()
        before = self.baseline_digests()
        p = self.run_verifier("--pin", W50)
        self.assert_run(p, 0, f"Baseline pinned in", f"({W50}).")
        after = self.baseline_digests()
        self.assertEqual({k: v for k, v in after.items() if k != W50}, before)
        self.assertEqual(after[W50], digest(self.es / W50))
        self.assertEqual(sorted(after), sorted(FILES))          # no backup files created
        p = self.run_verifier()
        self.assert_run(p, 0, IDENTICAL, self.line(W50))

    def test_pin_over_an_existing_baseline_keeps_a_backup(self):
        old = digest(self.base / W50)
        r = self.rows(W50)
        r[1][4] = "-9.99"
        self.write(W50, r)
        bak = self.base / f"{W50}.bak_{TODAY}_prepin"
        p = self.run_verifier("--pin", W50)
        self.assert_run(p, 0, f"{W50}: previous baseline kept as {bak.name}")
        self.assertEqual(digest(bak), old)
        self.assertEqual(digest(self.base / W50), digest(self.es / W50))
        # a second pin on the same day keeps the first backup and numbers the next one
        p = self.run_verifier("--pin", W50)
        self.assert_run(p, 0, f"previous baseline kept as {bak.name}_2")
        self.assertEqual(digest(bak), old)
        self.assertTrue((self.base / f"{bak.name}_2").is_file())

    def test_pin_all_backs_up_every_existing_baseline(self):
        before = self.baseline_digests()
        p = self.run_verifier("--pin")
        self.assert_run(p, 0, f"({', '.join(FILES)}).")
        for f in FILES:
            self.assertEqual(digest(self.base / f"{f}.bak_{TODAY}_prepin"), before[f])
            self.assertEqual(digest(self.base / f), digest(self.es / f))

    def test_pin_unknown_name_is_refused(self):
        before = self.baseline_digests()
        p = self.run_verifier("--pin", "nonsense.csv")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("Unknown file(s) ['nonsense.csv']", p.stdout + p.stderr)
        self.assertEqual(self.baseline_digests(), before)

    def test_pin_with_missing_current_file_is_refused(self):
        (self.es / W50).unlink()
        before = self.baseline_digests()
        p = self.run_verifier("--pin", W50)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("Cannot pin", p.stdout + p.stderr)
        self.assertEqual(self.baseline_digests(), before)

    def test_unknown_argument_is_refused(self):
        before = self.baseline_digests()
        p = self.run_verifier("--pim")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("Unknown argument(s) ['--pim']", p.stdout + p.stderr)
        self.assertEqual(self.baseline_digests(), before)


if __name__ == "__main__":
    unittest.main()
