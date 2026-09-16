Backups moved out of the working folders on 16 Sep 2026 (Cowork session).

Each file keeps its original name and its original relative path below this folder, e.g.
  _backups/Main/LaTeX Thesis/content/06_results.tex.bak_2026-09-16_gold_denomination
was
  Main/LaTeX Thesis/content/06_results.tex.bak_2026-09-16_gold_denomination

Naming: <file>.bak_<date>_<tag> = copy of <file> taken right before the session with that tag
wrote to it (modification time preserved). The *_prepin files are the verifier's copies of
overwritten baselines (Code/verify_results.py --pin).

Everything here is ignored by git (*.bak_*, *_backup_*). Safe to delete once a commit exists.
