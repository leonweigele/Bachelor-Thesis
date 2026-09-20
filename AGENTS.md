# Thesis project instructions

## Writing preference

Leon clarified on 15 September 2026 that the preference applies only to prose sentences. Do not use em dashes or semicolons within sentences written or revised for him, including chat replies, proposed wording, thesis prose, and sentences in captions, footnotes, and table notes.

- Rewrite sentences with full stops, commas, conjunctions, or parentheses as appropriate. Do not substitute en dashes for em dashes within prose sentences.
- Preserve conventional punctuation outside prose sentences. This includes separators between citations, bibliography formatting, original source titles and reference metadata, data-series labels, structured lists, and table placeholders.
- Keep necessary hyphens, date and page ranges, minus signs, and technical syntax intact. LaTeX backslash-semicolon is a spacing command and does not print a semicolon.
- Do not change citation delimiters, repeated-author bibliography formatting, or table symbols merely to enforce the prose preference.
- Check revised sentences for em dashes and semicolons before delivery. Do not require the entire PDF or every source file to contain zero occurrences, because legitimate structural and bibliographic uses are allowed.

## Current thesis

The working thesis is at `/Users/leon/Bachelor Thesis/Main/LaTeX Thesis/mainfile.tex`. Read the current source and `/Users/leon/Bachelor Thesis/THESIS-CONTEXT.md` before editing. Review folders contain historical copies.

## Backups and git

Backups of edited files go to `_backups/<original relative path>.bak_<date>_<tag>` (copy with the original modification time), never next to the file. See `_backups/README.txt`. Sessions that run inside a Cowork VM use git read-only (`GIT_OPTIONAL_LOCKS=0 git log/status/diff`) and never `git add`, `commit`, `stash` or `checkout` from there, because the VM cannot delete git's lock files. Commits are made from the Mac terminal.
