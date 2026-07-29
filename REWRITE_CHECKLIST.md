# Rewrite & citation-check list — Claude-generated passages

Created 2026-07-15. Work order: Block 1 → 4. For each item: rewrite in your own
words, then open the cited paper and confirm it actually says what the sentence
claims. Tick when both are done.

"Exact" = written today, locations precise. "Earlier" = from previous sessions,
find via the quoted phrase (grep it in `content/`).

---

## Block 1 — Ch. 5 Methodology + Ch. 3 Framework (do first; supervisor probes here)

Papers to have open: MacKinlay (1997 JEL); Campbell–Lo–MacKinlay (1997 book, ch. 4 —
library); Kothari & Warner (2007, free PDF); Kolari & Pynnönen (2010);
Baur & Lucey (2010); Chen & Rogoff (2003); Kilian (2009 AER).

- [ ] **05, §Event Study Methodology, opening sentence** (exact): constant-mean model
  + `\parencite{mackinlay_1997,campbell_lo_mackinlay_1997}` + "ending where the event
  window begins". VERIFY: MacKinlay describes constant-mean-return model; CLM ch. 4
  covers it too.
- [ ] **05, CAR windows sentence** (exact): "(−1,+1) guards against information
  reaching prices the day before" + `\parencite{kothari_warner_2007}`. VERIFY: KW
  discuss short windows around the event / leakage rationale.
- [ ] **05, CAR(0,50) sentence** (exact): "robustness exhibit on persistence — not a
  second identification". Rewrite in your voice.
- [ ] **05, difference-in-CARs sentence** (exact): "two-sample analogue of the CAR
  variance in \textcite{campbell_lo_mackinlay_1997}; independence … windows do not
  overlap". VERIFY: CLM ch. 4 gives Var(CAR) = L·σ²; the difference-SE
  √(L(σ²_A+σ²_B)) follows from independence — make sure you can derive/defend it.
- [ ] **05, Kolari–Pynnönen sentence** (exact): "...inference here is drawn at the
  portfolio level around single, non-overlapping event dates ... plain t-test is
  retained." VERIFY: KP 2010 is about cross-sectional correlation when averaging
  many securities on a common event date — check the sentence states their point
  correctly.
- [ ] **05, estimation window numbers**: [−140,−21], 1/120 sum bounds in the equation
  — confirm they match `Code/03_event_study.py` (EST_WIN) after any future change.
- [ ] **03, §3.2 title** (exact): "Classifying Shocks by Origin and Nature" — yours
  now, just confirm you like it in the compiled ToC.
- [ ] **03, §3.3 H1–H3 description list** (exact, fully Claude-written): rewrite all
  three items in your voice. VERIFY: H2 wording against Baur & Lucey (2010) —
  their hedge vs. safe-haven definitions; H3 against Chen & Rogoff (2003) +
  Kilian (2009) — commodity-currency channel, supply- vs demand-driven oil moves.
- [ ] **03, §3.1 definitions** (earlier): safe haven vs hedge definitional paragraph —
  check against Baur & Lucey (2010), Ranaldo & Söderlind (2010).

## Block 2 — Ch. 6 Results + Ch. 7 Discussion (pair with the number audit)

Papers: Ostry–Lloyd–Corsetti (2025); Kamin (2025); Ranaldo–Söderlind (2010);
Habib–Stracca (2012); Jiang–Krishnamurthy–Lustig–Richmond (2026).

- [ ] **07, §7.2 adjudication paragraph** (exact, fully Claude-written): "Read against
  the hypotheses of Section 3.3 … H1/H2/H3 supported". Rewrite; verify each verdict
  against your own fresh tables (not just the old prose).
- [ ] **07, §7.2 leftover fix** (exact): "U.S.-involved Hormuz" terminology — read the
  whole section for voice.
- [ ] **06, §6.0 intro sentence** (exact rewrite): "three of the shocks are
  U.S.-linked — the Liberation Day tariff announcement is U.S.-originated, and the
  twelve-day war and the 2026 Hormuz crisis are military shocks in which the United
  States is involved as a co-belligerent…". Rewrite in your rhythm.
- [ ] **06, terminology edits throughout** (exact): U.S.-linked / U.S.-involved
  replacements in §6.3 — reread the section aloud.
- [ ] **06 + 07, all hand-typed CAR numbers** (earlier): line-by-line against fresh
  `car_summary.csv` / `car_persistence_w50.csv` / `cross_event_table.csv`
  (= the TODO.md audit item; Claude can script the comparison).
- [ ] **06, figure captions ×4** (exact, Claude-written): liberation_day,
  iran_12day, hormuz, hormuz_w50 — rewrite in your caption style; check the
  WTI-omission note reads as YOUR reasoning (EIA benchmark share; Fattouh 2011
  Cushing logistics — verify both if you keep the note).

## Block 3 — Ch. 4 Background (facts & dates against sources)

Papers: Kamin (2025); Fattouh & Economou (2026); IMF MENAP (2026);
Kilian–Plante–Richter–Zhou (2026); ECB FSR/Bulletin (2025/26);
Caldara & Iacoviello (2022).

- [ ] **04, Hormuz t0 sentence** (exact, Claude-written): "Throughout the empirical
  chapters, day 0 of the Hormuz crisis is 28 February 2026 … closure … only as a
  sub-event." Rewrite; NOTE the weekend fact (28 Feb = Saturday → trading day 0 =
  Mon 2 Mar) is not yet in the text — add it here when you rewrite (open TODO item).
- [ ] **04, §structure-of-comparison edit** (exact): "involved as a co-belligerent"
  passage around the t0 sentence.
- [ ] **04, event narrative dates** (earlier, Claude-edited): every date (28 Feb
  strikes, 2 Mar closure declared, 4 Mar complete control, 7 Apr ceasefire,
  13 Apr blockade, 25 May strikes, 11 Jun/17 Jun ceasefire) vs the cited source
  (IMF MENAP, Fattouh–Economou). Your writing-style rule: every date sourced.
- [ ] **04, Ukraine control section** (earlier): claims vs Ranaldo–Söderlind /
  Habib–Stracca (classical haven response).
- [ ] **04, caption sources** ("Source: FRED", "Caldara and Iacoviello's…"): still
  accurate after the re-pull (they are — data unchanged in kind).

## Block 4 — Ch. 2 §2.5–2.7 + Ch. 1 + Ch. 8 sweep

(You already rewrote & verified §2.1–2.4 yourself — skip those.)

- [ ] **02, §2.5 Tariff Shocks** (earlier, Claude-rebuilt from old verified text):
  rewrite; verify Jiang et al. (2026) "flight away" characterization + any numbers.
- [ ] **02, §2.6 Commodity Currencies and Oil** (earlier): verify Chen–Rogoff /
  Kilian claims (same papers as Block 1 — reuse).
- [ ] **02, §2.7 Synthesis and Research Gap** (earlier, Claude-written): the "gap"
  paragraph — this is your thesis's selling point; make it fully yours. Also the
  "classification that separates a shock's origin from its nature" phrase (exact,
  today) appears here — keep terminology aligned.
- [ ] **01, Introduction** (earlier, partly Claude): central-question paragraph +
  "three episodes in all" design paragraph + results-preview paragraph
  ("mirror image", "collapses to essentially zero") + today's "involved as a
  co-belligerent" edit. Rewrite for voice; check the preview numbers against
  fresh tables after the Block-2 audit.
- [ ] **08, Conclusion** (earlier): whole-chapter voice pass; the DCC "future work"
  sentence should cite engle_2002 (kept in bib for exactly this — check it's
  actually \cite'd there).

## Methodology honesty (added 2026-07-15)

- [ ] **§5: state the honest estimation-window justification.** 120 days (not
  MacKinlay's ~250) — the reason is it REDUCES neighbour contamination, not that it
  eliminates it. Verified trading-day gaps: Liberation Day → 12-day war = 52 td, so
  the 12-day war's EST window [−140,−21] reaches back over Liberation Day + tariff
  pause (at −52, −47); a 250-day window would be worse. Headline pair Liberation
  Day → Hormuz = 238 td apart = fully separated. The ±20 event windows never overlap
  (nearest main events 52 td > 41-day span); overlap appears only at ±50 (hence ±50
  = persistence/robustness, not identification — code marks neighbours on those
  figs). Do NOT write "120 keeps all events clean" (false). Correct sentence: "a
  120-day window reduces overlap vs a longer window; the only event whose estimation
  window still reaches a neighbour is the 12-day war, whose response is
  insignificant, while the Liberation Day–Hormuz comparison is fully separated."
- [ ] **DECIDE whether to keep the 12-day war (iran_12day) as a reported event at
  all.** Case for cutting: muted/insignificant on every horizon; its benchmark is
  the one contaminated by Liberation Day; the "ambiguous actor" story adds a caveat
  rather than evidence; dropping it tightens the two-shocks-plus-benchmark design to
  exactly Liberation Day vs Hormuz vs Ukraine. Case for keeping: it's the June-2025
  onset of the SAME Iran conflict whose 2026 escalation is Hormuz, so it documents
  that the muted early phase and the sharp later phase differ (co-belligerent →
  ambiguous); it's honest to show a null. If KEPT: frame explicitly as a secondary
  "null result" event, not a headline. If CUT: remove from §6.2 + its figure + the
  cross-event diff rows + intro "twelve-day war" mentions; keep in Ch.4 narrative as
  context for the Hormuz escalation. Touches: 02_build_returns EVENTS (or just the
  reporting lists), 06_results §6.2, fig_car_iran_12day, tab_cross_event_diff, intro.

## Cross-cutting

- [ ] **tab_market_model.tex caption** (exact, Claude-written; auto-generated file —
  edit the caption in `Code/make_mm_table.py`, not the .tex, or it's overwritten
  on the next run).
- [ ] **11_appendix_a.tex: market-model paragraph + two appendix figure captions**
  (exact, Claude-written).
- [ ] **10_data_sources: the 11 bib entries** (exact): check each URL opens, each
  identifier list is complete, urldate 2026-07-15 is right; trim the TPU mention
  in `ds_epu` if TPU gets dropped.
- [ ] **In-code comments** you might submit (`03/04` figure-design comments,
  `lseg_pull.py`, `make_mm_table.py` docstrings) — fine to leave, but read once.
