# Bachelor Thesis — Geopolitical Risk and Currency Markets

University of Mannheim, 2026

## Structure
- `Main/Data/` — Data pipeline and analysis scripts
- `Main/LaTeX Thesis/` — Thesis document (LaTeX)

## Data Pipeline
```
cd Main/Data
python3 scripts/01_data_acquisition.py
python3 scripts/02_data_cleaning.py
python3 scripts/03_event_study.py
python3 scripts/04_var_analysis.py
python3 scripts/05_garch_cross_section.py
```
