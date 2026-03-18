# Thesis Data Pipeline
## Geopolitical Risk & Currency Markets

### Directory Structure
```
thesis/
├── scripts/
│   ├── 01_data_acquisition.py   ← Downloads FX, market, GPR, FRED data
│   └── 02_data_cleaning.py      ← Merges, computes returns, builds portfolios
├── data/
│   ├── raw/                     ← Raw downloaded files land here
│   └── processed/               ← Cleaned master_daily.csv output
└── output/
    ├── figures/
    └── tables/
```

### Setup (one time)
```bash
pip install yfinance pandas numpy pandas-datareader requests openpyxl arch statsmodels scipy matplotlib seaborn
```

### Run
```bash
cd thesis
python scripts/01_data_acquisition.py
# ... do manual downloads (see below) ...
python scripts/02_data_cleaning.py
```

**Important:** Always `cd thesis` first. Both scripts use relative paths to `data/`.

### Manual Downloads (after running 01)

1. **GPR Index** — https://www.matteoiacoviello.com/gpr.htm
   - Download daily + monthly files → save as `data/raw/gpr_daily.csv` and `data/raw/gpr_monthly.csv`

2. **EPU / TPU** — https://www.policyuncertainty.com
   - US Monthly EPU → `data/raw/epu_monthly.csv`
   - Trade Policy Uncertainty → `data/raw/tpu_monthly.csv`

3. **Course Factors** — Your Project 2 dataset from Prof. Dalgic
   - 30 currencies + Dollar Risk + Carry Trade Risk → `data/raw/factors_course.csv`

### What the pipeline produces

`data/processed/master_daily.csv` contains:
- Daily log returns for ~30 currencies (foreign/USD convention)
- Portfolio returns: safe, risky, oil-exporter, oil-importer, risky-minus-safe, exporter-minus-importer
- Market data: S&P 500, VIX, Brent, WTI, Gold, Bitcoin, DXY returns and levels
- FRED yields, breakevens, TIPS, spreads
- GPR daily index
- Event dummies: Liberation Day (Apr 2, 2025), Iran strikes (Jun 2025), Hormuz closure (Feb 2026)

### Currency Classification (starting point — refine with Project 2 betas)

**Safe:** JPY, CHF, EUR, DKK, SEK, SGD, ILS, TWD, KRW
**Risky:** AUD, NZD, GBP, NOK, CAD, BRL, MXN, TRY, ZAR, INR, THB, PHP, MYR, IDR, COP, CLP, PEN, PLN, HUF, CZK, RON, RUB, NGN
**Oil Exporters:** NOK, CAD, RUB, COP, MXN, BRL, SAR, KWD, NGN, MYR
**Oil Importers:** JPY, EUR, INR, KRW, TRY, THB, PHP, ZAR, PLN, HUF, CZK, TWD, IDR
