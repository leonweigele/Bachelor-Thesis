# LSEG / DATASTREAM DOWNLOAD LIST  (clean, full re-download)
# =========================================================
# Settings for EVERY request:
#   - Daily frequency, 2019-01-01 to 2026-06-30  (sample freeze)
#   - Export to Excel/CSV with a 'date' column
#   - Save into Data/manual/ named  lseg_<block>.csv  (get_data.py auto-merges)
#   - Source: WM/Refinitiv (WMR), datatype = Exchange Rate Middle (ER) where it
#     applies -> one consistent, citable source; FRED stays as a cross-check
#   - FX: NO currency conversion (the RIC fixes the quote). Assets: native ccy.
#
# ------------------------------------------------------------------
# BLOCK 1 — FX SPOT, full 32-currency panel          -> lseg_fx_spot.csv
# ------------------------------------------------------------------
# Majors, quoted USD per 1 FC:
#     EUR=  GBP=  AUD=  NZD=
# Quoted FC per 1 USD:
#     JPY=  CHF=  CAD=  SEK=  NOK=  DKK=
#     MXN=  BRL=  ZAR=  TRY=  INR=  KRW=  TWD=  THB=  SGD=  MYR=
#     IDR=  COP=  CLP=  PEN=  PLN=  HUF=  CZK=  ILS=  CNY=  HKD=
# Pegs / context:
#     SAR=  KWD=
#
# ------------------------------------------------------------------
# BLOCK 2 — 1M FX FORWARDS, same panel  -> lseg_fx_fwd1m_outright.csv
# ------------------------------------------------------------------
# Feeds build_fx_factors.py, which rebuilds the Dollar (DOL) + Carry (HML)
# factors through 2026 (the course factors are monthly and end in 2017).
# Forward discount = interest differential via CIP (Lustig-Roussanov-
# Verdelhan 2011).
#   PREFERRED — Datastream OUTRIGHT: search "<CURRENCY> 1 MONTH FORWARD"
#     (WMR or Barclays), datatype ER (middle). Gives the rate directly.
#     Save as  lseg_fx_fwd1m_outright.csv  (script default).
#   FALLBACK — Workspace points: RIC  <CCY>1M=  returns forward POINTS, not
#     the outright (outright = spot + points / 10^k; k=2 for JPY, else 4).
#     Save as  lseg_fx_fwd1m_points.csv  and set FWD_IS_POINTS=True in
#     build_fx_factors.py.
#   Same 32 currencies as Block 1. DKK, HKD, SAR, KWD, CNY are pegged/managed
#   -> pull them, but build_fx_factors.py drops them from the carry sort.
#
# ------------------------------------------------------------------
# BLOCK 3 — series NOT on FRED  (one file each, native currency)
# ------------------------------------------------------------------
#   XAU=         Gold spot, USD/oz                  -> lseg_gold.csv   (USD)
#   .STOXX50E    Euro Stoxx 50 price index          -> lseg_stoxx.csv  (EUR, not USD)
#   TRNLTTFMc1   TTF nat-gas front month, EUR/MWh   -> lseg_ttf.csv    (EUR, not USD)
#   # Native quote on purpose: keeps the EUR asset return free of the
#   # EUR/USD leg, which is analysed separately as the dollar.
#
# ------------------------------------------------------------------
# BLOCK 4 — optional backups (cross-check FRED, cheap while you're there)
# ------------------------------------------------------------------
#   .VIX   CBOE VIX           (cross-check FRED VIXCLS)
#   LCOc1  Brent front month  (cross-check FRED Brent spot)
#   .SPX   S&P 500            (cross-check FRED SP500)
