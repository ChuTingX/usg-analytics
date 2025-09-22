# 1. USG Analytics (code-only, dataset-specific)

This repository provides a **reproducible Python pipeline** to analyze **U.S. gold ETF price (Adj Close)** by reframing a time-series forecasting task as a **supervised regression** problem (simple one-day lags + rolling means, not AR/ARIMA).  
The raw dataset name is kept as **`data/raw/FINAL_USO.csv`** (as downloaded). All project code lives in **`code_usg/`**.

- Dataset: https://www.kaggle.com/datasets/sid321axn/gold-price-prediction-dataset  
- Code: `code_usg/`, `scripts/`  
- Configuration: `conf/`  
- Deliverables: `reports/` (figures, metrics), `data/processed/` (lagged table)

---

## 2. How it works (overview)

- **Cleaning & selection** → drop columns ending with `*_Trend`; keep a compact set of market variables centered on **Adj Close**.  
- **Temporal encoding** → create **1-day lagged predictors** (`*_prev`) so today’s target uses **yesterday’s information**.  
- **Smoothers** → add **7-day** and **30-day** rolling means of `Adj Close` to capture momentum.  
- **Models** → baseline (prev-day), OLS best-subset, Lasso, Ridge, Random Forest; plus a **simple averaging ensemble** of all four learned models.  
- **Evaluation** → 10-fold CV by default (or time-series splits if configured).  
- **Outputs** → correlation heatmap, time-series plots, residual diagnostics, metrics JSON, and the supervised (lagged) dataset.

> ⚠️ This is a **methods demonstration**, not a trading system. Do **not** use results for financial decisions.

---

## 3. Project layout

```
usg-analytics/
├─ code_usg/
├─ scripts/
├─ conf/
├─ data/
│  ├─ raw/
│  └─ processed/
├─ reports/
│  └─ figures/
├─ tests/
├─ README.md
├─ requirements.txt
└─ requirements-dev.txt
```

- **code_usg/** – reusable project code (features, modeling, plots, pipeline).  
- **scripts/** – entry points you run (EDA, full analysis, optional tuning).  
- **conf/** – configuration files (`params.yml`: paths, features, CV, model params).  
- **data/raw/** – original inputs (e.g., `FINAL_USO.csv`), **do not modify**.  
- **data/processed/** – pipeline outputs (e.g., `lagged_dataset.csv`).  
- **reports/** – deliverables (figures, metrics, tuned params).  
- **tests/** – pytest suite.

---

## 4. Quickstart — Step 1: Clone the repository

```bash
git clone https://github.com/<YOU>/<REPO>.git
cd <REPO>
```

---

## 5. Quickstart — Step 2: Create and activate an environment

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt -r requirements-dev.txt
```

> If using an IDE, set `PYTHONPATH=.` or mark `code_usg/` as a “Sources Root”.

---

## 6. Quickstart — Step 3: Configure the pipeline

Open `conf/params.yml` and adjust as needed:

```yaml
data:
  input_csv: data/raw/FINAL_USO.csv
  date_column: Date
  parse_dates: true
  index_as_date: true
  trend_suffix: "_Trend"

features:
  target: "Adj Close"
  variables: [ "Adj Close", "SP_close", "DJ_close", "USDI_Price", "EU_Price",
               "GDX_Close", "SF_Price", "PLT_Price", "PLD_Price",
               "RHO_PRICE", "USO_Close", "OF_Price", "OS_Price" ]
  rolling_means: [7, 30]
  lag_all_predictors: true

cv:
  strategy: "kfold"   # or "timeseries"
  n_splits: 10
  shuffle: true
  random_state: 42

models:
  random_forest:
    params:
      n_estimators: 200
      max_depth: null
      max_features: "sqrt"
      min_samples_split: 2
      min_samples_leaf: 1
      bootstrap: true
      random_state: 42
    tuning:
      enabled: true
      scoring: "neg_mean_squared_error"
      n_jobs: -1
      verbose: 1
      grid:
        n_estimators: [100, 200, 300, 500]
        max_depth: [null, 15, 25, 35]
        max_features: ["sqrt", 0.5, 1.0]
        min_samples_split: [2, 5, 10]
        min_samples_leaf: [1, 2]
        bootstrap: [true]

lasso:
  alphas_logspace: [-5, 2, 20]
  max_iter: 10000
  random_state: 42

ridge:
  alphas_logspace: [-5, 2, 20]
  max_iter: 10000
  random_state: 42
```

---

## 7. Quickstart — Step 4 (optional): Tune Random Forest

This searches the grid in `params.yml` and writes `reports/tuned_rf_params.json`.  
The main pipeline **auto-uses** tuned params if the file exists.

**macOS/Linux**
```bash
PYTHONPATH=. python scripts/tune_rf.py --config conf/params.yml --out reports
```

**Windows**
```bat
set PYTHONPATH=.
python scripts\tune_rf.py --config conf\params.yml --out reports
```

Outputs:
- `reports/tuned_rf_params.json` (best RF params)  
- `reports/rf_cv_results.csv` (full GridSearchCV table)

---

## 8. Quickstart — Step 5: Run the pipeline

**EDA only**
```bash
# macOS/Linux
PYTHONPATH=. python scripts/describe_and_plot.py --config conf/params.yml
# Windows
set PYTHONPATH=.
python scripts\describe_and_plot.py --config conf\params.yml
```

**Full analysis**
```bash
# macOS/Linux
PYTHONPATH=. python scripts/run_analysis.py --config conf/params.yml
# Windows
set PYTHONPATH=.
python scripts\run_analysis.py --config conf\params.yml
```

Outputs:
- `reports/descriptive_stats.csv`, `reports/median_values.csv`  
- `reports/figures/corr_heatmap.png`, `reports/figures/target_ts.png`, `reports/figures/predictors_ts.png`  
- `reports/figures/pred_vs_actual.png`, `reports/figures/residuals.png`  
- `reports/metrics.json`, `reports/run_metadata.json`  
- `data/processed/lagged_dataset.csv`

---

## 9. Why the ensemble helps (intuition)

Averaging **diverse** models reduces variance when their errors are **not perfectly correlated**—idiosyncratic mistakes tend to cancel.  
Reference: **Hansen & Salamon (1990)**, *Neural Network Ensembles*, **IEEE TPAMI** 12(10): 993–1001.

---

## 10. Tests

```bash
PYTHONPATH=. pytest -q
```

---

## 11. Troubleshooting

- **`ModuleNotFoundError: code_usg`** → set `PYTHONPATH=.` or mark `code_usg/` as sources root in your IDE.  
- **`FileNotFoundError: FINAL_USO.csv`** → ensure the file is placed at `data/raw/FINAL_USO.csv` (or update the config).  
- **Slow exhaustive subset search** → lower `cv.n_splits` or skip OLS best-subset if time-constrained.  
- **Chronological validation** → set `cv.strategy: timeseries` for time-aware splits.

---

## 12. Scope & disclaimer

This is a **simple investigation** showing how to:  
1) convert a time-series problem into standard regression via one-day lags + smoothers, and  
2) gain modest accuracy via a plain averaging ensemble.  

It **does not** provide predictive power suitable for trading or investing. Use at your own discretion.

---

## 13. License — MIT

Released under the **MIT License**. You may **use, copy, modify, merge, publish, distribute, sublicense, and/or sell** copies of the software, per the MIT terms.  
The software is provided **“as is”**, without warranty of any kind. See `LICENSE` for full text.