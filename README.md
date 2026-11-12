# Customer Churn Prediction

Classification model that flags customers at higher risk of churning, with an explanation of *why* each group is flagged.
Built with Python, pandas, scikit-learn, Matplotlib and Jupyter.

> **Data note:** no real customer data was available, so the project runs on a **synthetic dataset** (`src/generate_data.py`, 5,000 customers, seed 42) with realistic defects injected on purpose: missing values, duplicates, inconsistent labels, impossible values, outliers, and a leakage trap. The pipeline is written so you can point it at real CSVs with the same columns. Treat the numbers below as a demonstration of the method, **not** as evidence about any real business (see [Limitations](#limitations)).

## Results at a glance

| | |
|---|---|
| Selected model | Logistic regression (class-weighted), chosen on cross-validated PR-AUC |
| Test ROC AUC / PR AUC | **0.77** / **0.48** (a no-skill model scores 0.50 / 0.23) |
| At the F1-optimal threshold (0.49) | precision **0.40**, recall **0.72**, F1 **0.51** (95% bootstrap CI for F1: 0.47 to 0.56) |
| What that means | flags 42% of customers and catches 72% of the churners (base churn rate is 23%) |
| At the cost-optimal threshold (0.32, assuming a missed churner costs 5x a wasted offer) | recall 0.89, precision 0.33, flags 63% of customers |
| Leakage audit | removed 2 post-churn columns; a single leaky column scored AUC 1.00, the honest model 0.78 |

### Model comparison (5-fold CV on the training set, threshold 0.5)

| Model | Precision | Recall | F1 | ROC AUC | PR AUC |
|---|---|---|---|---|---|
| Majority-class dummy | 0.00 | 0.00 | 0.00 | 0.50 | 0.23 |
| Logistic (no class weights) | 0.59 | 0.29 | 0.39 | 0.78 | 0.51 |
| **Logistic regression** | 0.40 | 0.73 | **0.52** | 0.78 | 0.51 |
| Decision tree (depth 5) | 0.37 | 0.72 | 0.49 | 0.73 | 0.42 |
| Random forest | 0.46 | 0.55 | 0.50 | 0.77 | 0.49 |
| Gradient boosting | 0.41 | 0.68 | 0.51 | 0.77 | 0.49 |

The logistic regression, random forest and gradient boosting are **within noise of each other** (fold-to-fold std of PR-AUC is about 0.04 to 0.05, larger than the gaps between them). The simple model was picked because it is no worse and is directly interpretable. The decision tree is clearly weakest.

![Model comparison](reports/figures/03_model_comparison.png)

## What drives churn

Permutation importance on the held-out test set (PR-AUC lost when a feature's values are shuffled) and the direction of each effect from the logistic regression:

| Rank | Driver | Direction | Evidence |
|---|---|---|---|
| 1 | **Contract type** | Month-to-month raises risk, two-year lowers it | 35% churn month-to-month vs 5% on two-year contracts |
| 2 | **Tenure** | Newer customers churn more | 39% churn in the first 6 months vs 6% after 49 months |
| 3 | **Late payments** (6 months) | More late payments, higher risk | |
| 4 | **Support tickets** (6 months) | More tickets, higher risk | 15% churn with 0 tickets vs 32% with 3+ |
| 5 | **Payment method** | Auto-debit customers churn less | |
| 6-7 | **Usage trend** (`logins_slope`, `logins_trend_pct`) | Falling logins, higher risk | the behavioural signal; weaker than contract and tenure |

![Permutation importance](reports/figures/09_permutation_importance.png)
![Logistic regression coefficients](reports/figures/10_lr_coefficients.png)

## Method

1. **Cleaning** (`reports/cleaning_log.csv` lists every decision with the number of rows affected)

   | Problem | Treatment |
   |---|---|
   | Duplicates (150 profile rows, 60 account IDs, 120 usage rows) | Exact duplicates dropped; a duplicated `customer_id` keeps the most complete record |
   | Inconsistent categories (`M` / `male ` / `MALE`, `Std`, `M2M`, `Y` / `TRUE`) | Trim, lower-case, map to one canonical label; the code **fails loudly** on any unmapped value |
   | Impossible values (age 999, negative charges or usage) | Set to missing, row kept |
   | Missing values | Tenure re-derived from `signup_date` where possible; categoricals become `"Unknown"`; numerics are median-imputed **with a missing-indicator inside the pipeline** (fit on training folds only) |
   | Outliers (x10 charge typos, usage spikes) | Median-based usage aggregation; winsorised at the training-fit 1st/99th percentile inside the pipeline |
   | Unlabelled rows (19) | Dropped |
   | Post-churn columns | Removed (leakage) |

2. **Features:** 27 inputs. Behavioural ones are built from each customer's own 6-month usage history: average level (median for minutes / data), **trend** (slope and a scale-free trend), last-2-vs-first-2-month **drop ratio**, **volatility**, **inactive months**, plus `friction_score` (tickets + late payments), `is_new_customer`, and value-for-money ratios.
3. **Split:** one stratified 80/20 split (churn share 23.4% / 23.5%), asserted to share no customers. EDA uses training data only. Model selection and threshold tuning use CV **on the training set**; the test set is scored once at the end.
4. **Evaluation:** precision, recall, F1, ROC and PR curves, confusion matrices at three thresholds, 1,000-sample bootstrap intervals on the test set.
5. **Imbalance and leakage checks** (below).

![Confusion matrices on the test set](reports/figures/07_confusion_matrices_test.png)
![Threshold tuning](reports/figures/06_threshold_tuning.png)

### Class imbalance
About 1 in 4 customers churns, so accuracy is misleading: the do-nothing dummy scores **77% accuracy with 0% recall**. Every model uses class weights; an unweighted logistic regression is kept as a control. It has higher precision (0.59) but catches only 29% of churners, which is why weighting plus threshold tuning is used. Metrics reported are precision, recall, F1 and PR-AUC rather than accuracy.

### Data leakage checks (all in the notebook, section 4c)
| Check | Result |
|---|---|
| `cancellation_reason` / `account_closed_date` exist only for churners | Removed. As a single feature they score **AUC 1.000**, versus 0.775 for the honest model |
| No customer in both train and test | 0 shared IDs |
| Imputer, winsoriser, scaler, encoder fit on training folds only | all inside the `Pipeline` |
| Threshold chosen without test data | tuned on out-of-fold predictions |
| Strongest single feature | AUC 0.687, nothing suspiciously perfect |
| Label-shuffle test | AUC **0.500** on shuffled labels, as it should be |

## Error patterns (test set, F1-optimal threshold)

* **It misses long-tenured and contracted churners:** recall is only **12%** for customers with 49+ months and **17%** on two-year contracts, because few of them churn and those who do look like loyal customers (stable usage, no tickets).
* **It over-flags new and month-to-month customers:** recall is 93% for the first 6 months, but precision is below 50% in every segment except two-year contracts (where only a handful of customers are flagged), so most flagged customers would not have churned.
* **Precision stays at or below about 0.5 for every sizeable segment.** The signal in this data is moderate; the model ranks risk well (ROC AUC 0.77) but cannot cleanly separate individuals.
* The score distributions of churners and stayers overlap heavily (`reports/figures/12_score_distribution.png`), which is where the errors come from.

![Error by segment](reports/figures/13_error_by_segment.png)

## Limitations

* **Synthetic data.** Churn was generated from a logistic-style formula plus noise, which favours logistic regression and caps achievable accuracy. That the ensembles did not beat it is expected here and **should not be assumed on real data**.
* **The retention-cost ratio (5:1) is an assumption.** The cost-optimal threshold moves with it. Replace `FN_COST` / `FP_COST` in the notebook with real figures.
* **Test set is small** (997 customers, 234 churners): the F1 interval is 0.47 to 0.56. Segment-level numbers rest on even fewer (for example 12 two-year churners), so treat them as indicative.
* **Correlation, not causation.** Importance and coefficients show what predicts churn, not what would reduce it. Month-to-month customers may be risky *because* of who chooses that contract.
* **Single snapshot, no time-based validation.** All features end at one date. A real deployment should validate on a later period to check for drift.
* **Probabilities are not calibrated** (class weights shift them upward); use them for ranking or thresholding, not as literal probabilities.
* **Small categories are noisy.** A coefficient such as `gender_Other` (about 2% of customers) is statistical noise, not a finding.
* **Some features are redundant** (`logins_slope`, `logins_trend_pct`, `logins_drop_ratio`), so credit between them is arbitrary.

## Recommended next steps

1. **Run it on real data:** map real columns onto the same schema, confirm the cleaning log, and re-check the leakage audit (especially anything recorded after the churn decision).
2. **Validate out of time:** train on earlier months and test on a later one; monitor drift.
3. **Set the threshold from real economics**, or turn the score into a top-N% retention list sized to campaign capacity (the cost curve is flat near its minimum, so there is room to choose).
4. **Calibrate** (Platt or isotonic) if probabilities will be shown to people or used for expected-value maths.
5. **Test interventions**, for example a retention offer to month-to-month customers in their first year with falling logins, using an A/B test to measure uplift rather than relying on predicted risk.
6. **Improve the features that matter:** richer usage events, billing history and complaint text; then re-compare the ensembles, which may pull ahead once non-linear signal exists.
7. **Explain individual predictions** (SHAP or per-customer coefficient contributions) before handing scores to the retention team.

## Repository layout

```
churn-prediction/
├── README.md
├── requirements.txt
├── data/raw/                     # synthetic CSVs: profile, accounts, usage
├── notebooks/churn_analysis.ipynb  # full analysis with executed outputs
├── src/
│   ├── generate_data.py          # creates the messy synthetic data
│   ├── churn_analysis.py         # the analysis as a plain script (Jupytext percent format)
│   └── build_notebook.py         # rebuilds the notebook from the script
└── reports/
    ├── cleaning_log.csv          # every cleaning decision + rows affected
    ├── model_comparison_cv.csv, test_metrics_by_threshold.csv, test_metrics.json
    ├── permutation_importance.csv, error_by_segment.csv
    └── figures/                  # 13 charts (PNG)
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/generate_data.py          # optional: data/raw is already included
jupyter lab notebooks/churn_analysis.ipynb   # run all cells (about 30 seconds)
```

The notebook finds the project root from its own folder, so run it from `notebooks/` or the repo root. Everything is seeded (`SEED = 42`). Developed and tested with Python 3.13, pandas 3.0, NumPy 2.5, scikit-learn 1.9, Matplotlib 3.11.

## Publish to GitHub

```bash
cd churn-prediction
git init -b main
git add .
git commit -m "Customer churn prediction: cleaning, modelling, evaluation, README"
# create an empty repo named churn-prediction on github.com (no README/licence), then:
git remote add origin https://github.com/<your-username>/churn-prediction.git
git push -u origin main
```

With the GitHub CLI instead: `gh repo create churn-prediction --public --source . --push`.
GitHub renders the notebook with its charts and this README on the repo home page.
