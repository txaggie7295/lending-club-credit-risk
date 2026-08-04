# Lending Club Credit Risk Analysis
### Predicting Loan Default Using Origination-Time Attributes

**Jeffrey Symons | Data Analytics Bootcamp | July 2026**

---

## Business Problem

Consumer lenders lose billions annually to unexpected charge-offs. This project analyzes
Lending Club's historical consumer loan performance data (2007–2018) to build a credit
risk model capable of predicting loan default at the time of origination — using only
information available when the loan application is submitted.

**Key constraint:** No post-origination data, no Lending Club risk labels. Grade,
sub-grade, and interest rate were deliberately excluded — these are Lending Club's own
risk assessments derived from the same origination-time attributes used as model features. Including
them would mean modeling their model, not building an independent credit risk assessment.

---

## Key Findings

- **XGBoost achieves 0.7179 AUC** using origination-time attributes — no Lending Club risk labels
- **Removing grade costs only 0.009 AUC** — the raw bureau data carries independent
  predictive signal equivalent to Lending Club's proprietary grading system
- **Term is the strongest single predictor** — 60-month loans default at 2× the rate
  of 36-month loans (33% vs 16%)
- **At the recommended 0.70 risk threshold:**
  - Reject 11.96% of loan volume (approximately 1 in 8 loans)
  - Avoid 27.50% of defaults (more than 1 in 4 charge-offs)
  - Reduce portfolio default rate from 20.24% to 16.66%
  - Estimated net benefit: $125.8M (assumed LGD 70%, net yield 5%)
- **Policy validation:** Term adjustment and DTI-based rescue policies both failed —
  borderline loans default at 39–45% regardless of term or DTI, confirming the model's
  risk score is the most reliable policy instrument

---

## Data

**Source:** Kaggle — Lending Club Loan Data (publicly available)
**Size:** 2.6 million rows, 145 columns, ~1.1 GB raw CSV
**Period:** 2007–2018 loan originations
**Scope:** Individual applications only, resolved outcomes (Charged Off / Fully Paid)

**Data engineering highlights:**
- Used DuckDB to filter and select columns directly from the 1.1GB CSV before loading
  into pandas — avoiding memory constraints on a 16GB machine
- Dropped pre-August 2012 rows (4.92% of data) to retain 35 additional bureau attributes
  collected from 2012 onward — trading 5% of rows for nearly double the feature set
- Identified and dropped 16 columns with vintage coverage issues (Dec 2015 and Aug 2012 cutoffs)
- Dtype optimization reduced memory from 936MB to 259MB (72% reduction)
- Outlier treatment: DTI negatives removed, utilization capped at 200% (legitimate
  over-limit behavior preserved between 100–200%)
- Final working dataset: 1,217,422 rows, 38 features, saved as Parquet

---

## Model Performance

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.32 | 0.63 | 0.42 | 0.6990 |
| Random Forest | 0.45 | 0.24 | 0.31 | 0.7048 |
| **XGBoost (selected)** | **0.33** | **0.66** | **0.43** | **0.7179** |

XGBoost selected — highest AUC and recall. A 0.7179 AUC is respectable for consumer
credit; industry scorecards typically range 0.65–0.80.

---

## Policy Simulation

| Threshold | Volume Rejected | Defaults Avoided | Default Rate | Net Benefit |
|---|---|---|---|---|
| 0.50 | 40.8% | 65.5% | 11.78% | $278.2M |
| 0.55 | 32.3% | 56.5% | 13.00% | $244.9M |
| 0.60 | 24.6% | 47.0% | 14.23% | $207.6M |
| 0.65 | 17.7% | 37.0% | 15.49% | $166.5M |
| **0.70** | **12.0%** | **27.5%** | **16.66%** | **$125.8M** |
| 0.75 | 7.3% | 18.3% | 17.83% | $85.0M |
| 0.80 | 3.7% | 10.4% | 18.83% | $49.0M |
| 0.85 | 1.3% | 4.1% | 19.67% | $19.4M |

**Recommended threshold: 0.70** — operationally viable (rejects only 1 in 8 loans),
avoids more than 1 in 4 defaults, and generates $125.8M net benefit. Higher thresholds
maximize net benefit mathematically but reject 40%+ of volume — not operationally
viable and triggers fair lending scrutiny.

*Dollar figures are illustrative, using assumed LGD of 70% and net yield of 5%.*
*Average loan amount: $14,454*

---

## EDA Highlights

| Variable | Finding |
|---|---|
| Grade | 8× spread — Grade A (6%) to Grade G (50%) default rate |
| Term | 60-month loans default at 2× rate of 36-month loans |
| DTI | Nearly 3× spread across buckets — clean monotonic progression |
| Annual income | Clear inverse relationship — flattens above $125K |
| Home ownership | Counterintuitive — outright owners default more than mortgage holders |
| Verification status | Confounding variable — proxy for grade, not independent signal |
| Vintage | 2008 recession spike, 2016 CEO scandal peak, 2017-18 survivorship bias |
| Geography | 13-point spread — MS (26%) to DC (13%) |

---

## Tech Stack

| Category | Tools |
|---|---|
| Data loading | DuckDB, pandas |
| Data storage | Parquet (fastparquet) |
| Analysis | pandas, numpy |
| Visualization | matplotlib, seaborn, Tableau |
| Modeling | scikit-learn, XGBoost |
| Presentation | Canva |
| Version control | GitHub |
| Environment | JupyterLab, Python 3.x |

---

## Presentation

[View the full presentation on Canva](https://www.canva.com/your-link-here)

---

## Project Structure

```
lending-club-credit-risk/
├── README.md
├── .gitignore
├── notebooks/
│   ├── 01_data_preparation.ipynb    # DuckDB load, dtype optimization, parquet save
│   ├── 02_eda.ipynb                 # EDA analyses and charts
│   └── 03_modeling.ipynb           # Feature engineering, XGBoost, policy simulation
├── charts/
│   ├── default_rate_by_grade.png
│   ├── default_rate_by_vintage.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── roc_curve_comparison_final.png
│   └── policy_simulation.png
├── data/
│   ├── policy_simulation.csv        # 8-row threshold analysis table
│   └── roc_curves_final.csv         # ROC curve data for visualization
├── tableau/
│   └── LendingClub_CreditRisk.twbx  # Tableau workbook
└── docs/
    └── Symons_Jeffrey_CapstoneProposal.md
```

**Note:** Raw CSV, parquet, and Tableau export files are excluded due to file size.
Run the notebooks in order to regenerate all data files from the Kaggle source data.

---

## How to Run

1. Download the Lending Club dataset from Kaggle:
   https://www.kaggle.com/datasets/wordsforthewise/lending-club

2. Place `lending_club.csv` in the project root directory

3. Install dependencies:
   ```bash
   pip install pandas numpy duckdb fastparquet scikit-learn xgboost matplotlib seaborn
   ```

4. Create required folders:
   ```bash
   mkdir charts data tableau
   ```

5. Run notebooks in order:
   ```
   01_data_preparation.ipynb  →  generates lending_club_v2_correlated.parquet
   02_eda.ipynb               →  generates all EDA charts
   03_modeling.ipynb          →  generates model results, policy simulation, tableau export
   ```

---

## Important Notes

**Data leakage prevention:** All model features are restricted to information available
at loan origination. Post-origination variables (payment history, recovery amounts,
outstanding principal) are excluded.

**Circularity:** Lending Club's grade, sub-grade, and interest rate are excluded from
modeling. These are derived from the same origination-time attributes used as features — including
them would replicate Lending Club's existing risk system rather than build an independent
credit model. Removing grade costs only 0.009 AUC, confirming the raw bureau data
carries equivalent independent signal.

**Vintage coverage:** Sixteen columns were dropped due to vintage coverage issues —
data not collected by Lending Club until December 2015 or August 2012, introducing
vintage bias if included. Pre-August 2012 rows were dropped (4.92% of data) to retain
35 bureau attributes only available from 2012 onward.

**Dollar figures:** Policy simulation dollar amounts are illustrative, using assumed
loss given default (LGD) of 70% and net yield of 5% on approved loans. Actual figures
require lender-specific loss and revenue data.

---

## Background

This project was completed as the capstone for the Texas State University Data Analytics
Bootcamp. It demonstrates a full data science pipeline applied to a real-world credit
risk problem — from raw data engineering through exploratory analysis, machine learning
modeling, and business-oriented policy simulation.

The author has 25+ years of experience in consumer credit and auto lending, including
director-level roles in credit risk management. Domain expertise informed every
analytical decision in this project — from feature selection and outlier treatment
to model interpretation and policy recommendation.

---

*Texas State University Data Analytics Bootcamp | July 2026*
