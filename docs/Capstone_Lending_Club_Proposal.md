# Capstone Project Proposal
**Jeffrey Symons | Data Analytics Bootcamp**

---

## 1. Project Information

This project analyzes Lending Club's historical consumer loan performance data to build a
credit risk model capable of predicting loan default at the time of origination. Lending Club
is a peer-to-peer lending platform that originated personal installment loans from 2007
through 2018, making its loan-level performance data one of the most detailed publicly
available consumer credit datasets.

The real-world application mirrors a decision a lender faces every day: given only what is
known about a borrower at application, how accurately can future default be predicted — and
what is the portfolio-level impact of acting on that prediction? The analysis will simulate a
credit policy filter based on model output and estimate the reduction in charge-off exposure
that policy would have produced. This translates model accuracy into business outcomes — the
language risk managers and executives actually use.

---

## 2. Project Goals

The analysis will answer the following questions:

1. **What borrower and loan characteristics at origination are most predictive of
   charge-off?**
   Using classification modeling, identify which application-time variables — FICO range,
   DTI, loan grade, employment length, revolving utilization, delinquency history — carry
   the greatest predictive weight.

2. **How accurately can a model predict default using only origination-time data?**
   Evaluate model performance using precision, recall, ROC-AUC, and F1-score, with
   particular attention to minimizing false negatives (missed defaults) given the asymmetric
   cost of credit losses.

3. **What is the portfolio impact of applying a model-driven credit policy filter?**
   Simulate rejection of loans above a defined risk score threshold and estimate the
   resulting reduction in projected charge-off dollars and charge-off rate — translating
   model output into an actionable business decision.

4. **Are Lending Club's assigned loan grades appropriately calibrated to realized default
   risk?**
   Analyze actual charge-off rates by grade and subgrade against the implied risk premium
   embedded in interest rates, identifying grades where risk was systematically underpriced.

5. **How does default risk vary across loan vintage, term, and borrower segment?**
   Conduct vintage and segmentation analysis to identify whether risk was concentrated in
   specific origination periods, loan terms, or borrower profiles.

---

## 3. Data Overview

**Source:** Kaggle — Lending Club Loan Data (publicly available, sourced from Lending Club's
historical loan-level disclosures)

**Size:** Approximately 2.6 million rows, 100+ columns, ~1.1 GB raw CSV

**Format:** CSV; due to file size, initial data loading and filtering will be handled via
DuckDB before ingestion into pandas, with the cleaned working dataset persisted in Parquet
format for efficient repeated access throughout the project.

**Key Variables:**

| Category | Variables |
|---|---|
| Target | `loan_status` (binarized: Charged Off = 1, Fully Paid = 0) |
| Borrower Risk | `fico_range_low`, `fico_range_high`, `dti`, `annual_inc`, `emp_length`, `delinq_2yrs`, `pub_rec`, `revol_util`, `open_acc` |
| Loan Structure | `loan_amnt`, `term`, `int_rate`, `grade`, `sub_grade`, `purpose` |
| Verification | `verification_status`, `home_ownership` |
| Performance | `recoveries`, `total_rec_prncp`, `total_pymnt`, `collection_recovery_fee` |
| Vintage | `issue_d`, `earliest_cr_line` |

**Data Quality Considerations:**

- Dataset will be filtered to fully resolved loan statuses only (Charged Off / Fully Paid)
  to ensure clean binary classification labels — active and late-stage loans will be excluded
- Significant missing data is expected in several columns; imputation strategy and column
  exclusions will be documented during EDA
- Post-origination variables (e.g. `last_pymnt_amnt`, `recoveries`) will be excluded from
  predictive features to prevent data leakage — model inputs will be restricted to
  information available at the time of origination
- Class imbalance is expected (defaults are a minority class); handling strategy (SMOTE,
  class weighting, or threshold tuning) will be determined during the modeling phase

**Tools & Techniques:**
Python (pandas, scikit-learn, matplotlib, seaborn), DuckDB, Parquet, JupyterLab, Tableau,
GitHub