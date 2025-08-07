---
title: "Feature Engineering Notes — Home Credit Default Risk"
---

This document outlines the engineered features used to enhance predictive performance in the default classification model. Features were constructed from both raw application data and multi-level aggregations across 7 relational tables.

---

## From `application_train.csv`

### Ratio Features
- `CREDIT_TO_INCOME_RATIO` = `AMT_CREDIT / AMT_INCOME_TOTAL`  
- `ANNUITY_TO_INCOME_RATIO` = `AMT_ANNUITY / AMT_INCOME_TOTAL`  
- `CREDIT_TO_ANNUITY_RATIO` = `AMT_CREDIT / AMT_ANNUITY`  
- `INCOME_PER_FAM_MEMBER` = `AMT_INCOME_TOTAL / CNT_FAM_MEMBERS`  
- `EMPLOYED_BIRTH_RATIO` = `DAYS_EMPLOYED / DAYS_BIRTH`  

These domain-informed ratios provided strong signal, especially the credit-to-income and annuity-to-income measures, which consistently ranked among top features in SHAP importance across both sample and full datasets.

---

## External Risk Scores

### Combined Source Metrics
- `EXT_SOURCES_MEAN` = mean of EXT_SOURCE_1/2/3  
- `EXT_SOURCES_STD` = standard deviation of EXT_SOURCE_1/2/3  

These variables—especially `EXT_SOURCE_2` and `EXT_SOURCE_3`—were the highest-impact features in both global and local SHAP plots. Their predictive power remained stable in both the sample and full models.

---

## Temporal Ratios

- `REGISTRATION_AGE_RATIO` = `DAYS_REGISTRATION / DAYS_BIRTH`  
- `ID_PUBLISH_AGE_RATIO` = `DAYS_ID_PUBLISH / DAYS_BIRTH`  

Temporal features were moderately predictive and helped improve recall when paired with other behavioral indicators.

---

## From `bureau.csv` + `bureau_balance.csv`

Grouped by `SK_ID_CURR`:
- `NUM_PREV_CREDITS` = Count of bureau credits  
- `TOTAL_BUREAU_DEBT` = Sum of `AMT_CREDIT_SUM_DEBT`  
- `BUREAU_DEBT_TO_CREDIT_RATIO` = Debt / Credit per client  
- `AVG_BUREAU_OVERDUE` = Mean of `AMT_CREDIT_SUM_OVERDUE`  

These aggregated variables added signal about historical indebtedness and delinquency. While not individually dominant, they contributed to improved F1 after threshold tuning.

---

## From `previous_application.csv`

Grouped by `SK_ID_CURR`:
- `NUM_PREV_APPS` = Number of past applications  
- `APPROVAL_RATE` = Ratio of approved to total applications  
- `PREV_APP_CREDIT_MEAN` = Mean approved credit  
- `PREV_APP_DOWNPAYMENT_RATE` = `AMT_DOWN_PAYMENT / AMT_GOODS_PRICE`  

The `APPROVAL_RATE` feature was especially useful in identifying borderline applicants and added explainability in grouped SHAP analysis by domain.

---

## From `credit_card_balance.csv` and `POS_CASH_balance.csv`

Grouped by `SK_ID_CURR`:
- `CC_BALANCE_MEAN` = Mean monthly card balance  
- `POS_DPD_MEAN` = Mean delay from POS loans  
- `NUM_ACTIVE_CARDS` = Count of active credit lines  

These features contributed indirectly via aggregated risk indicators. Their cumulative importance was visualized using grouped SHAP summary plots.

---

## From `installments_payments.csv`

Grouped by `SK_ID_CURR`:
- `MEAN_PAYMENT_DELAY` = Mean of `DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT`  
- `MISSED_PAYMENT_RATIO` = Missed payments / total payments  
- `INSTALMENT_COMPLETION_RATIO` = Total paid / expected payments  

Repayment punctuality features improved recall in the full dataset model, especially when paired with credit behavior from other domains.

---

## Feature Impact Summary

- **Top Features (SHAP):** `EXT_SOURCE_2`, `EXT_SOURCE_3`, `AMT_ANNUITY`, `CREDIT_TO_INCOME_RATIO`, `EMPLOYED_BIRTH_RATIO`
- **Most Improved from Aggregation:** Installment-based delay ratios, previous application approval rate, and bureau-level credit ratios
- **Grouped SHAP Insight:** Application features dominated local impact, but previous application and installment histories contributed disproportionately to positive-class predictions

---

## Final Notes

- All aggregation used `.groupby('SK_ID_CURR')` followed by `mean`, `sum`, `std`, `count`, and domain-specific ratios
- Skewed numeric features were log-transformed when appropriate
- Features were validated by:
  - Null value thresholding
  - Correlation with `TARGET`
  - SHAP and permutation importance
- Feature selection was driven by business logic, model feedback, and impact on recall

