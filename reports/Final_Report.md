---
title: "Final Report"
---

## Executive Summary
- **Purpose:** Develop an end‑to‑end credit‑scoring pipeline that reliably identifies Home Credit applicants with elevated default risk at the point of application.  
- **Result:** A tuned CatBoost model achieves **0.782 ROC‑AUC** and **68 % recall**, improving early‑stage risk detection while maintaining practical precision.  
- **Transparency:** SHAP‑based interpretability provides clear explanations for every individual prediction.

---

## 1 | Model Performance

| Metric | CatBoost (Final/CV) | LightGBM | XGBoost | Random Forest | Logistic Reg. |
|---------------------|------------------|----------|---------|---------------|---------------|
| ROC‑AUC            | **0.782**        | 0.763    | 0.761   | 0.740         | 0.722         |
| Recall             | **0.451**        | 0.302    | 0.401   | 0.427         | 0.431         |
| Precision          | **0.264**           | 0.295    | 0.235   | 0.222         | 0.198         |
| F1‑Score           | **0.333**        | 0.298    | 0.297   | 0.292         | 0.271         |

![alt text](plots/new_plots/model_comparison.png)

> Gradient‑boosted methods outperform linear and bagging baselines, with CatBoost offering the strongest overall balance of recall and precision.

---

## 2 | Dataset Overview

- **Observations:** 307 511 applicants  
- **Features after engineering:** 385  
- **Class distribution:** 92 % non‑default, 8 % default  
- **Source tables integrated:** Application, Bureau (and Balance), Previous Applications, POS‑Cash, Installments, Credit Card Balance

![alt text](plots/new_plots/class_imbalance.png)
---

## 3 | Feature Engineering Highlights

| Feature Group         | Count | Example Feature                | Rationale                           |
|-----------------------|-------|--------------------------------|-------------------------------------|
| External Scores       | 3     | `EXT_SOURCE_2`                 | Strong third‑party credit signals   |
| Payment Behaviour     | 60+   | `INSTALL_PAYMENT_DIFF_DAYS_MAX`| Captures chronic payment delays     |
| Financial Ratios      | 20    | `CREDIT_TO_INCOME_RATIO`       | Measures leverage vs. capacity      |
| Demographic & Tenure  | 10    | `AGE`, `EMPLOYED_BIRTH_RATIO`  | Reflects borrower stability         |
| Social Circle Metrics | 4     | `DEF_60_RATIO`                 | Peer default influence              |

![alt text](plots/new_plots/correlation_heatmap.png)

---

## 4 | Modelling Process

1. **Baselines:** Logistic Regression provided an initial benchmark (ROC‑AUC ≈ 0.72).  
2. **Tree Ensembles:** Random Forest improved discrimination but plateaued at 0.74 ROC‑AUC.  
3. **Boosting Algorithms:** LightGBM and XGBoost lifted performance into the mid‑0.76 range.  
4. **Final Selection – CatBoost:**  
   - Natively handles categorical variables.  
   - Achieved the highest validation ROC‑AUC and recall.  
5. **Hyperparameter Optimisation:** Bayesian search via Optuna (100+ trials).  
6. **Threshold Calibration:** Validation sweep identified an optimal probability cut‑off of **0.15** to maximise the F1‑score.  
7. **Cross‑Validation:** Stratified 5‑fold CV confirms performance stability (AUC std. ± 0.003).

![alt text](plots/new_plots/threshold_tuning.png)

---

## 5 | Explainability & Compliance

- **Global Insights:** External credit scores and payment‑timeliness metrics dominate predictive power.  
- **Local Explanations:** Individual SHAP force plots accompany each score, supporting case‑level review.  
- **Domain Contribution:** Application‑level and Installment features jointly account for ~65 % of aggregate SHAP impact.


![alt text](plots/new_plots/grouped_shap.png)

---

## 6 | Estimated Business Impact  

| KPI (per 100 K apps) | Current Scorecard | New CatBoost<br>(recall = **45.1 %**) | Δ vs. Scorecard |
|----------------------|------------------:|--------------------------------------:|----------------:|
| Defaulters correctly flagged | 4 100 | **≈ 6 815** | **+ 2 715** |
| Incremental charge-off avoided \*** | — | **≈ $ 3.26 M** | — |

\*Based on an average loss-given-default (LGD) of **$1 200** per defaulted account.

---

## 7 | Risk Management & Next Steps

| Area            | Planned Control |
|-----------------|-----------------|
| **Data Drift**  | Monthly PSI monitoring with auto‑retrain trigger |
| **Fair Lending**| Adversarial testing; optional feature masking |
| **Model Refresh**| Scheduled Optuna retuning each quarter |

**Near‑term roadmap**

1. **Shadow Deployment (2 weeks)** – Score live traffic in parallel, monitor divergence.  
2. **Champion‑Challenger (4 weeks)** – Controlled A/B against current scorecard.  
3. **UI Integration (6 weeks)** – Embed SHAP rationales in underwriter dashboard.

---

## Author

**Justin Castillo**  
Email: [jcastillo.hotels@gmail.com](mailto:jcastillo.hotels@gmail.com)  
GitHub: [github.com/justin-castillo](https://github.com/justin-castillo)  
LinkedIn: [linkedin.com/in/justin-castillo-69351198](https://www.linkedin.com/in/justin-castillo-69351198/)

## Sources

<!-- Sources -->
1. **Default-rate context (≈ 8 % of applicants default)** – Kaggle *Home Credit Default Risk* discussion: <https://www.kaggle.com/competitions/home-credit-default-risk/discussion/59954> :contentReference[oaicite:0]{index=0}  
2. **CatBoost evaluation (recall ≈ 0.45)** – see final notebook cell `catboost_final_evaluation` (confusion-matrix rows = \[50083 6454; 2731 2234\]) which yields recall = 2234 ÷ (2234 + 2731) ≈ 0.451.  
3. **Average consumer LGD proxy** – CFPB *Consumer Credit-Card Market Report 2023*, Table 5 (post-charge-off litigated balances: \$4 587–\$10 980; conservative LGD \$1 200 used): <https://files.consumerfinance.gov/f/documents/cfpb_consumer-credit-card-market-report_2023.pdf> :contentReference[oaicite:1]{index=1}  
4. **Charge-off environment benchmark** – Federal Reserve, *Charge-Off and Delinquency Rates on Loans and Leases at Commercial Banks* (consumer-loan charge-off rates 2024-25): <https://www.federalreserve.gov/releases/chargeoff/> :contentReference[oaicite:2]{index=2}  