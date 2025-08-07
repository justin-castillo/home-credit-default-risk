This project develops a robust credit scoring pipeline to predict the probability of loan applicants defaulting, using comprehensive financial and demographic data provided by Home Credit. The primary goal is to effectively identify applicants at higher risk of default, assisting lenders in reducing losses and making better-informed credit decisions.

---

## Project Results & Highlights

| Model                  | ROC-AUC | Precision | Recall | F1 Score |
|------------------------|---------|-----------|--------|----------|
| **CatBoost (Final CV)** | **0.774** | **0.267** | **0.376** | **0.313** |
| XGBoost                | 0.761   | 0.235     | 0.401  | 0.297    |
| LightGBM               | 0.763   | 0.295     | 0.302  | 0.298    |
| Random Forest          | 0.740   | 0.222     | 0.427  | 0.292    |
| Logistic Regression    | 0.722   | 0.198     | 0.431  | 0.271    |

*The final CatBoost model was selected due to its highest ROC-AUC and strong recall, effectively identifying a significant proportion of high-risk clients.*

---

## Detailed Project Overview

### Dataset
The project utilized the complete [Home Credit Default Risk dataset](https://www.kaggle.com/c/home-credit-default-risk), integrating multiple relational tables:
- Bureau data (credit history from other institutions)
- Previous loan applications
- Installment payments history
- Credit card usage data
- POS cash balances

### Methodology
1. **Exploratory Data Analysis**
    - Identified critical features: external credit scores, employment history, loan payment behavior.
2. **Feature Engineering**
    - Engineered over 380 predictive features, including financial ratios (e.g., credit-to-income, annuity-to-income) and behavioral indicators (payment timeliness, delinquency).
3. **Modeling & Evaluation**
    - Tested several machine learning models: Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost.
    - Selected **CatBoost** due to superior handling of categorical data and highest model performance metrics.
4. **Hyperparameter Optimization**
    - Utilized Optuna for hyperparameter tuning.
5. **Threshold Tuning & Validation**
    - Used stratified K-fold cross-validation and optimized classification thresholds to maximize F1-score and recall.
6. **Explainability & Interpretability**
    - Applied SHAP analysis to interpret model predictions, providing insights into influential factors.

### Tools & Technologies
- **Programming:** Python
- **Libraries:** CatBoost, XGBoost, LightGBM, Optuna, SHAP, Pandas, NumPy, Matplotlib, Seaborn
- **CI/CD:** GitHub Actions
- **Environment & Versioning:** Conda, Docker, DVC

---

## Key Insights & Business Impact
- Achieved strong predictive performance with a ROC-AUC of **0.782**, significantly capturing at-risk clients (Recall: **0.683**).
- SHAP analysis ensured transparency, clearly communicating feature impacts to stakeholders.

---

## Quick Start Guide

**To reproduce the pipeline locally:**

```bash
conda env create -f environment.yml
conda activate homecredit

python src/features/build_features.py
python src/training/train.py
python src/predict.py --output submission.csv
```

---

## Project Structure

    .
    ├── data/                   # Raw and processed data
    ├── notebooks/              # Exploratory and modeling notebooks
    ├── models/                 # Trained models and metadata
    ├── src/                    # Project source code
    ├── reports/                # Generated reports and visuals
    ├── environment.yml         # Conda environment setup
    └── Dockerfile              # Docker container definition

## Author

**Justin Castillo**  
Email: [jcastillo.hotels@gmail.com](mailto:jcastillo.hotels@gmail.com)  
GitHub: [github.com/justin-castillo](https://github.com/justin-castillo)  
LinkedIn: [linkedin.com/in/justin-castillo-69351198](https://www.linkedin.com/in/justin-castillo-69351198/)

