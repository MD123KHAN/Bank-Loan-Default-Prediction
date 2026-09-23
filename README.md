# Bank Loan Default Prediction
### IBM SkillsBuild Data Analytics with AI Academic Internship
**BharatCares x AICTE — Capstone Project**

---

## Project Overview

This capstone project is a **single-file, full-stack Python application** that predicts bank loan defaults using a real-world mortgage dataset of **148,670 applications**. The file `AamirKhan_BankLoanDefault.py` contains both the **ML backend pipeline** and an **interactive Dash web frontend** — one command runs everything.

The app opens an interactive dashboard in your browser where you can explore EDA charts, model evaluation metrics, and a filterable risk scoring table — all driven by your actual `Loan_Default.csv` data.

---

## Business Problem & Impact

> *"Can we reliably flag high-risk loan applications **before** disbursement using historical borrower data?"*

- Dataset default rate: **24.6%** (36,639 out of 148,670 applications)
- Every missed default (False Negative) = direct financial write-off
- Model output: probability-based risk tier (Low / Medium / High) per applicant
- Enables: risk-adjusted pricing, portfolio management, regulatory compliance

---

## Dataset Information

| Attribute | Detail |
|---|---|
| **File** | `Loan_Default.csv` |
| **Rows** | 148,670 (one per mortgage application) |
| **Columns** | 34 (33 predictors + 1 target) |
| **Target** | `Status` — 0 = No Default, 1 = Default |
| **Class Balance** | 75.4% No Default / 24.6% Default |
| **Source** | [Kaggle — Loan Default Dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default) |

Key predictors: `Credit_Score`, `income`, `loan_amount`, `LTV`, `rate_of_interest`, `dtir1`, `Region`, `Gender`, `age`, `credit_type`

---

## Tech Stack & Tools

| Category | Library | Version |
|---|---|---|
| Data Manipulation | `pandas` | 2.2.2 |
| Numerical Computing | `numpy` | 1.26.4 |
| Machine Learning | `scikit-learn` | 1.4.2 |
| Interactive Charts | `plotly` | 7.1.0 |
| Web Frontend | `dash` | 4.4.1 |
| Notebook (optional) | `jupyter` / `notebook` | 1.0.0 / 7.2.1 |
| Language | Python | 3.10+ |

---

## Project Structure

```
IBM_Project/
|
|-- AamirKhan_BankLoanDefault.py    <- MAIN FILE: full ML backend + Dash frontend
|-- AamirKhan_ProjectReport.docx    <- Executive project report (9 sections + 7 charts)
|-- requirements.txt                <- Pinned library versions
|-- README.md                       <- This file
|
|-- Loan_Default.csv                <- Raw dataset (input, must be in same folder)
```

**Exactly 4 deliverable files** (excluding the dataset).

---

## Application Architecture

```
AamirKhan_BankLoanDefault.py
|
|-- BACKEND (runs at startup, ~30s on first load)
|   |-- Stage 1: Data Loading      -> pandas.read_csv('Loan_Default.csv')
|   |-- Stage 2: Cleaning          -> median/mode imputation, drop ID
|   |-- Stage 3: EDA               -> pre-compute all 7 Plotly chart datasets
|   |-- Stage 4: Feature Eng.      -> LabelEncoder on 22 categorical columns
|   |-- Stage 5: Model Training    -> LogisticRegression (80/20 split)
|   |-- Stage 6: Evaluation        -> accuracy, precision, recall, confusion matrix
|   `-- Stage 7: Risk Scoring      -> predict_proba + Low/Medium/High tiers
|
`-- FRONTEND (Dash web app on http://127.0.0.1:8050)
    |-- Tab 1: Overview            -> KPI cards, pipeline summary, missing value audit
    |-- Tab 2: EDA Charts          -> 4 interactive charts (income, region, credit, heatmap)
    |-- Tab 3: Model Evaluation    -> Confusion matrix, classification report table
    |-- Tab 4: Risk Scoring        -> Filterable risk table + Top 10 highest-risk applicants
    `-- Tab 5: Raw Data            -> Searchable/sortable full dataset preview
```

---

## Setup & Run Instructions

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Ensure dataset is present

Place `Loan_Default.csv` in the **same folder** as `AamirKhan_BankLoanDefault.py`.

```
IBM_Project/
  AamirKhan_BankLoanDefault.py
  Loan_Default.csv              <- must be here
  requirements.txt
  README.md
```

### Step 3 — Run the application

```bash
python AamirKhan_BankLoanDefault.py
```

### Step 4 — Open the dashboard

```
http://127.0.0.1:8050
```

The terminal will print all 7 pipeline stage confirmations, then display the URL. The ML pipeline runs once at startup (~20-40 seconds for 148k rows) and the app stays live until you press `Ctrl+C`.

### Google Colab (alternative)

```python
# Install dependencies
!pip install dash plotly pandas numpy scikit-learn

# Upload Loan_Default.csv and AamirKhan_BankLoanDefault.py to /content/
# Then run:
!python AamirKhan_BankLoanDefault.py
```

Use Colab's port forwarding or `ngrok` to access the dashboard URL.

---

## Frontend Features

| Tab | What you can do |
|---|---|
| **Overview** | See all KPI metrics at a glance; review which columns had missing values |
| **EDA Charts** | Interact with 4 charts — zoom, pan, hover for exact values |
| **Model Evaluation** | Inspect the confusion matrix heatmap and per-class precision/recall/F1 |
| **Risk Scoring** | Filter by risk tier (Low / Medium / High); see Top 10 riskiest applicants |
| **Raw Data** | Sort and filter all 148,670 rows directly in the browser |

---

## Model Results (Logistic Regression on Loan_Default.csv)

| Metric | Value |
|---|---|
| **Accuracy** | **74.87%** |
| **Precision** (Default class) | **29.46%** |
| **Recall** (Default class) | **1.42%** |
| **True Positives (TP)** | 104 |
| **True Negatives (TN)** | 22,157 |
| **False Positives (FP)** | 249 |
| **False Negatives (FN)** | 7,224 |

> **Note:** Low recall is expected for Logistic Regression on this 75/25 imbalanced dataset. This is the interpretable baseline model. Future work: SMOTE + Random Forest / Gradient Boosting to significantly raise recall.

### Risk Tier Distribution (Test Set — 29,734 applicants)

| Risk Tier | Threshold | Recommended Action |
|---|---|---|
| Low Risk | < 40% probability | Standard approval + routine monitoring |
| Medium Risk | 40% – 69.9% | Conditional approval (higher rate / collateral) |
| High Risk | >= 70% | Escalate to underwriter / Decline |

---

## Top 5 Business Recommendations

1. **Credit Score Hard Floor** — Reject applications with Score < 580 without collateral
2. **DTI Cap at 43%** — Auto-flag applications exceeding the Qualified Mortgage threshold
3. **LTV Risk Premium** — +0.25-0.50% rate for LTV > 80%; require PMI for LTV > 90%
4. **Regional Concentration Limits** — Tighter standards in above-average default regions
5. **AI Pre-Screening Deployment** — Use this app as the first-stage filter in loan origination

---

## Author

**Aamir Khan**
Intern — IBM SkillsBuild Data Analytics with AI Academic Internship

---

## Acknowledgments

| Organization | Role |
|---|---|
| [IBM SkillsBuild](https://skillsbuild.org/) | Learning resources, AI & Data Analytics curriculum |
| [BharatCares](https://bharatcares.org/) | Internship program organization and mentorship |
| [AICTE](https://www.aicte-india.org/) | Academic recognition and program association |
| [Kaggle](https://www.kaggle.com/) | Public dataset hosting |

---

> *"One file. One command. Full pipeline from raw data to interactive risk dashboard."*
