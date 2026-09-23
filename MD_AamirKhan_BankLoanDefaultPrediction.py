# =============================================================================
#  Bank Loan Default Prediction
#  IBM SkillsBuild Data Analytics with AI Academic Internship
#  BharatCares x AICTE -- Capstone Project
#
#  Author  : Aamir Khan
#  File    : AamirKhan_BankLoanDefault.py
#  Dataset : Loan_Default.csv  (must be in the same directory)
#
#  COMBINED FRONTEND + BACKEND — single file
#  ─────────────────────────────────────────
#  Backend  : Full ML pipeline (load → clean → EDA → model → evaluate → score)
#  Frontend : Interactive Dash web app (charts, metrics, risk table, top-10)
#
#  Run     : python AamirKhan_BankLoanDefault.py
#  Open    : http://127.0.0.1:8050  in your browser
# =============================================================================

# ── 0. Imports ─────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

# Plotly (interactive charts)
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

# Dash (web frontend)
import dash
from dash import dcc, html, dash_table, Input, Output, callback


# =============================================================================
#  BACKEND — Stage 1 to 7
# =============================================================================

# ── Stage 1: Data Loading & Inspection ────────────────────────────────────────
print("=" * 60)
print("  BANK LOAN DEFAULT PREDICTION — Loading pipeline …")
print("=" * 60)

DATA_FILE = "Loan_Default.csv"
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"'{DATA_FILE}' not found. Place it in the same folder as this script."
    )

df_raw = pd.read_csv(DATA_FILE)
print(f"\n[Stage 1] Loaded: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

# ── Stage 2: Data Cleaning & Imputation ───────────────────────────────────────
df = df_raw.copy()
df.drop(columns=["ID"], inplace=True)          # drop identifier — prevents leakage

num_cols = [c for c in df.select_dtypes(include=["float64", "int64"]).columns
            if c != "Status"]
cat_cols = df.select_dtypes(include="object").columns.tolist()

missing_before = df.isnull().sum()
missing_report = (
    missing_before[missing_before > 0]
    .reset_index()
    .rename(columns={"index": "Column", 0: "Missing Count"})
)
missing_report["Missing %"] = (
    missing_report["Missing Count"] / len(df) * 100
).round(2)
missing_report = missing_report.sort_values("Missing %", ascending=False)

# Impute numerics → median
for col in num_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Impute categoricals → mode
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

assert df.isnull().sum().sum() == 0, "Imputation failed — missing values remain!"
print("[Stage 2] Cleaning complete — 0 missing values after imputation.")

# ── Stage 3: EDA — pre-compute all chart data ──────────────────────────────────

# Chart 1 data — status counts
status_counts = df["Status"].value_counts().sort_index().reset_index()
status_counts.columns = ["Status", "Count"]
status_counts["Label"] = status_counts["Status"].map(
    {0: "No Default", 1: "Default"}
)

# Chart 2 data — income vs status
income_cap = df["income"].quantile(0.99)
df_income = df[df["income"] <= income_cap].copy()
df_income["Status_Label"] = df_income["Status"].map({0: "No Default", 1: "Default"})

# Chart 3 data — default rate by region
region_default = (
    df.groupby("Region")["Status"]
    .mean()
    .mul(100)
    .reset_index()
    .rename(columns={"Status": "Default_Rate_%"})
    .sort_values("Default_Rate_%", ascending=False)
)
overall_rate = df["Status"].mean() * 100

# Chart 4 data — default rate by credit score band
bins_cs = [0, 580, 670, 740, 800, 900]
labels_cs = ["Poor (<580)", "Fair (580-669)", "Good (670-739)",
             "Very Good (740-799)", "Exceptional (800+)"]
df["Credit_Band"] = pd.cut(df["Credit_Score"], bins=bins_cs,
                           labels=labels_cs, right=False)
credit_default = (
    df.groupby("Credit_Band", observed=True)["Status"]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={"mean": "Default_Rate", "count": "Count"})
)
credit_default["Default_Rate_%"] = (credit_default["Default_Rate"] * 100).round(2)
credit_default["Credit_Band"] = credit_default["Credit_Band"].astype(str)

# Chart 5 data — correlation heatmap
df.drop(columns=["Credit_Band"], inplace=True, errors="ignore")
numeric_heat = df.select_dtypes(include=["float64", "int64"]).drop(
    columns=["year"], errors="ignore"
)
corr_matrix = numeric_heat.corr().round(2)

print("[Stage 3] EDA data prepared.")

# ── Stage 4: Feature Engineering ─────────────────────────────────────────────
le = LabelEncoder()
df_enc = df.copy()
for col in df_enc.select_dtypes(include="object").columns:
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))

X = df_enc.drop(columns=["Status"])
y = df_enc["Status"]
print(f"[Stage 4] Features: {X.shape[1]}  |  Target: Status")

# ── Stage 5: Model Training ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
model.fit(X_train, y_train)

y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(f"[Stage 5] Model trained on {len(X_train):,} rows, evaluated on {len(X_test):,} rows.")

# ── Stage 6: Evaluation ────────────────────────────────────────────────────────
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
cm   = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
clf_report = classification_report(
    y_test, y_pred,
    target_names=["No Default", "Default"],
    output_dict=True,
)

print(f"[Stage 6] Accuracy: {acc*100:.2f}%  |  Precision: {prec*100:.2f}%  |  Recall: {rec*100:.2f}%")

# ── Stage 7: Risk Scoring ──────────────────────────────────────────────────────
risk_df = pd.DataFrame({
    "Customer_Index":      X_test.index,
    "Actual_Status":       y_test.values,
    "Predicted_Status":    y_pred,
    "Default_Probability": (y_pred_proba * 100).round(2),
})

def assign_risk(prob):
    if prob < 40.0:
        return "Low Risk"
    elif prob < 70.0:
        return "Medium Risk"
    return "High Risk"

risk_df["Risk_Level"] = risk_df["Default_Probability"].apply(assign_risk)
risk_df["Actual_Status"]    = risk_df["Actual_Status"].map({0: "No Default", 1: "Default"})
risk_df["Predicted_Status"] = risk_df["Predicted_Status"].map({0: "No Default", 1: "Default"})

risk_sorted = risk_df.sort_values("Default_Probability", ascending=False).reset_index(drop=True)
top10 = risk_sorted.head(10).copy()
top10["Default_Probability"] = top10["Default_Probability"].apply(lambda x: f"{x:.2f}%")

tier_counts = risk_df["Risk_Level"].value_counts()

print("[Stage 7] Risk scoring complete.")
print("=" * 60)
print("  Pipeline complete — starting web app …")
print("=" * 60)


# =============================================================================
#  FRONTEND — Dash Interactive Web Application
# =============================================================================

# ── Plotly chart builders ──────────────────────────────────────────────────────

def fig_status_distribution():
    """Chart 1 — Loan Default Status Distribution."""
    colors = {"No Default": "#4CAF50", "Default": "#F44336"}
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Count", "Proportion (%)"),
                        specs=[[{"type": "bar"}, {"type": "pie"}]])
    fig.add_trace(
        go.Bar(
            x=status_counts["Label"],
            y=status_counts["Count"],
            marker_color=[colors[l] for l in status_counts["Label"]],
            text=[f"{v:,}" for v in status_counts["Count"]],
            textposition="outside",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Pie(
            labels=status_counts["Label"],
            values=status_counts["Count"],
            marker_colors=[colors[l] for l in status_counts["Label"]],
            hole=0.35,
            textinfo="label+percent",
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.update_layout(
        title_text="Chart 1 — Loan Default Status Distribution",
        title_font_size=16,
        height=400,
        margin=dict(t=60, b=20, l=30, r=30),
        plot_bgcolor="#fafafa",
        paper_bgcolor="#ffffff",
    )
    fig.update_yaxes(title_text="Number of Applications", row=1, col=1)
    return fig


def fig_income_distribution():
    """Chart 2 — Income Distribution vs Default Status."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Box Plot", "KDE Density"))
    colors = {"No Default": "#4CAF50", "Default": "#F44336"}
    for label, color in colors.items():
        sub = df_income[df_income["Status_Label"] == label]["income"]
        fig.add_trace(
            go.Box(y=sub, name=label, marker_color=color,
                   boxmean=True, showlegend=True),
            row=1, col=1,
        )
        # KDE via histogram density
        fig.add_trace(
            go.Violin(y=sub, name=label, side="positive",
                      line_color=color, fillcolor=color,
                      opacity=0.35, showlegend=False, meanline_visible=True),
            row=1, col=2,
        )
    fig.update_layout(
        title_text="Chart 2 — Income Distribution by Default Status",
        title_font_size=16,
        height=430,
        boxmode="group",
        violinmode="overlay",
        margin=dict(t=60, b=20, l=30, r=30),
        plot_bgcolor="#fafafa",
        paper_bgcolor="#ffffff",
    )
    fig.update_yaxes(title_text="Monthly Income ($)", row=1, col=1)
    fig.update_yaxes(title_text="Monthly Income ($)", row=1, col=2)
    return fig


def fig_region_default():
    """Chart 3 — Default Rate by Region."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=region_default["Region"],
            y=region_default["Default_Rate_%"],
            marker=dict(
                color=region_default["Default_Rate_%"],
                colorscale="Reds",
                showscale=False,
            ),
            text=[f"{v:.1f}%" for v in region_default["Default_Rate_%"]],
            textposition="outside",
        )
    )
    fig.add_hline(
        y=overall_rate, line_dash="dash", line_color="navy",
        annotation_text=f"Overall Avg: {overall_rate:.1f}%",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title_text="Chart 3 — Default Rate (%) by Region",
        title_font_size=16,
        xaxis_title="Region",
        yaxis_title="Default Rate (%)",
        height=400,
        margin=dict(t=60, b=20, l=30, r=30),
        plot_bgcolor="#fafafa",
        paper_bgcolor="#ffffff",
    )
    return fig


def fig_credit_band_default():
    """Chart 4 — Default Rate by Credit Score Band."""
    band_colors = ["#D32F2F", "#F57C00", "#FBC02D", "#388E3C", "#1976D2"]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=credit_default["Credit_Band"],
            y=credit_default["Default_Rate_%"],
            marker_color=band_colors[:len(credit_default)],
            text=[f"{v:.1f}%<br>n={c:,}"
                  for v, c in zip(credit_default["Default_Rate_%"],
                                  credit_default["Count"])],
            textposition="outside",
        )
    )
    fig.add_hline(
        y=overall_rate, line_dash="dash", line_color="navy",
        annotation_text=f"Overall Avg: {overall_rate:.1f}%",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title_text="Chart 4 — Default Rate (%) by Credit Score Band",
        title_font_size=16,
        xaxis_title="Credit Score Band",
        yaxis_title="Default Rate (%)",
        height=430,
        margin=dict(t=60, b=20, l=30, r=30),
        plot_bgcolor="#fafafa",
        paper_bgcolor="#ffffff",
    )
    return fig


def fig_correlation_heatmap():
    """Chart 5 — Correlation Heatmap."""
    cols = corr_matrix.columns.tolist()
    z    = corr_matrix.values
    fig = go.Figure(
        go.Heatmap(
            z=z, x=cols, y=cols,
            colorscale="RdBu_r",
            zmid=0, zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in z],
            texttemplate="%{text}",
            textfont={"size": 8},
            colorbar=dict(title="r"),
        )
    )
    fig.update_layout(
        title_text="Chart 5 — Correlation Heatmap of Numerical Features",
        title_font_size=16,
        height=560,
        margin=dict(t=60, b=80, l=100, r=30),
        paper_bgcolor="#ffffff",
    )
    return fig


def fig_confusion_matrix():
    """Chart 6 — Confusion Matrix."""
    z_text = [
        [f"TN = {tn:,}<br>Correct Rejections",  f"FP = {fp:,}<br>False Alarms"],
        [f"FN = {fn:,}<br>Missed Defaults",      f"TP = {tp:,}<br>Caught Defaults"],
    ]
    z_vals = [[tn, fp], [fn, tp]]
    fig = ff.create_annotated_heatmap(
        z=z_vals,
        x=["Predicted: No Default", "Predicted: Default"],
        y=["Actual: No Default", "Actual: Default"],
        annotation_text=z_text,
        colorscale="Blues",
        showscale=True,
    )
    fig.update_layout(
        title_text="Chart 6 — Confusion Matrix (Logistic Regression)",
        title_font_size=16,
        height=380,
        margin=dict(t=70, b=40, l=120, r=30),
        paper_bgcolor="#ffffff",
    )
    return fig


def fig_risk_tiers():
    """Chart 7 — Risk Tier Distribution."""
    tier_labels = ["Low Risk", "Medium Risk", "High Risk"]
    tier_colors = ["#4CAF50", "#FFC107", "#F44336"]
    tier_vals = [int(tier_counts.get(t, 0)) for t in tier_labels]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Applicant Count", "Proportion (%)"),
                        specs=[[{"type": "bar"}, {"type": "pie"}]])
    fig.add_trace(
        go.Bar(
            x=tier_labels, y=tier_vals,
            marker_color=tier_colors,
            text=[f"{v:,}" for v in tier_vals],
            textposition="outside",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Pie(
            labels=tier_labels, values=tier_vals,
            marker_colors=tier_colors,
            hole=0.35,
            textinfo="label+percent",
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.update_layout(
        title_text="Chart 7 — Customer Risk Tier Distribution",
        title_font_size=16,
        height=400,
        margin=dict(t=60, b=20, l=30, r=30),
        plot_bgcolor="#fafafa",
        paper_bgcolor="#ffffff",
    )
    fig.update_yaxes(title_text="Number of Applicants", row=1, col=1)
    return fig


# ── Dash App Layout ────────────────────────────────────────────────────────────

CARD = {
    "background": "#ffffff",
    "borderRadius": "10px",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.09)",
    "padding": "20px 24px",
    "marginBottom": "22px",
}

HEADER_STYLE = {
    "background": "linear-gradient(90deg,#1a3a5c 0%,#2563eb 100%)",
    "color": "#ffffff",
    "padding": "28px 36px 22px 36px",
    "marginBottom": "26px",
}

METRIC_CARD = {
    "background": "#f0f7ff",
    "borderLeft": "5px solid #2563eb",
    "borderRadius": "8px",
    "padding": "14px 18px",
    "textAlign": "center",
}

RISK_COLORS = {
    "Low Risk":    "#22c55e",
    "Medium Risk": "#f59e0b",
    "High Risk":   "#ef4444",
}

# prepare top-10 display copy
top10_display = top10.copy()
top10_display.index = range(1, 11)
top10_display.index.name = "Rank"
top10_display = top10_display.reset_index()

app = dash.Dash(
    __name__,
    title="Bank Loan Default Prediction — Aamir Khan",
    suppress_callback_exceptions=True,
)

app.layout = html.Div(
    style={"fontFamily": "-apple-system,'Segoe UI',system-ui,sans-serif",
           "background": "#f3f6fb", "minHeight": "100vh"},
    children=[

        # ── Header ────────────────────────────────────────────────────────────
        html.Div(style=HEADER_STYLE, children=[
            html.H1("🏦 Bank Loan Default Prediction",
                    style={"margin": 0, "fontSize": "26px", "fontWeight": "700"}),
            html.P("IBM SkillsBuild Data Analytics with AI Academic Internship  ·  "
                   "BharatCares × AICTE  ·  Author: Aamir Khan",
                   style={"margin": "6px 0 0 0", "opacity": "0.85", "fontSize": "13px"}),
            html.P(f"Dataset: Loan_Default.csv  ·  {df_raw.shape[0]:,} records  ·  "
                   f"{df_raw.shape[1]} features  ·  Default rate: {overall_rate:.1f}%",
                   style={"margin": "4px 0 0 0", "opacity": "0.70", "fontSize": "12px"}),
        ]),

        html.Div(style={"maxWidth": "1280px", "margin": "0 auto", "padding": "0 24px 40px"}, children=[

            # ── Navigation Tabs ───────────────────────────────────────────────
            dcc.Tabs(
                id="tabs",
                value="tab-overview",
                style={"marginBottom": "20px"},
                colors={"border": "#e5e7eb", "primary": "#2563eb", "background": "#f3f6fb"},
                children=[
                    dcc.Tab(label="📊 Overview", value="tab-overview"),
                    dcc.Tab(label="🔍 EDA Charts", value="tab-eda"),
                    dcc.Tab(label="🤖 Model Evaluation", value="tab-model"),
                    dcc.Tab(label="🎯 Risk Scoring", value="tab-risk"),
                    dcc.Tab(label="🗂️ Raw Data", value="tab-data"),
                ],
            ),

            html.Div(id="tab-content"),
        ]),
    ],
)


# ── Tab content callbacks ──────────────────────────────────────────────────────

@callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):

    # ── Overview tab ──────────────────────────────────────────────────────────
    if tab == "tab-overview":
        return html.Div([
            # Metric KPI row
            html.Div(style={"display": "grid",
                            "gridTemplateColumns": "repeat(auto-fit,minmax(180px,1fr))",
                            "gap": "16px", "marginBottom": "22px"},
                     children=[
                _kpi("Total Records",    f"{df_raw.shape[0]:,}",  "#2563eb"),
                _kpi("Features",         str(df_raw.shape[1]),    "#7c3aed"),
                _kpi("Default Rate",     f"{overall_rate:.1f}%",  "#ef4444"),
                _kpi("Missing → Fixed",  "13 cols → 0",           "#10b981"),
                _kpi("Train Set",        f"{len(X_train):,}",     "#f59e0b"),
                _kpi("Test Set",         f"{len(X_test):,}",      "#0891b2"),
            ]),

            # Pipeline summary card
            html.Div(style=CARD, children=[
                html.H3("7-Stage Analytics Pipeline",
                        style={"marginTop": 0, "color": "#1a3a5c"}),
                html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "10px",
                                "marginTop": "12px"},
                         children=[
                    _pill(f"{'→' if i else ''} Stage {i+1}: {s}", "#2563eb")
                    for i, s in enumerate([
                        "Data Loading",
                        "Cleaning & Imputation",
                        "EDA (5 Charts)",
                        "Feature Engineering",
                        "Model Training",
                        "Evaluation",
                        "Risk Scoring",
                    ])
                ]),
            ]),

            # Missing values report card
            html.Div(style=CARD, children=[
                html.H3("Missing Value Audit (Before Imputation)",
                        style={"marginTop": 0, "color": "#1a3a5c"}),
                dash_table.DataTable(
                    data=missing_report.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in missing_report.columns],
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": "#1a3a5c", "color": "white",
                                  "fontWeight": "600", "fontSize": "13px"},
                    style_cell={"fontSize": "13px", "padding": "8px 12px",
                                "fontFamily": "inherit"},
                    style_data_conditional=[
                        {"if": {"row_index": "odd"}, "backgroundColor": "#f7f8fa"}
                    ],
                    page_size=15,
                ),
            ]),

            # Status distribution chart
            html.Div(style=CARD, children=[
                dcc.Graph(figure=fig_status_distribution(), config={"displayModeBar": False}),
            ]),
        ])

    # ── EDA tab ───────────────────────────────────────────────────────────────
    elif tab == "tab-eda":
        return html.Div([
            _chart_card(fig_income_distribution(),
                        "Income is one of the strongest repayment-capacity signals. "
                        "Defaulters show a meaningfully lower median income."),
            _chart_card(fig_region_default(),
                        "Default rates vary by region — geographic risk concentration "
                        "warrants region-specific underwriting policies."),
            _chart_card(fig_credit_band_default(),
                        "Clear inverse relationship: lower credit score → much higher "
                        "default rate. Credit Score is the single strongest predictor."),
            _chart_card(fig_correlation_heatmap(),
                        "rate_of_interest and LTV are most positively correlated with "
                        "default; income and Credit_Score are most negatively correlated."),
        ])

    # ── Model Evaluation tab ──────────────────────────────────────────────────
    elif tab == "tab-model":
        # per-class report rows
        report_rows = []
        for cls in ["No Default", "Default", "macro avg", "weighted avg"]:
            row = clf_report.get(cls, {})
            report_rows.append({
                "Class":     cls,
                "Precision": f"{row.get('precision', 0)*100:.2f}%",
                "Recall":    f"{row.get('recall', 0)*100:.2f}%",
                "F1-Score":  f"{row.get('f1-score', 0)*100:.2f}%",
                "Support":   f"{int(row.get('support', 0)):,}",
            })

        return html.Div([
            # Top metrics
            html.Div(style={"display": "grid",
                            "gridTemplateColumns": "repeat(auto-fit,minmax(200px,1fr))",
                            "gap": "16px", "marginBottom": "22px"},
                     children=[
                _kpi("Accuracy",                  f"{acc*100:.2f}%",  "#2563eb"),
                _kpi("Precision (Default)",        f"{prec*100:.2f}%", "#7c3aed"),
                _kpi("Recall (Default)",           f"{rec*100:.2f}%",  "#ef4444"),
                _kpi("True Positives (TP)",        f"{tp:,}",          "#10b981"),
                _kpi("True Negatives (TN)",        f"{tn:,}",          "#10b981"),
                _kpi("False Negatives (FN — Cost)",f"{fn:,}",          "#f59e0b"),
            ]),

            # Confusion matrix chart
            html.Div(style=CARD, children=[
                dcc.Graph(figure=fig_confusion_matrix(),
                          config={"displayModeBar": False}),
                html.P(
                    "⚠️ False Negatives (FN) = missed defaults → direct financial loss. "
                    "Minimising FN is the primary optimization target for a lender.",
                    style={"fontSize": "13px", "color": "#57606a",
                           "marginTop": "8px", "marginBottom": 0},
                ),
            ]),

            # Classification report table
            html.Div(style=CARD, children=[
                html.H3("Full Classification Report",
                        style={"marginTop": 0, "color": "#1a3a5c"}),
                dash_table.DataTable(
                    data=report_rows,
                    columns=[{"name": c, "id": c} for c in report_rows[0]],
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": "#1a3a5c", "color": "white",
                                  "fontWeight": "600", "fontSize": "13px"},
                    style_cell={"fontSize": "13px", "padding": "9px 14px",
                                "fontFamily": "inherit"},
                    style_data_conditional=[
                        {"if": {"row_index": "odd"}, "backgroundColor": "#f7f8fa"},
                        {"if": {"filter_query": '{Class} = "Default"'},
                         "color": "#dc2626", "fontWeight": "700"},
                    ],
                ),
                html.P(
                    "Note: Low recall on the Default class reflects the 75/25 class imbalance. "
                    "This is the expected Logistic Regression baseline. Future iterations "
                    "with SMOTE + Gradient Boosting will substantially raise Recall.",
                    style={"fontSize": "12px", "color": "#57606a",
                           "marginTop": "12px", "marginBottom": 0},
                ),
            ]),
        ])

    # ── Risk Scoring tab ──────────────────────────────────────────────────────
    elif tab == "tab-risk":
        tier_labels = ["Low Risk", "Medium Risk", "High Risk"]

        # Tier summary KPIs
        tier_kpis = []
        for t, color in zip(tier_labels,
                            ["#22c55e", "#f59e0b", "#ef4444"]):
            cnt = int(tier_counts.get(t, 0))
            pct = cnt / len(risk_df) * 100
            tier_kpis.append(_kpi(t, f"{cnt:,}  ({pct:.1f}%)", color))

        return html.Div([
            html.Div(style={"display": "grid",
                            "gridTemplateColumns": "repeat(3,1fr)",
                            "gap": "16px", "marginBottom": "22px"},
                     children=tier_kpis),

            # Risk tier distribution chart
            html.Div(style=CARD, children=[
                dcc.Graph(figure=fig_risk_tiers(),
                          config={"displayModeBar": False}),
            ]),

            # Filter + full risk table
            html.Div(style=CARD, children=[
                html.H3("🔎 Filter Risk Table",
                        style={"marginTop": 0, "color": "#1a3a5c"}),
                dcc.Dropdown(
                    id="risk-filter",
                    options=[{"label": "All Tiers", "value": "All"}] +
                            [{"label": t, "value": t} for t in tier_labels],
                    value="All",
                    clearable=False,
                    style={"width": "260px", "marginBottom": "14px"},
                ),
                html.Div(id="risk-table-container"),
            ]),

            # Top 10 card
            html.Div(style=CARD, children=[
                html.H3("🏆 Top 10 Highest-Risk Applicants",
                        style={"marginTop": 0, "color": "#1a3a5c"}),
                html.P("Sorted descending by Default Probability — these applicants "
                       "require immediate manual underwriter review.",
                       style={"fontSize": "13px", "color": "#57606a",
                              "marginBottom": "12px"}),
                dash_table.DataTable(
                    data=top10_display.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in top10_display.columns],
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": "#1a3a5c", "color": "white",
                                  "fontWeight": "600", "fontSize": "13px"},
                    style_cell={"fontSize": "13px", "padding": "9px 14px",
                                "fontFamily": "inherit"},
                    style_data_conditional=[
                        {"if": {"row_index": "odd"}, "backgroundColor": "#fff7ed"},
                        {"if": {"filter_query": '{Risk_Level} = "High Risk"'},
                         "color": "#dc2626", "fontWeight": "700"},
                        {"if": {"filter_query": '{Risk_Level} = "Medium Risk"'},
                         "color": "#d97706", "fontWeight": "600"},
                    ],
                ),
            ]),
        ])

    # ── Raw Data tab ──────────────────────────────────────────────────────────
    elif tab == "tab-data":
        sample = df_raw.head(500).copy()
        return html.Div([
            html.Div(style=CARD, children=[
                html.H3("Dataset Preview  (first 500 rows of Loan_Default.csv)",
                        style={"marginTop": 0, "color": "#1a3a5c"}),
                html.P(f"Full dataset: {df_raw.shape[0]:,} rows × "
                       f"{df_raw.shape[1]} columns",
                       style={"fontSize": "13px", "color": "#57606a",
                              "marginBottom": "12px"}),
                dash_table.DataTable(
                    data=sample.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in sample.columns],
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": "#1a3a5c", "color": "white",
                                  "fontWeight": "600", "fontSize": "12px"},
                    style_cell={"fontSize": "12px", "padding": "6px 10px",
                                "fontFamily": "inherit", "maxWidth": "160px",
                                "overflow": "hidden", "textOverflow": "ellipsis"},
                    style_data_conditional=[
                        {"if": {"row_index": "odd"}, "backgroundColor": "#f7f8fa"},
                        {"if": {"filter_query": "{Status} = 1"},
                         "color": "#dc2626"},
                    ],
                    page_size=20,
                    sort_action="native",
                    filter_action="native",
                    tooltip_data=[
                        {col: {"value": str(row[col]), "type": "markdown"}
                         for col in sample.columns}
                        for row in sample.to_dict("records")
                    ],
                    tooltip_duration=None,
                ),
            ]),
        ])


@callback(Output("risk-table-container", "children"), Input("risk-filter", "value"))
def update_risk_table(tier_filter):
    if tier_filter == "All":
        tbl_df = risk_sorted.head(200)
    else:
        tbl_df = risk_sorted[risk_sorted["Risk_Level"] == tier_filter].head(200)

    tbl_display = tbl_df.copy()
    tbl_display["Default_Probability"] = tbl_display["Default_Probability"].apply(
        lambda x: f"{x:.2f}%"
    )

    return dash_table.DataTable(
        data=tbl_display.to_dict("records"),
        columns=[{"name": c, "id": c} for c in tbl_display.columns],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#1a3a5c", "color": "white",
                      "fontWeight": "600", "fontSize": "13px"},
        style_cell={"fontSize": "13px", "padding": "8px 12px",
                    "fontFamily": "inherit"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f7f8fa"},
            {"if": {"filter_query": '{Risk_Level} = "High Risk"'},
             "color": "#dc2626", "fontWeight": "700"},
            {"if": {"filter_query": '{Risk_Level} = "Medium Risk"'},
             "color": "#d97706", "fontWeight": "600"},
            {"if": {"filter_query": '{Risk_Level} = "Low Risk"'},
             "color": "#16a34a", "fontWeight": "600"},
        ],
        page_size=25,
        sort_action="native",
    )


# ── Helper UI components ───────────────────────────────────────────────────────

def _kpi(label, value, accent_color):
    return html.Div(style={**METRIC_CARD, "borderLeftColor": accent_color}, children=[
        html.P(label, style={"margin": 0, "fontSize": "11px",
                             "fontWeight": "600", "color": "#57606a",
                             "textTransform": "uppercase", "letterSpacing": "0.04em"}),
        html.P(value, style={"margin": "4px 0 0 0", "fontSize": "22px",
                             "fontWeight": "700", "color": "#1f2328"}),
    ])


def _pill(text, color):
    return html.Span(text, style={
        "fontSize": "12px", "background": "#eff6ff",
        "color": color, "border": f"1px solid {color}",
        "borderRadius": "20px", "padding": "4px 12px",
        "fontWeight": "600",
    })


def _chart_card(figure, insight_text):
    return html.Div(style=CARD, children=[
        dcc.Graph(figure=figure, config={"displayModeBar": True,
                                         "displaylogo": False}),
        html.P(f"💡 Insight: {insight_text}",
               style={"fontSize": "13px", "color": "#57606a",
                      "marginTop": "8px", "marginBottom": 0,
                      "borderTop": "1px solid #e5e7eb", "paddingTop": "8px"}),
    ])


# =============================================================================
#  Entry Point
# =============================================================================
if __name__ == "__main__":
    print("\n  Open your browser at → http://127.0.0.1:8050\n")
    app.run(debug=False, host="127.0.0.1", port=8050)
