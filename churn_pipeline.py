"""
End-to-End ML Pipeline for Customer Churn Prediction
Using Scikit-learn Pipeline API with GridSearchCV
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
from io import StringIO

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score, f1_score, ConfusionMatrixDisplay
)

# ─────────────────────────────────────────────────────────────
# 1. GENERATE SYNTHETIC TELCO CHURN DATASET
# ─────────────────────────────────────────────────────────────
np.random.seed(42)
N = 2000

def generate_telco_data(n=N):
    tenure        = np.random.randint(1, 73, n)
    monthly       = np.round(np.random.uniform(20, 120, n), 2)
    total         = np.round(monthly * tenure * np.random.uniform(0.8, 1.0, n), 2)

    data = pd.DataFrame({
        "customerID":         [f"CUST-{i:04d}" for i in range(n)],
        "gender":             np.random.choice(["Male", "Female"], n),
        "SeniorCitizen":      np.random.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner":            np.random.choice(["Yes", "No"], n),
        "Dependents":         np.random.choice(["Yes", "No"], n),
        "tenure":             tenure,
        "PhoneService":       np.random.choice(["Yes", "No"], n, p=[0.9, 0.1]),
        "MultipleLines":      np.random.choice(["Yes", "No", "No phone service"], n),
        "InternetService":    np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]),
        "OnlineSecurity":     np.random.choice(["Yes", "No", "No internet service"], n),
        "TechSupport":        np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV":        np.random.choice(["Yes", "No", "No internet service"], n),
        "Contract":           np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.24, 0.21]),
        "PaperlessBilling":   np.random.choice(["Yes", "No"], n),
        "PaymentMethod":      np.random.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n
        ),
        "MonthlyCharges":     monthly,
        "TotalCharges":       total,
    })

    # Churn probability influenced by features
    churn_prob = (
        0.05
        + 0.25 * (data["Contract"] == "Month-to-month")
        - 0.10 * (data["tenure"] > 36)
        + 0.15 * (data["InternetService"] == "Fiber optic")
        + 0.08 * (data["SeniorCitizen"] == 1)
        - 0.05 * (data["Partner"] == "Yes")
    ).clip(0.02, 0.90)

    data["Churn"] = np.where(np.random.rand(n) < churn_prob, "Yes", "No")

    # Inject ~2 % missing values in TotalCharges
    mask = np.random.rand(n) < 0.02
    data.loc[mask, "TotalCharges"] = np.nan

    return data

df = generate_telco_data()
print("Dataset shape:", df.shape)
print("\nChurn distribution:\n", df["Churn"].value_counts())
print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])

# ─────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING & SPLIT
# ─────────────────────────────────────────────────────────────
df = df.drop(columns=["customerID"])

# Encode target
le = LabelEncoder()
y = le.fit_transform(df["Churn"])   # No=0, Yes=1
X = df.drop(columns=["Churn"])

# Identify column types
numeric_cols     = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_cols = [c for c in X.columns if c not in numeric_cols]

print(f"\nNumeric features    ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# ─────────────────────────────────────────────────────────────
# 3. PREPROCESSING SUB-PIPELINES
# ─────────────────────────────────────────────────────────────
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer,     numeric_cols),
    ("cat", categorical_transformer, categorical_cols),
])

# ─────────────────────────────────────────────────────────────
# 4. FULL PIPELINES (Preprocessor + Model)
# ─────────────────────────────────────────────────────────────
lr_pipeline = Pipeline(steps=[
    ("preprocessor",        preprocessor),
    ("classifier",          LogisticRegression(max_iter=1000, random_state=42)),
])

rf_pipeline = Pipeline(steps=[
    ("preprocessor",        preprocessor),
    ("classifier",          RandomForestClassifier(random_state=42, n_jobs=-1)),
])

# ─────────────────────────────────────────────────────────────
# 5. HYPERPARAMETER TUNING WITH GridSearchCV
# ─────────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Logistic Regression grid
lr_param_grid = {
    "classifier__C":       [0.01, 0.1, 1, 10],
    "classifier__penalty": ["l1", "l2"],
    "classifier__solver":  ["liblinear"],
}

print("\n[1/2] Tuning Logistic Regression …")
lr_gs = GridSearchCV(
    lr_pipeline, lr_param_grid,
    cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0
)
lr_gs.fit(X_train, y_train)
print(f"  Best params : {lr_gs.best_params_}")
print(f"  Best CV AUC : {lr_gs.best_score_:.4f}")

# Random Forest grid
rf_param_grid = {
    "classifier__n_estimators":      [100, 200],
    "classifier__max_depth":         [None, 10, 20],
    "classifier__min_samples_split": [2, 5],
}

print("\n[2/2] Tuning Random Forest …")
rf_gs = GridSearchCV(
    rf_pipeline, rf_param_grid,
    cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0
)
rf_gs.fit(X_train, y_train)
print(f"  Best params : {rf_gs.best_params_}")
print(f"  Best CV AUC : {rf_gs.best_score_:.4f}")

# ─────────────────────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────────────────────
def evaluate(name, model, X_t, y_t):
    y_pred  = model.predict(X_t)
    y_proba = model.predict_proba(X_t)[:, 1]
    return {
        "Model":    name,
        "Accuracy": round(accuracy_score(y_t, y_pred),    4),
        "F1":       round(f1_score(y_t, y_pred),          4),
        "ROC-AUC":  round(roc_auc_score(y_t, y_proba),   4),
        "y_pred":   y_pred,
        "y_proba":  y_proba,
    }

lr_res = evaluate("Logistic Regression", lr_gs.best_estimator_, X_test, y_test)
rf_res = evaluate("Random Forest",       rf_gs.best_estimator_, X_test, y_test)

print("\n─── Test-set results ──────────────────────────────")
for res in [lr_res, rf_res]:
    print(f"\n{res['Model']}")
    print(f"  Accuracy : {res['Accuracy']}")
    print(f"  F1 Score : {res['F1']}")
    print(f"  ROC-AUC  : {res['ROC-AUC']}")

print("\nClassification Report – Logistic Regression:")
print(classification_report(y_test, lr_res["y_pred"], target_names=["No Churn", "Churn"]))

print("Classification Report – Random Forest:")
print(classification_report(y_test, rf_res["y_pred"], target_names=["No Churn", "Churn"]))

# ─────────────────────────────────────────────────────────────
# 7. VISUALISATIONS  (single A3-style figure → PNG)
# ─────────────────────────────────────────────────────────────
COLORS = {"primary": "#4F46E5", "secondary": "#10B981", "accent": "#F59E0B",
          "danger": "#EF4444", "bg": "#F8FAFC", "text": "#1E293B"}

fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor(COLORS["bg"])

# ── Panel layout ──────────────────────────────────────────────
ax_roc    = fig.add_subplot(3, 3, 1)   # ROC curves
ax_cm_lr  = fig.add_subplot(3, 3, 2)   # CM – LR
ax_cm_rf  = fig.add_subplot(3, 3, 3)   # CM – RF
ax_scores = fig.add_subplot(3, 3, 4)   # Metric comparison bar
ax_fi     = fig.add_subplot(3, 3, 5)   # Feature importance (RF)
ax_dist   = fig.add_subplot(3, 3, 6)   # Churn distribution
ax_cv     = fig.add_subplot(3, 3, 7)   # GridSearch CV AUC comparison
ax_gs_lr  = fig.add_subplot(3, 3, 8)   # LR grid heatmap
ax_gs_rf  = fig.add_subplot(3, 3, 9)   # RF grid partial

for ax in fig.axes:
    ax.set_facecolor(COLORS["bg"])

# ── (1) ROC Curves ────────────────────────────────────────────
for res, col, lbl in [
    (lr_res, COLORS["primary"],   "Logistic Regression"),
    (rf_res, COLORS["secondary"], "Random Forest"),
]:
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    ax_roc.plot(fpr, tpr, color=col, lw=2,
                label=f"{lbl}  (AUC={res['ROC-AUC']:.3f})")
ax_roc.plot([0,1],[0,1], "k--", lw=1)
ax_roc.set(title="ROC Curves", xlabel="FPR", ylabel="TPR")
ax_roc.legend(fontsize=8)
ax_roc.set_title("ROC Curves", color=COLORS["text"], fontweight="bold")

# ── (2,3) Confusion Matrices ──────────────────────────────────
for ax, res, title in [
    (ax_cm_lr, lr_res, "Confusion Matrix\nLogistic Regression"),
    (ax_cm_rf, rf_res, "Confusion Matrix\nRandom Forest"),
]:
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn","Churn"],
                yticklabels=["No Churn","Churn"],
                ax=ax, cbar=False)
    ax.set(title=title, xlabel="Predicted", ylabel="Actual")
    ax.set_title(title, color=COLORS["text"], fontweight="bold")

# ── (4) Metric Comparison Bar ─────────────────────────────────
metrics = ["Accuracy", "F1", "ROC-AUC"]
x       = np.arange(len(metrics))
w       = 0.3
for i, (res, col) in enumerate([(lr_res, COLORS["primary"]), (rf_res, COLORS["secondary"])]):
    vals = [res[m] for m in metrics]
    bars = ax_scores.bar(x + i*w, vals, w, color=col,
                         label=res["Model"], alpha=0.85)
    for bar, v in zip(bars, vals):
        ax_scores.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                       f"{v:.3f}", ha="center", va="bottom", fontsize=7)
ax_scores.set(xticks=x+w/2, xticklabels=metrics, ylim=(0.5, 1.0),
              title="Model Metric Comparison")
ax_scores.set_title("Model Metric Comparison", color=COLORS["text"], fontweight="bold")
ax_scores.legend(fontsize=8)

# ── (5) Feature Importance (RF) ───────────────────────────────
rf_best     = rf_gs.best_estimator_
ohe_features = (rf_best.named_steps["preprocessor"]
                       .named_transformers_["cat"]
                       .named_steps["onehot"]
                       .get_feature_names_out(categorical_cols))
all_features = np.concatenate([numeric_cols, ohe_features])
importances  = rf_best.named_steps["classifier"].feature_importances_
top_idx      = np.argsort(importances)[-12:][::-1]
ax_fi.barh(range(12), importances[top_idx], color=COLORS["secondary"], alpha=0.85)
ax_fi.set(yticks=range(12),
          yticklabels=[all_features[i][:28] for i in top_idx])
ax_fi.invert_yaxis()
ax_fi.set_title("Top-12 RF Feature Importances", color=COLORS["text"], fontweight="bold")

# ── (6) Churn Distribution ────────────────────────────────────
counts = df["Churn"].value_counts()
bars   = ax_dist.bar(counts.index, counts.values,
                     color=[COLORS["secondary"], COLORS["danger"]], alpha=0.85)
for bar, v in zip(bars, counts.values):
    ax_dist.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                 f"{v}\n({v/len(df)*100:.1f}%)", ha="center", va="bottom", fontsize=9)
ax_dist.set(title="Churn Distribution", xlabel="Churn", ylabel="Count")
ax_dist.set_title("Churn Distribution", color=COLORS["text"], fontweight="bold")

# ── (7) CV AUC distribution across folds ─────────────────────
cv_lr = cross_val_score(lr_gs.best_estimator_, X_train, y_train, cv=cv, scoring="roc_auc")
cv_rf = cross_val_score(rf_gs.best_estimator_, X_train, y_train, cv=cv, scoring="roc_auc")
ax_cv.boxplot([cv_lr, cv_rf], tick_labels=["LR", "RF"],
              patch_artist=True,
              boxprops=dict(facecolor=COLORS["accent"], alpha=0.6))
ax_cv.set(title="5-Fold CV AUC", ylabel="ROC-AUC")
ax_cv.set_title("5-Fold CV AUC Distribution", color=COLORS["text"], fontweight="bold")

# ── (8) LR GridSearch heatmap (C vs penalty) ─────────────────
lr_cv_df = pd.DataFrame(lr_gs.cv_results_)
pivot = lr_cv_df.pivot_table(
    values="mean_test_score",
    index="param_classifier__penalty",
    columns="param_classifier__C"
)
sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGnBu", ax=ax_gs_lr, cbar=False)
ax_gs_lr.set_title("LR Grid: penalty × C (AUC)", color=COLORS["text"], fontweight="bold")

# ── (9) RF GridSearch – n_estimators vs max_depth ────────────
rf_cv_df = pd.DataFrame(rf_gs.cv_results_)
rf_cv_df["param_classifier__max_depth"] = (
    rf_cv_df["param_classifier__max_depth"].astype(str)
)
pivot_rf = rf_cv_df.pivot_table(
    values="mean_test_score",
    index="param_classifier__max_depth",
    columns="param_classifier__n_estimators"
)
sns.heatmap(pivot_rf, annot=True, fmt=".4f", cmap="BuGn", ax=ax_gs_rf, cbar=False)
ax_gs_rf.set_title("RF Grid: max_depth × n_estimators (AUC)", color=COLORS["text"], fontweight="bold")

# ── Global title ──────────────────────────────────────────────
fig.suptitle("Customer Churn Prediction – End-to-End ML Pipeline",
             fontsize=16, fontweight="bold", color=COLORS["text"], y=1.01)

plt.tight_layout()
out_png = "churn_pipeline_results.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
print(f"\nVisualisation saved → {out_png}")

# ─────────────────────────────────────────────────────────────
# 8. EXPORT PIPELINES WITH JOBLIB
# ─────────────────────────────────────────────────────────────
joblib.dump(lr_gs.best_estimator_, "lr_churn_pipeline.joblib")
joblib.dump(rf_gs.best_estimator_, "rf_churn_pipeline.joblib")
print("Pipelines exported:\n  lr_churn_pipeline.joblib\n  rf_churn_pipeline.joblib")

# ─────────────────────────────────────────────────────────────
# 9. INFERENCE DEMO (reload & predict)
# ─────────────────────────────────────────────────────────────
loaded_pipeline = joblib.load("rf_churn_pipeline.joblib")
sample          = X_test.iloc[:3].copy()
predictions     = loaded_pipeline.predict(sample)
probabilities   = loaded_pipeline.predict_proba(sample)[:, 1]

print("\n─── Inference demo (first 3 test rows) ────────────────")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    label = le.inverse_transform([pred])[0]
    print(f"  Row {i+1}: {label:3s}  (churn probability = {prob:.3f})")

print("\n✅ Pipeline complete.")
