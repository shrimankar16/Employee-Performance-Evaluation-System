"""
Employee Performance Model Training Pipeline
- Regression: predict PerformanceScore (0–100)
- Classification: predict PerformanceRating (5 classes)
"""

import pandas as pd
import numpy as np
import joblib, os, json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.impute import SimpleImputer
import warnings; warnings.filterwarnings("ignore")

BASE  = os.path.dirname(__file__)
DATA  = os.path.join(BASE, "data", "employee_performance.csv")
MDIR  = os.path.join(BASE, "models")

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA)
print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} cols")

# ── Feature selection ────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "Age", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
    "MonthlyIncome", "JobLevel", "NumCompaniesWorked", "TrainingHoursLastYear",
    "WorkLifeBalance", "JobSatisfaction", "RelationshipSatisfaction",
    "EnvironmentSatisfaction", "Absences", "TaskCompletionRate",
    "PeerRatingScore", "ManagerRatingScore", "ProjectsCompleted", "GoalAchievementPct"
]
CATEGORICAL_FEATURES = [
    "Gender", "Education", "Department", "OverTime", "BusinessTravel"
]

TARGET_REG  = "PerformanceScore"
TARGET_CLF  = "PerformanceRating"

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y_reg = df[TARGET_REG]
y_clf = df[TARGET_CLF]

# ── Preprocessor ─────────────────────────────────────────────────────────────
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer,         NUMERIC_FEATURES),
    ("cat", categorical_transformer,     CATEGORICAL_FEATURES),
])

# ── Train / Test split ────────────────────────────────────────────────────────
X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# ── REGRESSION model (predict numeric score) ─────────────────────────────────
reg_pipeline = Pipeline([
    ("prep",  preprocessor),
    ("model", GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05,
        max_depth=5, subsample=0.8, random_state=42
    ))
])
reg_pipeline.fit(X_train, yr_train)
yr_pred = reg_pipeline.predict(X_test)

mae  = mean_absolute_error(yr_test, yr_pred)
r2   = r2_score(yr_test, yr_pred)
print(f"\n📊 Regression — MAE: {mae:.2f}  |  R²: {r2:.4f}")

# ── CLASSIFICATION model (predict rating label) ───────────────────────────────
clf_pipeline = Pipeline([
    ("prep",  preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300, max_depth=10,
        min_samples_leaf=5, random_state=42, class_weight="balanced"
    ))
])
clf_pipeline.fit(X_train, yc_train)
yc_pred = clf_pipeline.predict(X_test)

acc = accuracy_score(yc_test, yc_pred)
print(f"\n🎯 Classification — Accuracy: {acc:.4f}")
print(classification_report(yc_test, yc_pred))

# ── Feature importance ────────────────────────────────────────────────────────
ohe_cols = list(clf_pipeline.named_steps["prep"]
                .transformers_[1][1].named_steps["ohe"]
                .get_feature_names_out(CATEGORICAL_FEATURES))
all_features = NUMERIC_FEATURES + ohe_cols

importances = clf_pipeline.named_steps["model"].feature_importances_
feat_imp = pd.Series(importances, index=all_features).sort_values(ascending=False)
top_features = feat_imp.head(10).to_dict()

# ── Save artifacts ─────────────────────────────────────────────────────────────
joblib.dump(reg_pipeline, os.path.join(MDIR, "regression_model.pkl"))
joblib.dump(clf_pipeline, os.path.join(MDIR, "classification_model.pkl"))

# Save column info for app
meta = {
    "numeric_features":      NUMERIC_FEATURES,
    "categorical_features":  CATEGORICAL_FEATURES,
    "target_reg":            TARGET_REG,
    "target_clf":            TARGET_CLF,
    "rating_labels":         sorted(df[TARGET_CLF].unique().tolist()),
    "top_features":          top_features,
    "metrics": {
        "regression": {"MAE": round(mae, 2), "R2": round(r2, 4)},
        "classification": {"Accuracy": round(acc, 4)}
    }
}
with open(os.path.join(MDIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n✅ Models saved to {MDIR}")
print(f"   Top features: {list(top_features.keys())[:5]}")
