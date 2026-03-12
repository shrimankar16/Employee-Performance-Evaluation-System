# 📊 Employee Performance Evaluation System

An end-to-end ML-powered web application to evaluate, predict, and analyze employee performance.

---

## 🗂️ Project Structure

```
employee_perf_system/
├── app.py                    ← Streamlit web app (main entry point)
├── generate_dataset.py       ← Synthetic dataset generator (2000 employees)
├── train_model.py            ← ML model training pipeline
├── requirements.txt          ← Python dependencies
├── data/
│   └── employee_performance.csv   ← Generated dataset
└── models/
    ├── regression_model.pkl       ← GradientBoosting for score prediction
    ├── classification_model.pkl   ← RandomForest for rating prediction
    └── meta.json                  ← Feature lists & model metrics
```

---

## ⚡ Quick Start

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Generate dataset
```bash
python generate_dataset.py
```
This creates `data/employee_performance.csv` with 2000 employee records.

### Step 3 — Train the models
```bash
python train_model.py
```
This trains two models and saves them to the `models/` folder.

### Step 4 — Run the web app
```bash
streamlit run app.py
```
The app opens at `http://localhost:8501`

---

## 🧪 Dataset

The dataset is synthetically generated to mimic real-world HR data (inspired by IBM HR Analytics dataset). It contains **2000 employees** with **27 features**.

### Feature Categories

| Category | Features |
|---------|---------|
| **Demographics** | Age, Gender, Education |
| **Work Profile** | Department, Job Level, Monthly Income |
| **Experience** | Years at Company, Years in Current Role, Years Since Last Promotion |
| **Performance Metrics** | Task Completion Rate, Goal Achievement %, Projects Completed |
| **Ratings** | Manager Rating (1–5), Peer Rating (1–5) |
| **Behavioral** | Overtime, Business Travel, Absences, Training Hours |
| **Satisfaction** | Job Satisfaction, Work-Life Balance, Environment Satisfaction, Relationship Satisfaction |
| **Target (Regression)** | PerformanceScore (0–100, continuous) |
| **Target (Classification)** | PerformanceRating (5 categories) |

### Rating Labels
| Label | Score Range |
|-------|-------------|
| Outstanding | ≥ 85 |
| Exceeds Expectations | 70–84 |
| Meets Expectations | 55–69 |
| Below Expectations | 40–54 |
| Needs Improvement | < 40 |

---

## 🧠 ML Models

### Model 1 — Regression (Score Prediction)
- **Algorithm:** GradientBoostingRegressor
- **Purpose:** Predict a numeric performance score (0–100)
- **Metrics:** R² ≈ 0.79, MAE ≈ 4.5
- **Why GBR?** Handles non-linear relationships, robust to outliers, works well with mixed feature types

### Model 2 — Classification (Rating Prediction)
- **Algorithm:** RandomForestClassifier (class_weight='balanced')
- **Purpose:** Predict rating label from 5 classes
- **Metrics:** Accuracy ≈ 72%
- **Why RF?** Good with imbalanced classes, interpretable feature importance, resistant to overfitting

### Preprocessing Pipeline
```
Raw Input → SimpleImputer → StandardScaler (numeric)
                         → OneHotEncoder (categorical)
         → ColumnTransformer → ML Model
```

---

## 🌐 Web App Pages

| Page | Description |
|------|-------------|
| 🏠 Dashboard | KPIs, charts, rating distribution, department scores |
| 🔮 Predict | Input employee details → get score + rating + recommendations |
| 📁 Batch | Upload CSV → bulk predict all employees |
| 📈 Analytics | Feature importance, correlations, dept deep dives |
| ℹ️ About | Documentation and usage guide |

---

## 🔧 Extending the System

### Use a real dataset instead of synthetic
Replace `generate_dataset.py` with loading from:
- IBM HR Analytics (Kaggle) — `ibm-hr-analytics-attrition-dataset`
- UCI Machine Learning Repository HR datasets

### Add more ML models to compare
In `train_model.py`, add:
```python
from sklearn.xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
```

### Deploy to cloud
```bash
# Streamlit Cloud (free)
# 1. Push to GitHub
# 2. Go to share.streamlit.io
# 3. Connect repo → deploy

# Or use Heroku / Railway / Render
```

---

## 📦 Requirements
- Python 3.9+
- See requirements.txt
