# =============================================================
# CODVEDA INTERNSHIP — LEVEL 3, TASK 1
# Predictive Modeling (Classification) — Customer Churn
# Tools: Python, scikit-learn, pandas, matplotlib
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import GridSearchCV

# -------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------
train_df = pd.read_csv('D:\Codveda_Technologies\Level3\Churn Prdiction Data\churn-bigml-80.csv')
test_df  = pd.read_csv('D:\Codveda_Technologies\Level3\Churn Prdiction Data\churn-bigml-20.csv')

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Training samples : {len(train_df)}")
print(f"Testing  samples : {len(test_df)}")
print(f"\nColumns: {list(train_df.columns)}")
print(f"\nClass distribution (train):\n{train_df['Churn'].value_counts()}")

# -------------------------------------------------------------
# 2. PREPROCESSING
# -------------------------------------------------------------
# Combine for consistent encoding
combined = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# Encode binary categorical columns
le = LabelEncoder()
for col in ['International plan', 'Voice mail plan', 'Churn']:
    combined[col] = le.fit_transform(combined[col].astype(str).str.strip())

# Drop non-informative columns
combined.drop(columns=['State', 'Area code'], inplace=True)

# Split back
train_clean = combined.iloc[:len(train_df)]
test_clean  = combined.iloc[len(train_df):]

X_train = train_clean.drop(columns=['Churn'])
y_train = train_clean['Churn']
X_test  = test_clean.drop(columns=['Churn'])
y_test  = test_clean['Churn']

# Feature scaling
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("\n✔  Preprocessing complete.")

# -------------------------------------------------------------
# 3. TRAIN MODELS
# -------------------------------------------------------------
models = {
    'Decision Tree'     : DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest'     : RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    results[name] = {
        'Accuracy' : accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall'   : recall_score(y_test, y_pred, zero_division=0),
        'F1-Score' : f1_score(y_test, y_pred, zero_division=0),
        'model'    : model,
        'y_pred'   : y_pred,
    }
    print(f"\n{name}")
    print(f"  Accuracy : {results[name]['Accuracy']:.4f}")
    print(f"  Precision: {results[name]['Precision']:.4f}")
    print(f"  Recall   : {results[name]['Recall']:.4f}")
    print(f"  F1-Score : {results[name]['F1-Score']:.4f}")

# -------------------------------------------------------------
# 4. HYPERPARAMETER TUNING — Random Forest (best model)
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING — Random Forest (Grid Search)")
print("=" * 60)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth'   : [None, 5, 10],
    'min_samples_split': [2, 5],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=0
)
grid_search.fit(X_train_sc, y_train)
best_rf   = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test_sc)

print(f"Best params : {grid_search.best_params_}")
print(f"Tuned F1    : {f1_score(y_test, y_pred_best):.4f}")
print(f"Tuned Acc   : {accuracy_score(y_test, y_pred_best):.4f}")

# -------------------------------------------------------------
# 5. VISUALIZATIONS
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Level 3 — Task 1: Churn Classification Results', fontsize=15, fontweight='bold')

# (a) Metrics comparison bar chart
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
model_names = list(results.keys())
x = np.arange(len(metrics))
width = 0.25

ax = axes[0, 0]
for i, name in enumerate(model_names):
    vals = [results[name][m] for m in metrics]
    ax.bar(x + i * width, vals, width, label=name)
ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.1)
ax.set_title('Model Metrics Comparison')
ax.set_ylabel('Score')
ax.legend(fontsize=8)

# (b) Confusion matrix — best tuned Random Forest
ax = axes[0, 1]
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(cm, display_labels=['No Churn', 'Churn'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Confusion Matrix — Tuned Random Forest')

# (c) Feature importance
ax = axes[1, 0]
importances = best_rf.feature_importances_
feat_names  = X_train.columns
indices     = np.argsort(importances)[::-1][:10]
ax.barh(range(10), importances[indices][::-1], color='steelblue')
ax.set_yticks(range(10))
ax.set_yticklabels(feat_names[indices][::-1], fontsize=8)
ax.set_title('Top 10 Feature Importances')
ax.set_xlabel('Importance')

# (d) F1 score comparison including tuned model
ax = axes[1, 1]
all_names  = model_names + ['RF Tuned']
all_f1     = [results[n]['F1-Score'] for n in model_names] + [f1_score(y_test, y_pred_best)]
colors     = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
bars = ax.bar(all_names, all_f1, color=colors)
ax.set_ylim(0, 1)
ax.set_title('F1-Score Comparison')
ax.set_ylabel('F1-Score')
for bar, val in zip(bars, all_f1):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('task1_classification_results.png', dpi=150, bbox_inches='tight')
print("\n✔  Plot saved: task1_classification_results.png")
plt.close()

print("\n✅  Task 1 Complete!")