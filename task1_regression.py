# ============================================================
# Level 2 - Task 1: Regression Analysis (Improved Visualization)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ── 1. Load Dataset ──────────────────────────────────────────
col_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM',
             'AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

df = pd.read_csv(
    r'D:\Codveda_Technologies\Level1\4__house_Prediction_Data_Set.csv',
    header=None,
    sep=r'\s+',
    names=col_names
)

# ── 2. Features ──────────────────────────────────────────────
X = df[['RM', 'LSTAT', 'PTRATIO']]
y = df['MEDV']

# ── 3. Train-Test Split ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 4. Model ─────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ── 5. Metrics ───────────────────────────────────────────────
r2   = r2_score(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# ── 6. Visualization ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    'Level 2 – Task 1: Regression Analysis\nHouse Price Prediction',
    fontsize=14,
    fontweight='bold'
)

# 🎯 Plot 1: Actual vs Predicted
axes[0].scatter(y_test, y_pred, color='dodgerblue', alpha=0.7, edgecolors='white', s=60)
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'red', linestyle='--', linewidth=2)

axes[0].set_title('Actual vs Predicted')
axes[0].set_xlabel('Actual Price')
axes[0].set_ylabel('Predicted Price')
axes[0].grid(alpha=0.3)

# 🎯 Plot 2: Residual Plot
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, color='orange', alpha=0.7, edgecolors='white', s=60)
axes[1].axhline(0, color='black', linestyle='--', linewidth=1.5)

axes[1].set_title('Residual Plot')
axes[1].set_xlabel('Predicted Price')
axes[1].set_ylabel('Residuals')
axes[1].grid(alpha=0.3)

# 🎯 Plot 3: RM vs MEDV
axes[2].scatter(df['RM'], df['MEDV'], color='green', alpha=0.5, s=40)

simple_model = LinearRegression().fit(df[['RM']], df['MEDV'])
rm_range = np.linspace(df['RM'].min(), df['RM'].max(), 100).reshape(-1, 1)

axes[2].plot(rm_range, simple_model.predict(rm_range),
             color='red', linewidth=2)

axes[2].set_title('RM vs House Price')
axes[2].set_xlabel('Average Rooms (RM)')
axes[2].set_ylabel('House Price')
axes[2].grid(alpha=0.3)

# ── 7. Add Metrics INSIDE IMAGE ──────────────────────────────
textstr = (
    f'R² = {r2:.3f}\n'
    f'MSE = {mse:.3f}\n'
    f'RMSE = {rmse:.3f}\n\n'
    f'RM coef = {model.coef_[0]:.2f}\n'
    f'LSTAT coef = {model.coef_[1]:.2f}\n'
    f'PTRATIO coef = {model.coef_[2]:.2f}'
)

# Add text box to figure
fig.text(
    0.75, 0.15, textstr,
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
)

plt.tight_layout()

# ── 8. Save Image ────────────────────────────────────────────
plt.savefig(
    r'D:\Codveda_Technologies\Level2\task1_regression.png',
    dpi=150,
    bbox_inches='tight'
)

print("✅ Plot saved successfully!")
plt.show()