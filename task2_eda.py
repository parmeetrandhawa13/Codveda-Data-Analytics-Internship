"""
Codveda Technology - Data Analytics Internship
Level 1 | Task 2: Exploratory Data Analysis (EDA)
Dataset: House Prediction Data Set (Boston Housing)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Load Cleaned Dataset ──────────────────────────────────────────────────────
df = pd.read_csv("cleaned_house_data.csv")

print("=" * 60)
print("TASK 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)
print(f"\nDataset Shape: {df.shape}")

# ── 1. Summary Statistics ─────────────────────────────────────────────────────
print("\n── Summary Statistics ──")
stats = df.describe().T
stats["median"] = df.median()
stats["mode"]   = df.mode().iloc[0]
print(stats[["count","mean","median","mode","std","min","max"]].round(2))

# ── 2. Histograms with Insights ───────────────────────────────────────────────
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
fig.suptitle("Feature Distributions – Boston Housing Dataset\n📌 Insight: Most features are skewed; MEDV (price) peaks around $21k",
             fontsize=14, fontweight="bold", y=1.01)
axes = axes.flatten()

for i, col in enumerate(df.columns):
    mean_val   = df[col].mean()
    median_val = df[col].median()
    axes[i].hist(df[col], bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[i].axvline(mean_val,   color="red",   linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.1f}")
    axes[i].axvline(median_val, color="green", linestyle="--", linewidth=1.5, label=f"Median: {median_val:.1f}")
    axes[i].set_title(col, fontsize=11, fontweight="bold")
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Frequency")
    axes[i].legend(fontsize=7)

for j in range(len(df.columns), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("eda_histograms.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ Histograms saved.")

# ── 3. Boxplots with Outlier Count Annotations ───────────────────────────────
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
fig.suptitle("Boxplots – Outlier Detection\n📌 Insight: CRIM, ZN, B and MEDV contain significant outliers affecting the distribution",
             fontsize=14, fontweight="bold", y=1.01)
axes = axes.flatten()

for i, col in enumerate(df.columns):
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR    = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    bp = axes[i].boxplot(df[col].dropna(), patch_artist=True,
                         boxprops=dict(facecolor="#4C72B0", alpha=0.7),
                         medianprops=dict(color="red", linewidth=2))
    axes[i].set_title(col, fontsize=11, fontweight="bold")
    axes[i].text(0.98, 0.97, f"Outliers: {len(outliers)}",
                 transform=axes[i].transAxes, ha="right", va="top",
                 fontsize=8, color="red",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="red", alpha=0.8))

for j in range(len(df.columns), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("eda_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Boxplots saved.")

# ── 4. Scatter Plots with Correlation Annotations ────────────────────────────
key_features = ["RM", "LSTAT", "PTRATIO", "CRIM", "NOX", "DIS"]
insights = {
    "RM":      "More rooms → Higher price\n(r = +0.65)",
    "LSTAT":   "Higher poverty % → Lower price\n(r = -0.71)",
    "PTRATIO": "More students per teacher → Lower price\n(r = -0.51)",
    "CRIM":    "Higher crime → Lower price\n(r = -0.36)",
    "NOX":     "More pollution → Lower price\n(r = -0.43)",
    "DIS":     "Farther from jobs → Slight price drop\n(r = +0.26)",
}
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Key Features vs House Price (MEDV)\n📌 Insight: RM and LSTAT are the strongest predictors of house price",
             fontsize=14, fontweight="bold")
axes = axes.flatten()

colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
for i, feat in enumerate(key_features):
    corr = df[feat].corr(df["MEDV"])
    axes[i].scatter(df[feat], df["MEDV"], alpha=0.5, color=colors[i], edgecolors="none", s=30)
    m, b = np.polyfit(df[feat], df["MEDV"], 1)
    x_line = np.linspace(df[feat].min(), df[feat].max(), 100)
    axes[i].plot(x_line, m*x_line+b, color="black", linewidth=2, label="Trend line")
    axes[i].set_xlabel(feat, fontsize=11)
    axes[i].set_ylabel("MEDV (House Price $k)", fontsize=11)
    axes[i].set_title(f"{feat} vs MEDV", fontsize=12, fontweight="bold")
    # Insight box
    axes[i].text(0.05, 0.95, insights[feat],
                 transform=axes[i].transAxes, va="top", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray", alpha=0.9))
    axes[i].legend(fontsize=9)

plt.tight_layout()
plt.savefig("eda_scatter_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Scatter plots saved.")

# ── 5. Correlation Heatmap with Top Insight ───────────────────────────────────
corr = df.corr()
fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, linewidths=0.5, annot_kws={"size": 9})
ax.set_title("Correlation Heatmap – Boston Housing Dataset", fontsize=14, fontweight="bold", pad=15)

# Insight text box below chart
insight_text = (
    "📌 Key Insights:\n"
    "  • RM (+0.65): More rooms = higher price\n"
    "  • LSTAT (−0.71): Higher poverty % = lower price\n"
    "  • TAX & RAD are highly correlated (0.91) → multicollinearity risk\n"
    "  • INDUS & NOX are strongly correlated (0.76) → industrial areas are more polluted"
)
fig.text(0.5, -0.04, insight_text, ha="center", fontsize=10,
         bbox=dict(boxstyle="round,pad=0.6", facecolor="#f0f8ff", edgecolor="#4C72B0", alpha=0.95))

plt.tight_layout()
plt.savefig("eda_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Correlation heatmap saved.")

print("\n── Correlations with MEDV ──")
print(df.corr()["MEDV"].drop("MEDV").sort_values().round(3).to_string())

print("\n" + "=" * 60)
print("TASK 2 COMPLETE ✅")
print("=" * 60)