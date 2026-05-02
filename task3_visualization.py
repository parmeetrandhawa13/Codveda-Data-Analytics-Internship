"""
Codveda Technology - Data Analytics Internship
Level 1 | Task 3: Basic Data Visualization
Dataset: House Prediction Data Set (Boston Housing)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

df = pd.read_csv("cleaned_house_data.csv")

print("=" * 60)
print("TASK 3: BASIC DATA VISUALIZATION")
print("=" * 60)

sns.set_theme(style="whitegrid", palette="muted")

# ── 1. Bar Plot – Average Price by CHAS with Insight ─────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
avg_price = df.groupby("CHAS")["MEDV"].mean().reset_index()
avg_price["Label"] = avg_price["CHAS"].map({0: "Not on River", 1: "On River"})
pct_diff = ((avg_price.loc[1,"MEDV"] - avg_price.loc[0,"MEDV"]) / avg_price.loc[0,"MEDV"]) * 100

bars = ax.bar(avg_price["Label"], avg_price["MEDV"],
              color=["#4C72B0","#DD8452"], edgecolor="white", width=0.5)
for bar, val in zip(bars, avg_price["MEDV"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"${val:.1f}k", ha="center", fontsize=13, fontweight="bold")

# Insight annotation
ax.annotate(f"River properties are\n{pct_diff:.1f}% more expensive",
            xy=(1, avg_price.loc[1,"MEDV"]),
            xytext=(0.6, avg_price.loc[1,"MEDV"] + 3),
            fontsize=10, color="#C44E52", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.5))

insight = "Insight: Properties located along\nthe Charles River command a price\npremium, suggesting waterfront\nlocation adds significant value."
ax.text(0.02, 0.97, insight, transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="gray", alpha=0.9))

ax.set_title("Average House Price: River vs Non-River Properties", fontsize=14, fontweight="bold")
ax.set_xlabel("Property Location", fontsize=12)
ax.set_ylabel("Average House Price (MEDV $k)", fontsize=12)
ax.set_ylim(0, avg_price["MEDV"].max() * 1.4)
plt.tight_layout()
plt.savefig("viz_bar_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("Bar plot saved.")

# ── 2. Line Chart – Price Trend sorted by RM with Insight ────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
df_sorted = df.sort_values("RM").reset_index(drop=True)
ax.plot(df_sorted.index, df_sorted["MEDV"], color="#4C72B0",
        linewidth=1.2, alpha=0.6, label="House Price")
window = 20
df_sorted["MA"] = df_sorted["MEDV"].rolling(window).mean()
ax.plot(df_sorted.index, df_sorted["MA"], color="#C44E52",
        linewidth=2.5, label=f"{window}-Point Moving Avg")

# Annotate key zones
ax.axvspan(0, 150, alpha=0.07, color="red",   label="Few rooms zone")
ax.axvspan(350, 506, alpha=0.07, color="green", label="Many rooms zone")
ax.text(60,  8,  "Low price zone\n(Few rooms)", fontsize=9,  color="red",   fontweight="bold")
ax.text(370, 42, "High price zone\n(Many rooms)", fontsize=9, color="green", fontweight="bold")

insight = "Insight: As the number of rooms\nincreases, house prices rise sharply.\nThe moving average confirms a clear\nupward trend from left to right."
ax.text(0.01, 0.97, insight, transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="gray", alpha=0.9))

ax.set_title("House Price Trend Ordered by Number of Rooms (RM)", fontsize=14, fontweight="bold")
ax.set_xlabel("Index (sorted by Rooms)", fontsize=12)
ax.set_ylabel("House Price (MEDV $k)", fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("viz_line_chart.png", dpi=150, bbox_inches="tight")
plt.close()
print("Line chart saved.")

# ── 3. Scatter Plot – RM vs MEDV with Insight ────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(df["RM"], df["MEDV"], c=df["LSTAT"], cmap="RdYlGn_r",
                alpha=0.75, edgecolors="none", s=60)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("LSTAT (% Lower Status Population)", fontsize=10)

m, b = np.polyfit(df["RM"], df["MEDV"], 1)
x_line = np.linspace(df["RM"].min(), df["RM"].max(), 100)
ax.plot(x_line, m*x_line+b, color="black", linewidth=2.5,
        label=f"Trend: Price = {m:.1f}×Rooms + {b:.1f}")

# Annotate best value cluster
ax.annotate("Best value:\nMany rooms,\nlow poverty",
            xy=(7.2, 45), xytext=(7.5, 38),
            fontsize=9, color="darkgreen", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="darkgreen"))

# Annotate worst cluster
ax.annotate("Worst value:\nFew rooms,\nhigh poverty",
            xy=(4.5, 10), xytext=(3.8, 22),
            fontsize=9, color="red", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="red"))

insight = ("Insight: Houses with more rooms (RM > 6)\n"
           "and lower poverty (green dots) command\n"
           "the highest prices. Poverty level (LSTAT)\n"
           "clearly dampens the effect of room count.")
ax.text(0.02, 0.97, insight, transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="gray", alpha=0.9))

ax.set_title("Rooms (RM) vs House Price (MEDV)\nColored by Poverty Level (LSTAT)", fontsize=13, fontweight="bold")
ax.set_xlabel("Average Number of Rooms (RM)", fontsize=12)
ax.set_ylabel("Median House Price (MEDV $k)", fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("viz_scatter_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("Scatter plot saved.")

# ── 4. Combined Dashboard with Insights ──────────────────────────────────────
fig = plt.figure(figsize=(20, 15))
fig.suptitle("Boston Housing Dataset – Visual Insights Dashboard\nCodveda Technology | Data Analytics Internship | Level 1",
             fontsize=16, fontweight="bold", y=1.01)

# --- Price Distribution ---
ax1 = fig.add_subplot(2, 3, 1)
ax1.hist(df["MEDV"], bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
ax1.axvline(df["MEDV"].mean(),   color="red",   linestyle="--", linewidth=2, label=f'Mean: ${df["MEDV"].mean():.1f}k')
ax1.axvline(df["MEDV"].median(), color="green", linestyle="--", linewidth=2, label=f'Median: ${df["MEDV"].median():.1f}k')
ax1.text(35, ax1.get_ylim()[1]*0.6 if ax1.get_ylim()[1] > 0 else 20,
         "Most homes\npriced $15k–$25k", fontsize=8, color="#333",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
ax1.set_title("Price Distribution", fontweight="bold")
ax1.set_xlabel("MEDV ($k)")
ax1.set_ylabel("Frequency")
ax1.legend(fontsize=8)

# --- Bar: Avg Price by RAD ---
ax2 = fig.add_subplot(2, 3, 2)
avg_rad = df.groupby("RAD")["MEDV"].mean().sort_values()
colors_rad = ["#C44E52" if v < avg_rad.mean() else "#55A868" for v in avg_rad.values]
ax2.barh(avg_rad.index.astype(str), avg_rad.values, color=colors_rad)
ax2.axvline(df["MEDV"].mean(), color="black", linestyle="--", linewidth=1.5, label="Overall avg")
ax2.text(23, 1, "Below avg\nAbove avg", fontsize=8,
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
ax2.set_title("Avg Price by Highway Access (RAD)", fontweight="bold")
ax2.set_xlabel("Avg MEDV ($k)")
ax2.set_ylabel("RAD Index")
ax2.legend(fontsize=8)

# --- Scatter RM vs MEDV ---
ax3 = fig.add_subplot(2, 3, 3)
ax3.scatter(df["RM"], df["MEDV"], alpha=0.5, color="#55A868", s=20)
m, b = np.polyfit(df["RM"], df["MEDV"], 1)
x_line = np.linspace(df["RM"].min(), df["RM"].max(), 100)
ax3.plot(x_line, m*x_line+b, color="black", linewidth=2)
ax3.text(0.05, 0.9, f"r = +0.65\nStrongest +ve predictor",
         transform=ax3.transAxes, fontsize=8,
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
ax3.set_title("Rooms vs Price (r=+0.65)", fontweight="bold")
ax3.set_xlabel("RM (Rooms)")
ax3.set_ylabel("MEDV ($k)")

# --- Scatter LSTAT vs MEDV ---
ax4 = fig.add_subplot(2, 3, 4)
ax4.scatter(df["LSTAT"], df["MEDV"], alpha=0.5, color="#C44E52", s=20)
m, b = np.polyfit(df["LSTAT"], df["MEDV"], 1)
x_line = np.linspace(df["LSTAT"].min(), df["LSTAT"].max(), 100)
ax4.plot(x_line, m*x_line+b, color="black", linewidth=2)
ax4.text(0.05, 0.9, f"r = −0.71\nStrongest −ve predictor",
         transform=ax4.transAxes, fontsize=8,
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
ax4.set_title("Poverty % vs Price (r=−0.71)", fontweight="bold")
ax4.set_xlabel("LSTAT (%)")
ax4.set_ylabel("MEDV ($k)")

# --- Boxplot ---
ax5 = fig.add_subplot(2, 3, 5)
top_features = ["RM", "NOX", "LSTAT", "PTRATIO", "DIS"]
bp = ax5.boxplot([df[c].dropna() for c in top_features],
                 labels=top_features, patch_artist=True,
                 medianprops=dict(color="white", linewidth=2))
colors_box = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2"]
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax5.text(0.02, 0.97, "NOX & LSTAT show\nhighest variability",
         transform=ax5.transAxes, va="top", fontsize=8,
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
ax5.set_title("Key Feature Distributions", fontweight="bold")
ax5.set_ylabel("Value")

# --- Correlation Bar ---
ax6 = fig.add_subplot(2, 3, 6)
corr_vals = df.corr()["MEDV"].drop("MEDV").sort_values()
colors_corr = ["#C44E52" if v < 0 else "#55A868" for v in corr_vals]
bars6 = ax6.barh(corr_vals.index, corr_vals.values, color=colors_corr, edgecolor="white")
ax6.axvline(0, color="black", linewidth=1)
# Annotate top 2
ax6.text(corr_vals["RM"]   + 0.01, list(corr_vals.index).index("RM"),    "Best +ve", fontsize=7, va="center", color="green")
ax6.text(corr_vals["LSTAT"]- 0.01, list(corr_vals.index).index("LSTAT"), "Best −ve", fontsize=7, va="center", ha="right", color="red")
ax6.set_title("Feature Correlation with Price", fontweight="bold")
ax6.set_xlabel("Pearson r")

plt.tight_layout()
plt.savefig("viz_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("Dashboard saved.")

print("\n" + "=" * 60)
print("TASK 3 COMPLETE ✅")
print("=" * 60)
print("\nAll plots saved in current directory.")