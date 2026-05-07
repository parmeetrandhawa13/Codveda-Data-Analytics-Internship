# =============================================================
# CODVEDA INTERNSHIP — LEVEL 3, TASK 2
# Building Dashboards — Churn Data Dashboard (matplotlib)
# Note: This script produces a multi-panel dashboard that
#       replicates the kind of visuals you'd build in
#       Power BI / Tableau. Import this CSV into Power BI
#       or Tableau for the interactive version.
# Tools: Python, pandas, matplotlib, seaborn
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------
# 1. LOAD & PREPARE DATA
# -------------------------------------------------------------
train_df = pd.read_csv('D:\Codveda_Technologies\Level3\Churn Prdiction Data\churn-bigml-80.csv')
test_df  = pd.read_csv('D:\Codveda_Technologies\Level3\Churn Prdiction Data\churn-bigml-20.csv')
df = pd.concat([train_df, test_df], ignore_index=True)

# Standardize Churn to boolean
df['Churn'] = df['Churn'].astype(str).str.strip().map(
    {'True': True, 'False': False, 'true': True, 'false': False}
)

print("=" * 60)
print("DASHBOARD DATA OVERVIEW")
print("=" * 60)
print(f"Total records : {len(df)}")
print(f"Churn rate    : {df['Churn'].mean()*100:.1f}%")
print(f"States        : {df['State'].nunique()}")

# -------------------------------------------------------------
# 2. DASHBOARD LAYOUT
# -------------------------------------------------------------
fig = plt.figure(figsize=(20, 14), facecolor='#1e1e2e')
fig.suptitle('📊  Customer Churn Analytics Dashboard',
             fontsize=20, fontweight='bold', color='white', y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

DARK_BG   = '#2a2a3e'
TEXT_COL  = 'white'
ACCENT    = '#7c83fd'
GREEN     = '#56cfad'
RED_COL   = '#ff6b6b'
YELLOW    = '#ffd166'

def style_ax(ax, title):
    ax.set_facecolor(DARK_BG)
    ax.set_title(title, color=TEXT_COL, fontsize=11, fontweight='bold', pad=8)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')

# ------ KPI CARDS (row 0) ------
kpi_data = [
    ('Total Customers', f"{len(df):,}", ACCENT),
    ('Churned', f"{df['Churn'].sum():,}", RED_COL),
    ('Churn Rate', f"{df['Churn'].mean()*100:.1f}%", YELLOW),
]
for i, (label, value, color) in enumerate(kpi_data):
    ax_kpi = fig.add_subplot(gs[0, i])
    ax_kpi.set_facecolor(DARK_BG)
    ax_kpi.set_xlim(0, 1); ax_kpi.set_ylim(0, 1)
    ax_kpi.axis('off')
    ax_kpi.add_patch(plt.Rectangle((0.05, 0.1), 0.9, 0.8,
                                    color=color, alpha=0.15, zorder=0))
    ax_kpi.text(0.5, 0.62, value, ha='center', va='center',
                fontsize=28, fontweight='bold', color=color)
    ax_kpi.text(0.5, 0.28, label, ha='center', va='center',
                fontsize=12, color=TEXT_COL)
    for spine in ax_kpi.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2)

# ------ CHURN BY INTERNATIONAL PLAN (row 1, col 0) ------
ax1 = fig.add_subplot(gs[1, 0])
intl_churn = df.groupby('International plan')['Churn'].mean() * 100
bars = ax1.bar(intl_churn.index, intl_churn.values,
               color=[GREEN, RED_COL], width=0.5)
style_ax(ax1, 'Churn Rate by International Plan')
ax1.set_ylabel('Churn Rate (%)', color=TEXT_COL, fontsize=8)
for bar, val in zip(bars, intl_churn.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', color=TEXT_COL, fontsize=9)

# ------ CUSTOMER SERVICE CALLS DISTRIBUTION (row 1, col 1) ------
ax2 = fig.add_subplot(gs[1, 1])
churned     = df[df['Churn'] == True]['Customer service calls']
not_churned = df[df['Churn'] == False]['Customer service calls']
ax2.hist(not_churned, bins=10, alpha=0.7, color=GREEN,  label='No Churn')
ax2.hist(churned,     bins=10, alpha=0.7, color=RED_COL, label='Churn')
style_ax(ax2, 'Customer Service Calls Distribution')
ax2.set_xlabel('# Service Calls', color=TEXT_COL, fontsize=8)
ax2.set_ylabel('Count', color=TEXT_COL, fontsize=8)
ax2.legend(facecolor=DARK_BG, labelcolor=TEXT_COL, fontsize=8)

# ------ TOP 10 STATES BY CHURN RATE (row 1, col 2) ------
ax3 = fig.add_subplot(gs[1, 2])
state_churn = df.groupby('State')['Churn'].mean().sort_values(ascending=False).head(10) * 100
ax3.barh(state_churn.index[::-1], state_churn.values[::-1], color=ACCENT)
style_ax(ax3, 'Top 10 States by Churn Rate')
ax3.set_xlabel('Churn Rate (%)', color=TEXT_COL, fontsize=8)

# ------ TOTAL DAY MINUTES vs CHARGE SCATTER (row 2, col 0) ------
ax4 = fig.add_subplot(gs[2, 0])
colors_scatter = df['Churn'].map({True: RED_COL, False: GREEN})
ax4.scatter(df['Total day minutes'], df['Total day charge'],
            c=colors_scatter, alpha=0.4, s=10)
style_ax(ax4, 'Day Minutes vs Day Charge')
ax4.set_xlabel('Total Day Minutes', color=TEXT_COL, fontsize=8)
ax4.set_ylabel('Total Day Charge ($)', color=TEXT_COL, fontsize=8)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=GREEN, label='No Churn'),
                   Patch(facecolor=RED_COL, label='Churn')]
ax4.legend(handles=legend_elements, facecolor=DARK_BG,
           labelcolor=TEXT_COL, fontsize=8)

# ------ CHURN BY VOICEMAIL PLAN (row 2, col 1) ------
ax5 = fig.add_subplot(gs[2, 1])
vm_churn = df.groupby('Voice mail plan')['Churn'].mean() * 100
ax5.pie(vm_churn.values,
        labels=[f'{l}\n({v:.1f}% churn)' for l, v in zip(vm_churn.index, vm_churn.values)],
        colors=[GREEN, RED_COL], autopct='%1.1f%%',
        textprops={'color': TEXT_COL, 'fontsize': 9},
        wedgeprops={'edgecolor': DARK_BG})
ax5.set_facecolor(DARK_BG)
ax5.set_title('Churn by Voicemail Plan', color=TEXT_COL,
              fontsize=11, fontweight='bold', pad=8)

# ------ ACCOUNT LENGTH BOXPLOT (row 2, col 2) ------
ax6 = fig.add_subplot(gs[2, 2])
churn_groups = [df[df['Churn'] == False]['Account length'],
                df[df['Churn'] == True]['Account length']]
bp = ax6.boxplot(churn_groups, labels=['No Churn', 'Churn'],
                 patch_artist=True,
                 boxprops=dict(facecolor=DARK_BG, color=ACCENT),
                 medianprops=dict(color=YELLOW, linewidth=2),
                 whiskerprops=dict(color=TEXT_COL),
                 capprops=dict(color=TEXT_COL),
                 flierprops=dict(markerfacecolor=RED_COL, markersize=4))
bp['boxes'][0].set_facecolor(GREEN)
bp['boxes'][1].set_facecolor(RED_COL)
style_ax(ax6, 'Account Length by Churn Status')
ax6.set_ylabel('Account Length (months)', color=TEXT_COL, fontsize=8)

plt.savefig('task2_dashboard.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("✔  Dashboard saved: task2_dashboard.png")
print("\n💡  Power BI / Tableau Instructions:")
print("    1. Open Power BI Desktop → Get Data → CSV")
print("    2. Load churn-bigml-80.csv and churn-bigml-20.csv")
print("    3. Append the two tables in Power Query")
print("    4. Create visuals: Bar (Churn by State), Pie (Int'l Plan),")
print("       Line (Account Length), Scatter (Day Minutes vs Charge)")
print("    5. Add slicers: State, International Plan, Voice Mail Plan")
print("    6. Publish to Power BI Service and share the link")
print("\n✅  Task 2 Complete!")