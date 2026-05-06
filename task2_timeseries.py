# ============================================================
# Level 2 - Task 2: Time Series Analysis (Improved Version)
# Dataset: Stock Prices
# Tools: Python, pandas, matplotlib, statsmodels
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# ── 1. Load Dataset ──────────────────────────────────────────
df_raw = pd.read_csv(r'D:\Codveda_Technologies\Level2\2) Stock Prices Data Set.csv')

print("="*60)
print("        TIME SERIES ANALYSIS - STOCK PRICES")
print("="*60)

print(f"\nColumns      : {list(df_raw.columns)}")
print(f"Total Rows   : {df_raw.shape[0]}")
print(f"Unique Stocks: {df_raw['symbol'].nunique()}")

# ── 2. Filter Stock (AAPL) ───────────────────────────────────
symbol = 'AAPL'
df = df_raw[df_raw['symbol'] == symbol].copy()

df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)
df.set_index('date', inplace=True)

print(f"\nSelected Stock : {symbol}")
print(f"Date Range     : {df.index.min().date()} → {df.index.max().date()}")

# ── 3. Moving Averages ───────────────────────────────────────
df['MA_20']  = df['close'].rolling(20).mean()
df['MA_50']  = df['close'].rolling(50).mean()
df['MA_200'] = df['close'].rolling(200).mean()

# ── 4. Monthly Resampling (for decomposition) ────────────────
monthly = df['close'].resample('M').mean()

# ── 5. Decomposition ─────────────────────────────────────────
decomp = seasonal_decompose(monthly, model='additive', period=12)

# ── 6. Visualization ─────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))

fig.suptitle(
    f'Level 2 – Task 2: Time Series Analysis\n{symbol} Stock Price',
    fontsize=14,
    fontweight='bold'
)

# 🎯 Plot 1: Close + Moving Averages
ax1 = fig.add_subplot(3, 2, (1, 2))
ax1.plot(df.index, df['close'],  color='steelblue',  lw=1,   alpha=0.7, label='Close')
ax1.plot(df.index, df['MA_20'],  color='orange',     lw=1.5, label='MA 20')
ax1.plot(df.index, df['MA_50'],  color='green',      lw=1.5, label='MA 50')
ax1.plot(df.index, df['MA_200'], color='red',        lw=2,   label='MA 200')

ax1.set_title(f'{symbol} Close Price with Moving Averages')
ax1.set_ylabel('Price (USD)')
ax1.legend(loc='upper left')
ax1.grid(alpha=0.3)

# 🎯 Plot 2: Trend
ax2 = fig.add_subplot(3, 2, 3)
ax2.plot(decomp.trend, color='darkorange', lw=2)
ax2.set_title('Trend Component')
ax2.set_ylabel('Price')
ax2.grid(alpha=0.3)

# 🎯 Plot 3: Seasonal
ax3 = fig.add_subplot(3, 2, 4)
ax3.plot(decomp.seasonal, color='mediumseagreen', lw=2)
ax3.set_title('Seasonal Component')
ax3.set_ylabel('Effect')
ax3.grid(alpha=0.3)

# 🎯 Plot 4: Residual
ax4 = fig.add_subplot(3, 2, 5)
ax4.plot(decomp.resid, color='salmon', lw=1.5)
ax4.axhline(0, color='black', linestyle='--')
ax4.set_title('Residual Component')
ax4.set_ylabel('Residual')
ax4.grid(alpha=0.3)

# 🎯 Plot 5: Monthly Volume
ax5 = fig.add_subplot(3, 2, 6)
monthly_vol = df['volume'].resample('M').sum() / 1e9

ax5.bar(monthly_vol.index, monthly_vol.values,
        color='cornflowerblue', alpha=0.7, width=20)

ax5.set_title('Monthly Trading Volume')
ax5.set_ylabel('Volume (Billions)')
ax5.grid(alpha=0.3, axis='y')

# ── 7. Add Insights INSIDE IMAGE ─────────────────────────────
insight_text = (
    "Insights:\n"
    "- Trend shows long-term price movement\n"
    "- Seasonal component shows repeating yearly pattern\n"
    "- Moving averages smooth fluctuations\n"
    "- MA crossover indicates trend changes\n"
    "- Residual = random noise/unexplained variation"
)

fig.text(
    0.70, 0.02,
    insight_text,
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.85, edgecolor='black')
)

plt.tight_layout()

# ── 8. Save Output ───────────────────────────────────────────
plt.savefig(
    r'D:\Codveda_Technologies\Level2\task2_timeseries.png',
    dpi=150,
    bbox_inches='tight'
)

print("\n✅ Plot saved successfully!")
plt.show()