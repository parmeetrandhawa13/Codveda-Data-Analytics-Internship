# ============================================================
# Codveda Internship - Level 3, Task 3
# NLP - Sentiment Analysis Dashboard (Enhanced Version)
# Dataset: Sentiment Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings

from textblob import TextBlob
from collections import Counter
from wordcloud import WordCloud

warnings.filterwarnings('ignore')

# ============================================================
# 1. STOPWORDS
# ============================================================

STOP_WORDS = set([
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','both','each','few','more','most',
    'other','some','such','no','nor','not','only','own','same','so','than','too',
    'very','can','will','just','don','should','now'
])

# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(
    r"D:\Codveda_Technologies\Level3\3) Sentiment dataset.csv"
)

df.columns = df.columns.str.strip()

df['Text'] = df['Text'].astype(str).str.strip()
df['Sentiment'] = df['Sentiment'].astype(str).str.strip()

print("=" * 60)
print("        NLP SENTIMENT ANALYSIS DASHBOARD")
print("=" * 60)

print(f"\nTotal Records : {len(df)}")

# ============================================================
# 3. SENTIMENT MAPPING
# ============================================================

positive_keys = [
    'positive','joy','excitement','contentment','happiness',
    'love','gratitude','hope','pride','amusement','inspiration',
    'relief','enthusiasm','admiration','satisfaction','optimism'
]

negative_keys = [
    'negative','sadness','anger','fear','disgust','anxiety',
    'frustration','disappointment','grief','shame','guilt',
    'hatred','envy','jealousy','loneliness','despair'
]

def map_sentiment(s):
    s = s.lower()

    if any(k in s for k in positive_keys):
        return 'Positive'

    elif any(k in s for k in negative_keys):
        return 'Negative'

    else:
        return 'Neutral'

df['Sentiment_Group'] = df['Sentiment'].apply(map_sentiment)

# ============================================================
# 4. TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove special characters
    text = re.sub(r'[^a-z\s]', '', text)

    tokens = text.split()

    # Remove stopwords
    tokens = [w for w in tokens
              if w not in STOP_WORDS and len(w) > 2]

    return ' '.join(tokens)

df['Cleaned_Text'] = df['Text'].apply(preprocess_text)

# ============================================================
# 5. TEXTBLOB ANALYSIS
# ============================================================

def get_polarity(text):
    return TextBlob(text).sentiment.polarity

def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def classify_sentiment(polarity):

    if polarity > 0.1:
        return 'Positive'

    elif polarity < -0.1:
        return 'Negative'

    else:
        return 'Neutral'

df['Polarity'] = df['Text'].apply(get_polarity)
df['Subjectivity'] = df['Text'].apply(get_subjectivity)

df['Predicted_Sent'] = df['Polarity'].apply(classify_sentiment)

print("\nPredicted Sentiment Distribution:\n")
print(df['Predicted_Sent'].value_counts())

# ============================================================
# 6. DASHBOARD STYLING
# ============================================================

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

SENT_PAL = {
    'Positive': '#66bb6a',
    'Negative': '#ef5350',
    'Neutral': '#ffa726'
}

# ============================================================
# 7. CREATE DASHBOARD
# ============================================================

fig = plt.figure(figsize=(22, 17))

fig.patch.set_facecolor('#f4f6f8')

fig.suptitle(
    "Sentiment Analysis Dashboard",
    fontsize=22,
    fontweight='bold',
    color='#283593',
    y=0.98
)

# ============================================================
# 8. KPI METRICS
# ============================================================

total_records = len(df)
avg_polarity = df['Polarity'].mean()

positive_pct = (
    (df['Predicted_Sent'] == 'Positive').mean() * 100
)

kpi_text = (
    f"Total Records : {total_records}\n"
    f"Average Polarity : {avg_polarity:.2f}\n"
    f"Positive Sentiment : {positive_pct:.1f}%"
)

fig.text(
    0.79,
    0.92,
    kpi_text,
    fontsize=11,
    bbox=dict(
        facecolor='white',
        edgecolor='#5c6bc0',
        alpha=0.95,
        boxstyle='round,pad=0.6'
    )
)

# ============================================================
# 9. PLOT 1 - SENTIMENT DISTRIBUTION
# ============================================================

ax1 = fig.add_subplot(3, 3, 1)

counts = df['Predicted_Sent'].value_counts()

bars = ax1.bar(
    counts.index,
    counts.values,
    color=[SENT_PAL.get(s, '#90a4ae') for s in counts.index],
    edgecolor='white',
    linewidth=2,
    width=0.6
)

for bar, val in zip(bars, counts.values):

    ax1.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 5,
        str(val),
        ha='center',
        fontsize=10,
        fontweight='bold'
    )

ax1.set_title(
    "Predicted Sentiment Distribution",
    fontweight='bold'
)

ax1.set_ylabel("Count")

# ============================================================
# 10. PLOT 2 - PIE CHART
# ============================================================

ax2 = fig.add_subplot(3, 3, 2)

ax2.pie(
    counts,
    labels=counts.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=[SENT_PAL.get(s, '#90a4ae') for s in counts.index],
    wedgeprops={
        'edgecolor': 'white',
        'linewidth': 2
    }
)

ax2.set_title(
    "Sentiment Share",
    fontweight='bold'
)

# ============================================================
# 11. PLOT 3 - POLARITY DISTRIBUTION
# ============================================================

ax3 = fig.add_subplot(3, 3, 3)

ax3.hist(
    df['Polarity'],
    bins=40,
    color='#5c6bc0',
    edgecolor='white',
    linewidth=1.2,
    alpha=0.9
)

ax3.axvline(
    0,
    color='red',
    linestyle='--',
    linewidth=2,
    label='Neutral'
)

ax3.set_title(
    "Polarity Score Distribution",
    fontweight='bold'
)

ax3.set_xlabel("Polarity")
ax3.set_ylabel("Frequency")

ax3.legend()

# ============================================================
# 12. PLOT 4 - PLATFORM POLARITY
# ============================================================

ax4 = fig.add_subplot(3, 3, 4)

if 'Platform' in df.columns:

    platform_pol = (
        df.groupby('Platform')['Polarity']
        .mean()
        .sort_values()
    )

    platform_pol.plot(
        kind='barh',
        ax=ax4,
        color='#26a69a',
        edgecolor='white'
    )

    ax4.set_title(
        "Average Polarity by Platform",
        fontweight='bold'
    )

    ax4.axvline(
        0,
        color='red',
        linestyle='--'
    )

# ============================================================
# 13. PLOT 5 - COUNTRY ANALYSIS
# ============================================================

ax5 = fig.add_subplot(3, 3, 5)

if 'Country' in df.columns:

    top_countries = (
        df['Country']
        .value_counts()
        .head(8)
        .index
    )

    country_df = df[df['Country'].isin(top_countries)]

    country_sent = (
        country_df.groupby(
            ['Country', 'Predicted_Sent']
        ).size().unstack(fill_value=0)
    )

    country_sent.plot(
        kind='bar',
        ax=ax5,
        color=[
            SENT_PAL.get(c, '#90a4ae')
            for c in country_sent.columns
        ],
        edgecolor='white'
    )

    ax5.set_title(
        "Sentiment by Country",
        fontweight='bold'
    )

    ax5.set_ylabel("Count")

    ax5.tick_params(axis='x', rotation=30)

# ============================================================
# 14. PLOT 6 - POLARITY VS SUBJECTIVITY
# ============================================================

ax6 = fig.add_subplot(3, 3, 6)

for sent, grp in df.groupby('Predicted_Sent'):

    ax6.scatter(
        grp['Polarity'],
        grp['Subjectivity'],
        alpha=0.5,
        s=40,
        label=sent,
        color=SENT_PAL.get(sent, '#90a4ae'),
        edgecolors='white',
        linewidth=0.5
    )

ax6.set_title(
    "Polarity vs Subjectivity",
    fontweight='bold'
)

ax6.set_xlabel("Polarity")
ax6.set_ylabel("Subjectivity")

ax6.legend()

# ============================================================
# 15. PLOT 7 - POSITIVE WORD CLOUD
# ============================================================

ax7 = fig.add_subplot(3, 3, 7)

pos_text = ' '.join(
    df[df['Predicted_Sent'] == 'Positive']
    ['Cleaned_Text']
    .dropna()
)

if pos_text.strip():

    wc_pos = WordCloud(
        width=500,
        height=300,
        background_color='#f5f5f5',
        colormap='Greens',
        max_words=80
    ).generate(pos_text)

    ax7.imshow(wc_pos, interpolation='bilinear')

ax7.axis('off')

ax7.set_title(
    "Positive Word Cloud",
    fontweight='bold',
    color='green'
)

# ============================================================
# 16. PLOT 8 - NEGATIVE WORD CLOUD
# ============================================================

ax8 = fig.add_subplot(3, 3, 8)

neg_text = ' '.join(
    df[df['Predicted_Sent'] == 'Negative']
    ['Cleaned_Text']
    .dropna()
)

if neg_text.strip():

    wc_neg = WordCloud(
        width=500,
        height=300,
        background_color='#f5f5f5',
        colormap='Reds',
        max_words=80
    ).generate(neg_text)

    ax8.imshow(wc_neg, interpolation='bilinear')

ax8.axis('off')

ax8.set_title(
    "Negative Word Cloud",
    fontweight='bold',
    color='red'
)

# ============================================================
# 17. PLOT 9 - TOP WORDS
# ============================================================

ax9 = fig.add_subplot(3, 3, 9)

all_words = ' '.join(
    df['Cleaned_Text']
    .dropna()
).split()

top_words = (
    pd.Series(Counter(all_words))
    .nlargest(15)
    .sort_values()
)

top_words.plot(
    kind='barh',
    ax=ax9,
    color='#7e57c2',
    edgecolor='white'
)

ax9.set_title(
    "Top 15 Most Common Words",
    fontweight='bold'
)

ax9.set_xlabel("Frequency")



#============================================================
# 19. SAVE OUTPUT
# ============================================================

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig(
    r"D:\Codveda_Technologies\Level3\task3_sentiment_analysis.png",
    dpi=300,
    bbox_inches='tight',
    facecolor=fig.get_facecolor()
)

print("\n✅ Dashboard saved successfully!")
print("📁 File: task3_sentiment_analysis.png")

plt.show()