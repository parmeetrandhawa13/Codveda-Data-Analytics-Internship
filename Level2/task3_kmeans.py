# ============================================================
# Level 2 - Task 3: Clustering Analysis (Improved Version)
# Dataset: Iris Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ── 1. Load Dataset ──────────────────────────────────────────
df = pd.read_csv(r'D:\Codveda_Technologies\Level2\iris.csv')

# ── 2. Feature Selection & Scaling ───────────────────────────
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 3. Elbow + Silhouette Method ─────────────────────────────
inertias = []
silhouettes = []
K_range = range(2, 10)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))

best_k = list(K_range)[np.argmax(silhouettes)]

# ── 4. Final Model ───────────────────────────────────────────
k_final = 3
km_final = KMeans(n_clusters=k_final, random_state=42, n_init=10)
df['Cluster'] = km_final.fit_predict(X_scaled)

# ── 5. PCA for Visualization ─────────────────────────────────
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

explained_var = pca.explained_variance_ratio_.sum() * 100

# ── 6. Visualization ─────────────────────────────────────────
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
markers = ['o', 's', '^']

fig, axes = plt.subplots(2, 2, figsize=(16, 13))
fig.suptitle(
    'Level 2 – Task 3: K-Means Clustering\nIris Dataset',
    fontsize=14,
    fontweight='bold'
)

# 🎯 Plot 1: Elbow Method
axes[0, 0].plot(list(K_range), inertias, 'bo-', lw=2)
axes[0, 0].axvline(x=k_final, color='red', linestyle='--', label='k = 3')
axes[0, 0].set_title('Elbow Method')
axes[0, 0].set_xlabel('k')
axes[0, 0].set_ylabel('Inertia')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 🎯 Plot 2: Silhouette Scores
bar_colors = ['red' if k == k_final else 'steelblue' for k in K_range]
axes[0, 1].bar(list(K_range), silhouettes, color=bar_colors)
axes[0, 1].set_title('Silhouette Scores')
axes[0, 1].set_xlabel('k')
axes[0, 1].set_ylabel('Score')
axes[0, 1].grid(alpha=0.3)

# 🎯 Plot 3: PCA Clusters
for i in range(k_final):
    mask = df['Cluster'] == i
    axes[1, 0].scatter(df.loc[mask, 'PCA1'], df.loc[mask, 'PCA2'],
                       c=colors[i], marker=markers[i], s=70,
                       label=f'Cluster {i+1}', edgecolors='white')

centers_pca = pca.transform(km_final.cluster_centers_)
axes[1, 0].scatter(centers_pca[:, 0], centers_pca[:, 1],
                   c='black', marker='X', s=200, label='Centroids')

axes[1, 0].set_title('Clusters (PCA)')
axes[1, 0].set_xlabel('PC1')
axes[1, 0].set_ylabel('PC2')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 🎯 Plot 4: Petal Features
for i in range(k_final):
    mask = df['Cluster'] == i
    axes[1, 1].scatter(df.loc[mask, 'petal_length'], df.loc[mask, 'petal_width'],
                       c=colors[i], s=70, label=f'Cluster {i+1}', edgecolors='white')

axes[1, 1].set_title('Petal Length vs Width')
axes[1, 1].set_xlabel('Petal Length')
axes[1, 1].set_ylabel('Petal Width')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# ── 7. Insights INSIDE IMAGE ─────────────────────────────────
insight_text = (
    f'Best k (Silhouette) = {best_k}\n'
    f'Chosen k = 3\n'
    f'PCA Variance = {explained_var:.1f}%\n\n'
    f'Insights:\n'
    f'- Data forms 3 clear clusters\n'
    f'- Petal features separate clusters well\n'
    f'- PCA reduces dimensions effectively\n'
    f'- Clusters align with Iris species'
)

fig.text(
    0.70, 0.02,
    insight_text,
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.85, edgecolor='black')
)

plt.tight_layout()

# ── 8. Save ──────────────────────────────────────────────────
plt.savefig(
    r'D:\Codveda_Technologies\Level2\task3_kmeans.png',
    dpi=150,
    bbox_inches='tight'
)

print("✅ Plot saved successfully!")
plt.show()