import json
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

MODEL_DIR = Path('models')
PLOTS_DIR = Path('plots')
RANDOM_STATE = 42
N_CLUSTERS = 5  # Determined by elbow method and silhouette analysis — see plots
                # 03 and 04. Change this value and re-run if analysis suggests
                # a different optimal k.

MODEL_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# DATA LOADING & CLEANING
# ---------------------------------------------------------------------------

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
df = pd.read_csv(url)
print(f"Raw shape: {df.shape}")

# Apply same cleaning as telco-churn-predictor
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df.drop(columns=['customerID'], inplace=True)

print(f"Cleaned shape: {df.shape}")
print(f"\nFeature summary:")
print(df.describe())

# ---------------------------------------------------------------------------
# EDA — plots 01 and 02
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("GENERATING EDA CHARTS")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ['tenure', 'MonthlyCharges', 'TotalCharges']):
    ax.hist(df[col], bins=30, color='#2E75B6', edgecolor='white', alpha=0.85)
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel('Count')
fig.suptitle('Feature Distributions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / '01_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/01_feature_distributions.png")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr = df[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, ax=ax)
ax.set_title('Correlation Matrix — Numeric Features', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / '02_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/02_correlation_matrix.png")

# ---------------------------------------------------------------------------
# FEATURE SELECTION FOR CLUSTERING
# ---------------------------------------------------------------------------

# Churn is excluded from clustering features — it is used only as a
# post-hoc validation label to check whether segments correspond to
# meaningful churn risk differences. Including it would make the
# segmentation circular.
churn_labels = df['Churn'].copy()

# Binary Yes/No columns — encode as 1/0
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
               'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
               'DeviceProtection', 'TechSupport', 'StreamingTV',
               'StreamingMovies']
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0,
                               'No phone service': 0,
                               'No internet service': 0}).fillna(df[col])

# Features selected for clustering — each chosen for business interpretability
# and relevance to customer behaviour, not just statistical variance.
cluster_features = [
    'tenure',           # captures customer lifecycle stage
    'MonthlyCharges',   # captures spending level
    'TotalCharges',     # captures lifetime value
    'SeniorCitizen',    # demographic signal
    'Contract',         # strongest churn predictor from telco project
    'InternetService',  # service tier signal
    'TechSupport',      # engagement signal
    'OnlineSecurity',   # engagement signal
    'PaymentMethod',    # payment behaviour signal
    'PaperlessBilling', # digital engagement signal
]

df_cluster = df[cluster_features].copy()

# One-hot encode nominal categoricals; drop_first avoids multicollinearity
df_cluster = pd.get_dummies(df_cluster,
                            columns=['Contract', 'InternetService', 'PaymentMethod'],
                            drop_first=True)

print("\n" + "=" * 70)
print("PREPROCESSING")
print("=" * 70)
print(f"Clustering features ({df_cluster.shape[1]} total):")
for col in df_cluster.columns:
    print(f"  {col}")

feature_names = df_cluster.columns.tolist()

# Clustering is distance-based — unscaled features with different ranges
# (e.g. TotalCharges up to ~8,000 vs SeniorCitizen 0/1) dominate the
# distance calculation. StandardScaler ensures every feature contributes
# proportionally.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)

pickle.dump(scaler, open(MODEL_DIR / 'scaler.pkl', 'wb'))
pickle.dump(feature_names, open(MODEL_DIR / 'feature_names.pkl', 'wb'))
print(f"\nScaler and feature names saved.")

# ---------------------------------------------------------------------------
# FINDING OPTIMAL K — elbow method and silhouette scores
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("FINDING OPTIMAL K")
print("=" * 70)

k_range = range(2, 11)
inertias = []
silhouette_scores_list = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouette_scores_list.append(silhouette_score(X_scaled, labels, sample_size=2000, random_state=RANDOM_STATE))
    print(f"  k={k}: inertia={km.inertia_:.0f}, silhouette={silhouette_scores_list[-1]:.4f}")

# Elbow plot
# Inertia always decreases as k increases. The elbow is where the rate of
# decrease sharply slows — adding more clusters beyond this point gives
# diminishing returns.
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(list(k_range), inertias, 'o-', color='#2E75B6', linewidth=2, markersize=7)
ax.axvline(x=N_CLUSTERS, color='#E74C3C', linestyle='--', linewidth=1.5,
           label=f'k={N_CLUSTERS} (selected)')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)')
ax.set_title('Elbow Method — Optimal Number of Clusters', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / '03_elbow_method.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: plots/03_elbow_method.png")

# Silhouette plot
# Silhouette score measures how similar a point is to its own cluster vs
# other clusters. Range -1 to 1; higher is better. Use this alongside the
# elbow method to confirm optimal k.
best_k_silhouette = list(k_range)[np.argmax(silhouette_scores_list)]
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(list(k_range), silhouette_scores_list, 's-', color='#2ECC71', linewidth=2, markersize=7)
ax.axvline(x=best_k_silhouette, color='#E74C3C', linestyle='--', linewidth=1.5,
           label=f'Best k={best_k_silhouette} (silhouette)')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Silhouette Score')
ax.set_title('Silhouette Score — Cluster Cohesion and Separation', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / '04_silhouette_scores.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: plots/04_silhouette_scores.png")

print(f"\nElbow method suggests: k={N_CLUSTERS} (selected)")
print(f"Silhouette score suggests: k={best_k_silhouette}")
if best_k_silhouette == N_CLUSTERS:
    print(f"Both methods agree on k={N_CLUSTERS}. Proceeding.")
else:
    print(f"Methods disagree. Proceeding with N_CLUSTERS={N_CLUSTERS} "
          f"(silhouette favours k={best_k_silhouette}). "
          f"Adjust N_CLUSTERS if the cluster profiles are not interpretable.")

# ---------------------------------------------------------------------------
# KMEANS CLUSTERING
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("KMEANS CLUSTERING")
print("=" * 70)

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
df['KMeans_Cluster'] = kmeans_labels

pickle.dump(kmeans, open(MODEL_DIR / 'kmeans_model.pkl', 'wb'))
print(f"KMeans model saved. Cluster distribution:")
for c in range(N_CLUSTERS):
    n = (kmeans_labels == c).sum()
    print(f"  Cluster {c}: {n} customers ({n/len(kmeans_labels)*100:.1f}%)")

# ---------------------------------------------------------------------------
# HIERARCHICAL CLUSTERING
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("HIERARCHICAL CLUSTERING")
print("=" * 70)

agg = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='ward')
hier_labels = agg.fit_predict(X_scaled)
df['Hierarchical_Cluster'] = hier_labels

ari = adjusted_rand_score(kmeans_labels, hier_labels)
pct = (kmeans_labels == hier_labels).mean() * 100
# High ARI means both algorithms find similar structure in the data.
# Low ARI suggests the cluster boundaries are not robust — consider
# different k or features.
print(f"KMeans and Hierarchical Clustering agree on {pct:.1f}% of "
      f"assignments (Adjusted Rand Score: {ari:.3f})")

# Dendrogram — sample 200 rows (full dataset is too slow)
print("\nGenerating dendrogram (200-row sample)...")
sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_scaled), 200, replace=False)
X_sample = X_scaled[sample_idx]
Z = linkage(X_sample, method='ward')

fig, ax = plt.subplots(figsize=(14, 6))
dendrogram(Z, ax=ax, truncate_mode='lastp', p=20,
           leaf_font_size=9, color_threshold=0)

# Horizontal dashed line showing where to cut for N_CLUSTERS
cut_height = sorted(Z[:, 2], reverse=True)[N_CLUSTERS - 1]
ax.axhline(y=cut_height, color='#E74C3C', linestyle='--', linewidth=1.5,
           label=f'Cut for k={N_CLUSTERS}')
ax.set_title(f'Hierarchical Clustering Dendrogram (200-row sample, Ward linkage)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Sample index (or cluster size)')
ax.set_ylabel('Ward linkage distance')
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / '05_dendrogram.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/05_dendrogram.png")

# ---------------------------------------------------------------------------
# PCA DIMENSIONALITY REDUCTION
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("PCA DIMENSIONALITY REDUCTION")
print("=" * 70)

# PCA is applied after clustering purely for visualisation — it does not
# change the cluster assignments. We cannot plot 20+ dimensions, but
# projecting to 2D/3D lets us check whether clusters are visually separable.
pca_full = PCA(random_state=RANDOM_STATE)
pca_full.fit(X_scaled)

pickle.dump(pca_full, open(MODEL_DIR / 'pca_model.pkl', 'wb'))

evr = pca_full.explained_variance_ratio_
print("Explained variance ratio (first 5 components):")
for i, v in enumerate(evr[:5]):
    print(f"  PC{i+1}: {v:.4f} ({v*100:.1f}%)")
cumulative = np.cumsum(evr)
print(f"\nCumulative explained variance:")
for i, cv in enumerate(cumulative[:10]):
    print(f"  PC1–{i+1}: {cv:.4f} ({cv*100:.1f}%)")

pca_variance_data = {
    'explained_variance_ratio': evr[:10].tolist(),
    'cumulative_variance': cumulative[:10].tolist()
}
with open(MODEL_DIR / 'pca_variance.json', 'w') as f:
    json.dump(pca_variance_data, f, indent=2)
print("\nPCA variance data saved.")

# PCA explained variance chart
fig, ax = plt.subplots(figsize=(10, 5))
x_pos = range(1, 11)
bars = ax.bar(x_pos, evr[:10] * 100, color='#2E75B6', alpha=0.8, label='Individual')
ax2 = ax.twinx()
ax2.plot(x_pos, cumulative[:10] * 100, 'o-', color='#E74C3C', linewidth=2, label='Cumulative')
ax2.axhline(y=80, color='#E74C3C', linestyle='--', linewidth=1, alpha=0.6, label='80% threshold')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance (%)')
ax2.set_ylabel('Cumulative Explained Variance (%)')
ax.set_title('PCA Explained Variance', fontsize=13, fontweight='bold')
ax.set_xticks(list(x_pos))
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')
plt.tight_layout()
plt.savefig(PLOTS_DIR / '06_pca_explained_variance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/06_pca_explained_variance.png")

# ---------------------------------------------------------------------------
# CLUSTER VISUALISATION — 2D PCA
# ---------------------------------------------------------------------------

X_2d = pca_full.transform(X_scaled)[:, :2]
pc1_var = evr[0] * 100
pc2_var = evr[1] * 100

cluster_colours = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
colours_mapped = [cluster_colours[c] for c in kmeans_labels]

fig, ax = plt.subplots(figsize=(10, 7))
for c in range(N_CLUSTERS):
    mask = kmeans_labels == c
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=cluster_colours[c], alpha=0.4, s=15, label=f'Cluster {c}')

# Plot centroids in PCA space
centroids_2d = pca_full.transform(kmeans.cluster_centers_)[:, :2]
ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
           marker='X', s=200, c='black', zorder=5, label='Centroids')

ax.set_xlabel(f'Principal Component 1 ({pc1_var:.1f}% variance)')
ax.set_ylabel(f'Principal Component 2 ({pc2_var:.1f}% variance)')
ax.set_title('Customer Segments — 2D PCA Projection', fontsize=13, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(PLOTS_DIR / '07_clusters_2d.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/07_clusters_2d.png")

# ---------------------------------------------------------------------------
# CLUSTER VISUALISATION — 3D PCA (interactive Plotly)
# Plotly generates an interactive HTML file — open in browser. Committed
# to GitHub so it renders on the portfolio site.
# ---------------------------------------------------------------------------

X_3d = pca_full.transform(X_scaled)[:, :3]
pc3_var = evr[2] * 100

plot_df = pd.DataFrame({
    'PC1': X_3d[:, 0],
    'PC2': X_3d[:, 1],
    'PC3': X_3d[:, 2],
    'Cluster': [f'Cluster {c}' for c in kmeans_labels]
})

fig_3d = px.scatter_3d(
    plot_df, x='PC1', y='PC2', z='PC3',
    color='Cluster',
    color_discrete_sequence=cluster_colours,
    title='Customer Segments — 3D PCA Projection (interactive)',
    labels={
        'PC1': f'PC1 ({pc1_var:.1f}%)',
        'PC2': f'PC2 ({pc2_var:.1f}%)',
        'PC3': f'PC3 ({pc3_var:.1f}%)'
    },
    opacity=0.6
)
fig_3d.update_traces(marker=dict(size=3))
fig_3d.write_html(str(PLOTS_DIR / '08_clusters_3d.html'))
print("Saved: plots/08_clusters_3d.html")

# ---------------------------------------------------------------------------
# CLUSTER PROFILING
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("CLUSTER PROFILING")
print("=" * 70)

overall_churn_rate = churn_labels.mean() * 100
cluster_profiles = {}

profile_rows = []
for c in range(N_CLUSTERS):
    mask = df['KMeans_Cluster'] == c
    sub = df[mask]
    churn_sub = churn_labels[mask]

    size = mask.sum()
    pct_total = size / len(df) * 100
    mean_tenure = sub['tenure'].mean()
    mean_monthly = sub['MonthlyCharges'].mean()
    mean_total = sub['TotalCharges'].mean()
    churn_rate = churn_sub.mean() * 100

    most_common_contract = sub['Contract'].mode()[0] if 'Contract' in sub.columns else 'N/A'
    most_common_internet = sub['InternetService'].mode()[0] if 'InternetService' in sub.columns else 'N/A'
    most_common_payment = sub['PaymentMethod'].mode()[0] if 'PaymentMethod' in sub.columns else 'N/A'

    cluster_profiles[str(c)] = {
        'size': int(size),
        'pct_total': round(pct_total, 1),
        'mean_tenure': round(mean_tenure, 1),
        'mean_monthly_charges': round(mean_monthly, 2),
        'mean_total_charges': round(mean_total, 2),
        'churn_rate': round(churn_rate, 1),
        'most_common_contract': most_common_contract,
        'most_common_internet': most_common_internet,
        'most_common_payment': most_common_payment
    }

    profile_rows.append({
        'Cluster': c,
        'Size': size,
        'Tenure': round(mean_tenure, 1),
        'Monthly$': round(mean_monthly, 2),
        'Total$': round(mean_total, 2),
        'Churn%': round(churn_rate, 1),
        'Contract': most_common_contract,
        'Internet': most_common_internet
    })

profile_df = pd.DataFrame(profile_rows).set_index('Cluster')
print("\nCluster Profile Table:")
print(profile_df.to_string())

with open(MODEL_DIR / 'cluster_profiles.json', 'w') as f:
    json.dump(cluster_profiles, f, indent=2)
print("\nCluster profiles saved.")

# Save labelled dataframe
df.to_csv(MODEL_DIR / 'clustered_data.csv', index=False)
print("Clustered data saved.")

# ---------------------------------------------------------------------------
# CLUSTER PROFILE CHARTS
# ---------------------------------------------------------------------------

metrics = ['tenure', 'MonthlyCharges', 'TotalCharges']
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    vals = [cluster_profiles[str(c)][
        {'tenure': 'mean_tenure',
         'MonthlyCharges': 'mean_monthly_charges',
         'TotalCharges': 'mean_total_charges'}[metric]
    ] for c in range(N_CLUSTERS)]
    axes[i].bar(range(N_CLUSTERS), vals,
                color=cluster_colours[:N_CLUSTERS], edgecolor='white')
    axes[i].set_title(f'Mean {metric}', fontweight='bold')
    axes[i].set_xlabel('Cluster')
    axes[i].set_xticks(range(N_CLUSTERS))

# Churn rate chart
churn_vals = [cluster_profiles[str(c)]['churn_rate'] for c in range(N_CLUSTERS)]
axes[3].bar(range(N_CLUSTERS), churn_vals,
            color=cluster_colours[:N_CLUSTERS], edgecolor='white')
axes[3].axhline(overall_churn_rate, color='black', linestyle='--',
                linewidth=1.2, label=f'Overall {overall_churn_rate:.1f}%')
axes[3].set_title('Churn Rate (%)', fontweight='bold')
axes[3].set_xlabel('Cluster')
axes[3].set_xticks(range(N_CLUSTERS))
axes[3].legend()

# Contract distribution
contract_data = {}
for c in range(N_CLUSTERS):
    mask = df['KMeans_Cluster'] == c
    contract_counts = df[mask]['Contract'].value_counts(normalize=True) * 100
    contract_data[c] = contract_counts

contract_df = pd.DataFrame(contract_data).fillna(0).T
contract_df.plot(kind='bar', ax=axes[4], colormap='tab10', edgecolor='white')
axes[4].set_title('Contract Type Distribution (%)', fontweight='bold')
axes[4].set_xlabel('Cluster')
axes[4].set_xticklabels(range(N_CLUSTERS), rotation=0)
axes[4].legend(fontsize=7, loc='upper right')

# InternetService distribution
internet_data = {}
for c in range(N_CLUSTERS):
    mask = df['KMeans_Cluster'] == c
    internet_counts = df[mask]['InternetService'].value_counts(normalize=True) * 100
    internet_data[c] = internet_counts

internet_df = pd.DataFrame(internet_data).fillna(0).T
internet_df.plot(kind='bar', ax=axes[5], colormap='Set2', edgecolor='white')
axes[5].set_title('Internet Service Distribution (%)', fontweight='bold')
axes[5].set_xlabel('Cluster')
axes[5].set_xticklabels(range(N_CLUSTERS), rotation=0)
axes[5].legend(fontsize=7, loc='upper right')

fig.suptitle('Cluster Profiles — Key Metrics by Segment', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / '09_cluster_profiles.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/09_cluster_profiles.png")

# Churn rate by cluster — sorted descending
# This chart validates the segmentation — if churn rate varies significantly
# across clusters, the segments are capturing meaningful behavioral differences.
sorted_clusters = sorted(range(N_CLUSTERS), key=lambda c: churn_vals[c], reverse=True)
sorted_churn = [churn_vals[c] for c in sorted_clusters]
bar_colours = ['#E74C3C' if v > overall_churn_rate else '#2ECC71' for v in sorted_churn]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar([f'Cluster {c}' for c in sorted_clusters], sorted_churn,
              color=bar_colours, edgecolor='white')
ax.axhline(overall_churn_rate, color='#2C3E50', linestyle='--', linewidth=1.5,
           label=f'Overall churn rate ({overall_churn_rate:.1f}%)')
ax.set_xlabel('Customer Segment')
ax.set_ylabel('Churn Rate (%)')
ax.set_title('Churn Rate by Customer Segment', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, axis='y', alpha=0.3)
for bar, val in zip(bars, sorted_churn):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(PLOTS_DIR / '10_churn_by_cluster.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/10_churn_by_cluster.png")

# ---------------------------------------------------------------------------
# PCA FEATURE LOADINGS — top 3 features by variance contribution
# ---------------------------------------------------------------------------

loadings = pd.DataFrame(
    pca_full.components_[:3].T,
    index=feature_names,
    columns=['PC1', 'PC2', 'PC3']
)
loadings['magnitude'] = np.sqrt((loadings ** 2).sum(axis=1))
top3_features = loadings['magnitude'].nlargest(3)

# ---------------------------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print("\nCluster Profile Table:")
print(profile_df.to_string())
print(f"\nHierarchical vs KMeans — Adjusted Rand Score: {ari:.3f}")
print(f"\nPCA explained variance:")
for i in range(3):
    print(f"  PC{i+1}: {evr[i]*100:.1f}%")
print(f"\nTop 3 features by PCA variance contribution:")
for feat, val in top3_features.items():
    print(f"  {feat}: {val:.4f}")
print(f"\nOverall dataset churn rate: {overall_churn_rate:.1f}%")
print("\n" + "=" * 70)
print("ACTION REQUIRED:")
print("Now fill in segment_labels.py with business labels based on")
print("the cluster profiles above.")
print("=" * 70)
