"""
BRFSS K-Means Clustering Pipeline
====================================
Unsupervised clustering on the full 391-row dataset.

Missing PoorDiet values (307 rows) are imputed using the mean
PoorDiet per stratification group — preserving the demographic
diet patterns observed in EDA rather than applying a single
global mean.

Features used for clustering:
    - Obesity (from obesity_national)
    - Inactive (from activity_national)
    - PoorDiet (from diet_national, imputed where missing)

Steps:
    1. Build full dataset with stratification-group mean imputation
    2. Find optimal k using elbow method and silhouette score
    3. Fit K-Means with optimal k
    4. Visualise clusters in 2D using PCA
    5. Profile each cluster
    6. Compare cluster membership against stratification categories

Usage:
    python brfss_cluster.py --input_dir clean_outputs/ --output_dir cluster_outputs/

Requirements:
    pip install pandas numpy scikit-learn matplotlib seaborn pyarrow
"""

import argparse
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

INDEX     = ['YearStart', 'StratificationCategory1', 'Stratification1']
MAIN_BLUE = '#2E75B6'
DARK_BLUE = '#1F3864'
RED       = '#C00000'
ORANGE    = '#ED7D31'
GREEN     = '#70AD47'
PURPLE    = '#7030A0'
CLUSTER_COLORS = [MAIN_BLUE, RED, GREEN, ORANGE, PURPLE, DARK_BLUE]


# ── Helpers ───────────────────────────────────────────────────────────────────

def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"      Saved -> {path}")


def apply_style():
    sns.set_theme(style='whitegrid', font='Arial')
    plt.rcParams.update({
        'axes.titlesize': 13, 'axes.labelsize': 11,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
    })


# ── Step 1 — Build dataset with stratification-group mean imputation ──────────

def build_dataset(obesity, activity, diet):
    """
    Join obesity + activity (391 rows), then join diet (84 rows).
    Impute missing PoorDiet using the mean per stratification group
    (StratificationCategory1 x Stratification1), preserving demographic
    diet patterns rather than applying a single global mean.
    """
    print("\n[1/6] Building dataset with stratification-group mean imputation...")

    # Join obesity and activity
    df = obesity[INDEX + ['Obesity']].merge(
        activity[INDEX + ['Inactive']], on=INDEX, how='inner'
    )

    # Join diet — leaves PoorDiet null for 307 rows not covered by diet data
    df = df.merge(
        diet[INDEX + ['PoorDiet']], on=INDEX, how='left'
    )

    print(f"      Shape after join: {df.shape}")
    print(f"      PoorDiet missing before imputation: {df['PoorDiet'].isna().sum()} "
          f"({df['PoorDiet'].isna().mean():.1%})")

    # Compute mean PoorDiet per stratification group from the 84 real rows
    group_means = (
        df.groupby(['StratificationCategory1', 'Stratification1'])['PoorDiet']
        .mean()
        .reset_index()
        .rename(columns={'PoorDiet': 'PoorDiet_GroupMean'})
    )

    df = df.merge(group_means,
                  on=['StratificationCategory1', 'Stratification1'],
                  how='left')

    # Fill missing PoorDiet with the group mean
    df['PoorDiet_imputed'] = df['PoorDiet'].isna()
    df['PoorDiet'] = df['PoorDiet'].fillna(df['PoorDiet_GroupMean'])

    # For any groups with no diet data at all, fall back to global mean
    global_mean = diet['PoorDiet'].mean()
    remaining_null = df['PoorDiet'].isna().sum()
    if remaining_null > 0:
        print(f"      {remaining_null} rows have no group mean - using global mean "
              f"({global_mean:.2f}%)")
        df['PoorDiet'] = df['PoorDiet'].fillna(global_mean)

    df = df.drop(columns=['PoorDiet_GroupMean'])
    df = df.dropna(subset=['Obesity', 'Inactive', 'PoorDiet']).reset_index(drop=True)

    print(f"      PoorDiet missing after imputation: {df['PoorDiet'].isna().sum()}")
    print(f"      Rows with real diet data: {(~df['PoorDiet_imputed']).sum()}")
    print(f"      Rows with imputed diet data: {df['PoorDiet_imputed'].sum()}")
    print(f"      Final shape: {df.shape}")

    # Print group means used for imputation
    print(f"\n      Group means used for imputation:")
    gm = group_means.sort_values(['StratificationCategory1', 'Stratification1'])
    for _, row in gm.iterrows():
        if not np.isnan(row['PoorDiet_GroupMean']):
            print(f"        {row['StratificationCategory1']:<20} "
                  f"{row['Stratification1']:<30} "
                  f"PoorDiet mean: {row['PoorDiet_GroupMean']:.2f}%")

    return df


# ── Step 2 — Find optimal k ───────────────────────────────────────────────────

def find_optimal_k(X_scaled, out_dir, k_range=range(2, 9)):
    """Elbow method and silhouette score to find optimal k."""
    print("\n[2/6] Finding optimal k...")

    inertias    = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels)
        silhouettes.append(sil)
        print(f"      k={k}  Inertia={km.inertia_:.1f}  Silhouette={sil:.4f}")

    # Plot elbow + silhouette
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(list(k_range), inertias, color=MAIN_BLUE,
            linewidth=2, marker='o', markersize=7)
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)')
    ax.set_title('Elbow Method', fontweight='bold')
    ax.set_xticks(list(k_range))

    ax = axes[1]
    ax.plot(list(k_range), silhouettes, color=RED,
            linewidth=2, marker='o', markersize=7)
    best_k = list(k_range)[np.argmax(silhouettes)]
    ax.axvline(best_k, color=DARK_BLUE, linestyle='--',
               linewidth=1.5, label=f'Best k={best_k}')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score', fontweight='bold')
    ax.set_xticks(list(k_range))
    ax.legend(frameon=False)

    fig.suptitle('Optimal k Selection — Elbow Method & Silhouette Score',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    save(fig, os.path.join(out_dir, 'cluster_1_optimal_k.png'))

    print(f"\n      Best k by silhouette: {best_k} "
          f"(score={max(silhouettes):.4f})")
    return best_k, inertias, silhouettes


# ── Step 3 — Fit K-Means ──────────────────────────────────────────────────────

def fit_kmeans(df, X_scaled, k):
    """Fit final K-Means model and add cluster labels to dataframe."""
    print(f"\n[3/6] Fitting K-Means with k={k}...")
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    df['Cluster'] = km.fit_predict(X_scaled)
    print(f"      Cluster sizes:")
    for c, n in df['Cluster'].value_counts().sort_index().items():
        print(f"        Cluster {c}: {n} rows ({n/len(df):.1%})")
    return df, km


# ── Step 4 — PCA visualisation ────────────────────────────────────────────────

def plot_pca(df, X_scaled, k, out_dir):
    """Reduce to 2D with PCA and plot coloured by cluster."""
    print("\n[4/6] PCA visualisation...")
    pca  = PCA(n_components=2, random_state=42)
    pca2 = pca.fit_transform(X_scaled)

    df['PCA1'] = pca2[:, 0]
    df['PCA2'] = pca2[:, 1]

    var_exp = pca.explained_variance_ratio_
    print(f"      PCA variance explained: PC1={var_exp[0]:.1%}  "
          f"PC2={var_exp[1]:.1%}  Total={sum(var_exp):.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left — coloured by cluster
    ax = axes[0]
    for c in sorted(df['Cluster'].unique()):
        sub = df[df['Cluster'] == c]
        ax.scatter(sub['PCA1'], sub['PCA2'],
                   color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                   alpha=0.6, s=25, edgecolors='white', linewidth=0.3,
                   label=f'Cluster {c} (n={len(sub)})')
    ax.set_xlabel(f'PC1 ({var_exp[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({var_exp[1]:.1%} variance)')
    ax.set_title('Clusters (K-Means)', fontweight='bold')
    ax.legend(frameon=False, fontsize=8)

    # Right — coloured by stratification category
    ax = axes[1]
    cats    = df['StratificationCategory1'].unique()
    palette = sns.color_palette('tab10', len(cats))
    for cat, color in zip(cats, palette):
        sub = df[df['StratificationCategory1'] == cat]
        ax.scatter(sub['PCA1'], sub['PCA2'],
                   color=color, alpha=0.6, s=25,
                   edgecolors='white', linewidth=0.3, label=cat)
    ax.set_xlabel(f'PC1 ({var_exp[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({var_exp[1]:.1%} variance)')
    ax.set_title('Stratification Category', fontweight='bold')
    ax.legend(frameon=False, fontsize=7, loc='upper right')

    fig.suptitle(f'K-Means Clusters (k={k}) - PCA 2D Projection\n'
                 f'Features: Obesity, Inactive, PoorDiet (imputed for 307 rows)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    save(fig, os.path.join(out_dir, 'cluster_2_pca.png'))
    return df, var_exp


# ── Step 5 — Cluster profiles ─────────────────────────────────────────────────

def profile_clusters(df, k, out_dir):
    """Mean of each feature per cluster."""
    print("\n[5/6] Profiling clusters...")

    profile = df.groupby('Cluster')[['Obesity', 'Inactive', 'PoorDiet']].mean().round(2)
    print(profile.to_string())
    profile.to_csv(os.path.join(out_dir, 'cluster_profiles.csv'))

    # Radar-style bar chart — one group per cluster
    features = ['Obesity', 'Inactive', 'PoorDiet']
    x = np.arange(len(features))
    w = 0.8 / k

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, c in enumerate(range(k)):
        vals = [profile.loc[c, f] for f in features]
        ax.bar(x + i * w - (k-1) * w / 2, vals, w,
               color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
               edgecolor='white', linewidth=0.5,
               label=f'Cluster {c}')

    ax.set_xticks(x)
    ax.set_xticklabels(features)
    ax.set_ylabel('Mean Prevalence (%)')
    ax.set_title(f'Cluster Profiles - Mean Feature Values (k={k})',
                 fontweight='bold')
    ax.legend(frameon=False, fontsize=9)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    save(fig, os.path.join(out_dir, 'cluster_3_profiles.png'))

    return profile


# ── Step 6 — Cluster vs stratification crosstab ───────────────────────────────

def cluster_vs_strat(df, k, out_dir):
    """Heatmap showing how clusters map to stratification categories."""
    print("\n[6/6] Cluster vs stratification crosstab...")

    # Proportion of each stratification category in each cluster
    crosstab = pd.crosstab(
        df['StratificationCategory1'],
        df['Cluster'],
        normalize='index'
    ).round(3)

    print(crosstab.to_string())
    crosstab.to_csv(os.path.join(out_dir, 'cluster_crosstab.csv'))

    fig, ax = plt.subplots(figsize=(max(6, k * 1.5), 5))
    sns.heatmap(crosstab, annot=True, fmt='.2f', cmap='Blues',
                linewidths=0.5, linecolor='white', ax=ax,
                annot_kws={'size': 10})
    ax.set_title(f'Cluster Distribution by Stratification Category\n'
                 f'(Row proportions - how each category splits across clusters)',
                 fontweight='bold', pad=12)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Stratification Category')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    save(fig, os.path.join(out_dir, 'cluster_4_crosstab.png'))

    # Also plot: for each cluster, show breakdown by strat category
    cluster_comp = pd.crosstab(
        df['Cluster'],
        df['StratificationCategory1'],
        normalize='index'
    ).round(3)

    fig, ax = plt.subplots(figsize=(10, max(3.5, k * 0.8)))
    cluster_comp.plot(kind='barh', stacked=True,
                      colormap='tab10', ax=ax, edgecolor='white')
    ax.set_xlabel('Proportion')
    ax.set_ylabel('Cluster')
    ax.set_title(f'Composition of Each Cluster by Stratification Category',
                 fontweight='bold')
    ax.legend(frameon=False, fontsize=8,
              bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    save(fig, os.path.join(out_dir, 'cluster_5_composition.png'))

    return crosstab


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='BRFSS K-Means clustering')
    parser.add_argument('--input_dir',  default='clean_outputs')
    parser.add_argument('--output_dir', default='cluster_outputs')
    parser.add_argument('--k',          type=int, default=None,
                        help='Force a specific k (default: auto from silhouette)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    apply_style()

    print(f"\nBRFSS K-Means Clustering Pipeline")
    print(f"{'='*60}")

    # Load
    obesity  = pd.read_parquet(os.path.join(args.input_dir, 'obesity_national.parquet'))
    activity = pd.read_parquet(os.path.join(args.input_dir, 'activity_national.parquet'))
    diet     = pd.read_parquet(os.path.join(args.input_dir, 'diet_national.parquet'))

    # Build dataset
    df = build_dataset(obesity, activity, diet)

    # Scale features
    features  = ['Obesity', 'Inactive', 'PoorDiet']
    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(df[features])

    # Find optimal k
    if args.k:
        k = args.k
        print(f"\n[2/6] Using specified k={k}")
    else:
        k, _, _ = find_optimal_k(X_scaled, args.output_dir)

    # Fit K-Means
    df, km = fit_kmeans(df, X_scaled, k)

    # PCA visualisation
    df, var_exp = plot_pca(df, X_scaled, k, args.output_dir)

    # Cluster profiles
    profile = profile_clusters(df, k, args.output_dir)

    # Cluster vs stratification
    crosstab = cluster_vs_strat(df, k, args.output_dir)

    # Save full clustered dataset
    out_path = os.path.join(args.output_dir, 'clustered_dataset.parquet')
    df.to_parquet(out_path, index=False)
    print(f"\n      Full clustered dataset saved -> {out_path}")

    print(f"\n{'='*60}")
    print(f"Clustering complete. Outputs saved to: {args.output_dir}/")
    print(f"\n  cluster_1_optimal_k.png    - elbow and silhouette charts")
    print(f"  cluster_2_pca.png          - 2D PCA projection coloured by cluster and strat")
    print(f"  cluster_3_profiles.png     - mean feature values per cluster")
    print(f"  cluster_4_crosstab.png     - heatmap: strat category vs cluster")
    print(f"  cluster_5_composition.png  - stacked bar: composition of each cluster")
    print(f"  cluster_profiles.csv       - numeric cluster profiles")
    print(f"  cluster_crosstab.csv       - numeric crosstab")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()