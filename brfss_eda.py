"""
BRFSS Exploratory Data Analysis (EDA) Pipeline
================================================
Answers the following questions before modeling:

OBESITY (391 rows, 2011-2024):
  1. What does the national obesity distribution look like?
  2. Is there a clear upward trend over time?
  3. Which demographic groups have the highest obesity rates?
  4. How much does obesity vary across stratification groups?

ACTIVITY (391 rows, 2011-2024):
  1. What does the national inactivity distribution look like?
  2. Has inactivity changed over time?
  3. Which demographic groups are most inactive?
  4. Does inactivity move with obesity?

DIET (84 rows, 2017, 2019, 2021):
  1. What are national LowFruit, LowVeg, and PoorDiet rates?
  2. How does PoorDiet vary across demographic groups?
  3. Does PoorDiet correlate with Obesity in the overlapping years?

COMBINED (84 rows where all three overlap):
  1. Correlation between Obesity, Inactive, PoorDiet
  2. Are predictors collinear?

Usage:
    python brfss_eda.py --input_dir clean_outputs/ --output_dir eda_outputs/

Requirements:
    pip install pandas matplotlib seaborn statsmodels pyarrow scipy
"""

import argparse
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────────

MAIN_BLUE = '#2E75B6'
DARK_BLUE = '#1F3864'
RED       = '#C00000'
ORANGE    = '#ED7D31'
GREEN     = '#70AD47'

STRAT_ORDER = {
    'Age (years)':    ['18 - 24', '25 - 34', '35 - 44', '45 - 54', '55 - 64', '65 or older'],
    'Income':         ['Less than $15,000', '$15,000 - $24,999', '$25,000 - $34,999',
                       '$35,000 - $49,999', '$50,000 - $74,999', '$75,000 or greater'],
    'Sex':            ['Male', 'Female'],
    'Education':      ['Less than high school', 'High school graduate',
                       'Some college or technical school', 'College graduate'],
    'Race/Ethnicity': None,  # sorted by median
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"      Saved -> {path}")


def apply_style():
    sns.set_theme(style='whitegrid', font='Arial')
    plt.rcParams.update({
        'axes.titlesize':  13,
        'axes.labelsize':  11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.spines.top':   False,
        'axes.spines.right': False,
    })


def total_only(df):
    """Filter to Total/Overall stratification only."""
    return df[df['StratificationCategory1'] == 'Total'].copy()


def strat_only(df, category):
    """Filter to a specific stratification category."""
    return df[df['StratificationCategory1'] == category].copy()


def trend_line(ax, x, y, color='black'):
    """Add a linear trend line to an axis."""
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(sorted(x), p(sorted(x)),
            linestyle='--', linewidth=1.2, color=color, alpha=0.6,
            label=f'Trend ({z[0]:+.2f}%/yr)')


# ── PART 1 - OBESITY ──────────────────────────────────────────────────────────

def eda_obesity(obesity, out_dir):
    print("\n" + "="*60)
    print("PART 1 - OBESITY")
    print("="*60)

    total = total_only(obesity)

    # Q1 - Distribution
    print("\n  Q1: Distribution of national obesity prevalence")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # All strata
    ax = axes[0]
    sns.histplot(obesity['Obesity'], bins=35, kde=True,
                 color=MAIN_BLUE, edgecolor='white', linewidth=0.4, ax=ax)
    ax.axvline(obesity['Obesity'].mean(), color=DARK_BLUE, linestyle='--',
               linewidth=1.4, label=f"Mean: {obesity['Obesity'].mean():.1f}%")
    ax.axvline(obesity['Obesity'].median(), color=RED, linestyle=':',
               linewidth=1.4, label=f"Median: {obesity['Obesity'].median():.1f}%")
    ax.set_title('All Stratification Groups', fontweight='bold')
    ax.set_xlabel('National Obesity Prevalence (%)')
    ax.set_ylabel('Count')
    ax.legend(frameon=False, fontsize=9)

    # Total only
    ax = axes[1]
    sns.histplot(total['Obesity'], bins=20, kde=True,
                 color=DARK_BLUE, edgecolor='white', linewidth=0.4, ax=ax)
    ax.axvline(total['Obesity'].mean(), color=MAIN_BLUE, linestyle='--',
               linewidth=1.4, label=f"Mean: {total['Obesity'].mean():.1f}%")
    ax.axvline(total['Obesity'].median(), color=RED, linestyle=':',
               linewidth=1.4, label=f"Median: {total['Obesity'].median():.1f}%")
    ax.set_title('Total Population Only', fontweight='bold')
    ax.set_xlabel('National Obesity Prevalence (%)')
    ax.set_ylabel('Count')
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle('Q1 - Distribution of National Obesity Prevalence',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    save(fig, os.path.join(out_dir, 'obesity_q1_distribution.png'))

    # Print stats
    print(f"      All strata  - Mean: {obesity['Obesity'].mean():.1f}%  "
          f"Std: {obesity['Obesity'].std():.1f}%  "
          f"Min: {obesity['Obesity'].min():.1f}%  "
          f"Max: {obesity['Obesity'].max():.1f}%")
    print(f"      Total only  - Mean: {total['Obesity'].mean():.1f}%  "
          f"Std: {total['Obesity'].std():.1f}%  "
          f"Min: {total['Obesity'].min():.1f}%  "
          f"Max: {total['Obesity'].max():.1f}%")

    # Q2 - Trend over time (Total only)
    print("\n  Q2: Obesity trend over time (2011-2024)")
    trend = total.groupby('YearStart')['Obesity'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.fill_between(trend['YearStart'], trend['Obesity'],
                    alpha=0.15, color=RED)
    ax.plot(trend['YearStart'], trend['Obesity'],
            color=RED, linewidth=2.5, marker='o', markersize=6,
            label='National Obesity Rate')
    trend_line(ax, trend['YearStart'], trend['Obesity'], RED)
    ax.set_title('Q2 - National Obesity Trend (Total Population, 2011-2024)',
                 fontweight='bold', pad=12)
    ax.set_xlabel('Year')
    ax.set_ylabel('Obesity Prevalence (%)')
    ax.set_xticks(trend['YearStart'])
    ax.tick_params(axis='x', rotation=45)
    ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    save(fig, os.path.join(out_dir, 'obesity_q2_trend.png'))

    start = trend.loc[trend['YearStart'] == trend['YearStart'].min(), 'Obesity'].values[0]
    end   = trend.loc[trend['YearStart'] == trend['YearStart'].max(), 'Obesity'].values[0]
    print(f"      {int(trend['YearStart'].min())}: {start:.1f}%  ->  "
          f"{int(trend['YearStart'].max())}: {end:.1f}%  "
          f"(+{end - start:.1f} percentage points)")

    # Q3 - Obesity by demographic group
    print("\n  Q3: Obesity by demographic group")
    categories = [c for c in STRAT_ORDER.keys()
                  if c in obesity['StratificationCategory1'].unique()]

    for cat in categories:
        sub = strat_only(obesity, cat)
        if sub.empty:
            continue

        order = STRAT_ORDER.get(cat)
        if order is None:
            order = (sub.groupby('Stratification1')['Obesity']
                     .median().sort_values(ascending=False).index.tolist())

        # Keep only groups that exist in data
        order = [g for g in order if g in sub['Stratification1'].unique()]

        fig, ax = plt.subplots(figsize=(11, 4.5))
        sns.boxplot(data=sub, x='Stratification1', y='Obesity',
                    order=order, palette='Blues_r', ax=ax, linewidth=0.8,
                    flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.4})
        ax.set_title(f'Q3 - National Obesity Prevalence by {cat}',
                     fontweight='bold', pad=12)
        ax.set_xlabel(cat)
        ax.set_ylabel('Obesity (%)')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
        plt.xticks(rotation=30, ha='right')
        fname = f"obesity_q3_{cat.lower().replace('/', '_').replace(' ', '_')}.png"
        save(fig, os.path.join(out_dir, fname))
        print(f"      {cat} done")

    # Q4 - Variation across stratification groups (latest year)
    print("\n  Q4: Obesity variation across stratification groups")
    latest = obesity[obesity['YearStart'] == obesity['YearStart'].max()]
    summary = (latest.groupby('StratificationCategory1')['Obesity']
               .agg(['mean', 'std', 'min', 'max'])
               .round(2))
    print(summary.to_string())

    fig, ax = plt.subplots(figsize=(10, 4))
    cats = summary.index.tolist()
    means = summary['mean'].values
    stds  = summary['std'].values
    bars  = ax.bar(cats, means, color=MAIN_BLUE, edgecolor='white', linewidth=0.5)
    ax.errorbar(cats, means, yerr=stds, fmt='none',
                color=DARK_BLUE, capsize=4, linewidth=1.5)
    ax.set_title(f'Q4 - Obesity Variation by Stratification Category '
                 f'({int(latest["YearStart"].max())})',
                 fontweight='bold', pad=12)
    ax.set_ylabel('Mean Obesity (%)')
    ax.set_xlabel('')
    plt.xticks(rotation=20, ha='right')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    save(fig, os.path.join(out_dir, 'obesity_q4_variation_by_strat.png'))


# ── PART 2 - ACTIVITY ─────────────────────────────────────────────────────────

def eda_activity(activity, obesity, out_dir):
    print("\n" + "="*60)
    print("PART 2 - PHYSICAL ACTIVITY")
    print("="*60)

    total_act = total_only(activity)
    total_obs = total_only(obesity)

    # Q1 - Distribution
    print("\n  Q1: Distribution of national inactivity")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    sns.histplot(activity['Inactive'], bins=35, kde=True,
                 color=MAIN_BLUE, edgecolor='white', linewidth=0.4, ax=ax)
    ax.axvline(activity['Inactive'].mean(), color=DARK_BLUE, linestyle='--',
               linewidth=1.4, label=f"Mean: {activity['Inactive'].mean():.1f}%")
    ax.set_title('All Stratification Groups', fontweight='bold')
    ax.set_xlabel('Physical Inactivity (%)')
    ax.set_ylabel('Count')
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    sns.histplot(total_act['Inactive'], bins=20, kde=True,
                 color=DARK_BLUE, edgecolor='white', linewidth=0.4, ax=ax)
    ax.axvline(total_act['Inactive'].mean(), color=MAIN_BLUE, linestyle='--',
               linewidth=1.4, label=f"Mean: {total_act['Inactive'].mean():.1f}%")
    ax.set_title('Total Population Only', fontweight='bold')
    ax.set_xlabel('Physical Inactivity (%)')
    ax.set_ylabel('Count')
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle('Q1 - Distribution of National Physical Inactivity',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    save(fig, os.path.join(out_dir, 'activity_q1_distribution.png'))

    # Q2 - Trend over time
    print("\n  Q2: Inactivity trend over time")
    trend_act = total_act.groupby('YearStart')['Inactive'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.fill_between(trend_act['YearStart'], trend_act['Inactive'],
                    alpha=0.15, color=MAIN_BLUE)
    ax.plot(trend_act['YearStart'], trend_act['Inactive'],
            color=MAIN_BLUE, linewidth=2.5, marker='o', markersize=6,
            label='Physical Inactivity')
    trend_line(ax, trend_act['YearStart'], trend_act['Inactive'], MAIN_BLUE)
    ax.set_title('Q2 - National Physical Inactivity Trend (2011-2024)',
                 fontweight='bold', pad=12)
    ax.set_xlabel('Year')
    ax.set_ylabel('Inactivity Prevalence (%)')
    ax.set_xticks(trend_act['YearStart'])
    ax.tick_params(axis='x', rotation=45)
    ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    save(fig, os.path.join(out_dir, 'activity_q2_trend.png'))

    # Q3 - Inactivity by demographic group
    print("\n  Q3: Inactivity by demographic group")
    categories = [c for c in STRAT_ORDER.keys()
                  if c in activity['StratificationCategory1'].unique()]

    for cat in categories:
        sub = strat_only(activity, cat)
        if sub.empty:
            continue
        order = STRAT_ORDER.get(cat)
        if order is None:
            order = (sub.groupby('Stratification1')['Inactive']
                     .median().sort_values(ascending=False).index.tolist())
        order = [g for g in order if g in sub['Stratification1'].unique()]

        fig, ax = plt.subplots(figsize=(11, 4.5))
        sns.boxplot(data=sub, x='Stratification1', y='Inactive',
                    order=order, palette='Blues_r', ax=ax, linewidth=0.8,
                    flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.4})
        ax.set_title(f'Q3 - National Inactivity by {cat}',
                     fontweight='bold', pad=12)
        ax.set_xlabel(cat)
        ax.set_ylabel('Inactivity (%)')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
        plt.xticks(rotation=30, ha='right')
        fname = f"activity_q3_{cat.lower().replace('/', '_').replace(' ', '_')}.png"
        save(fig, os.path.join(out_dir, fname))
        print(f"      {cat} done")

    # Q4 - Does inactivity move with obesity?
    print("\n  Q4: Does inactivity move with obesity over time?")
    merged = trend_act.merge(
        total_obs.groupby('YearStart')['Obesity'].mean().reset_index(),
        on='YearStart'
    )

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    ax1.plot(merged['YearStart'], merged['Obesity'],
             color=RED, linewidth=2.5, marker='o', markersize=6,
             label='Obesity')
    ax1.set_ylabel('Obesity Prevalence (%)', color=RED)
    ax1.tick_params(axis='y', labelcolor=RED)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))

    ax2.plot(merged['YearStart'], merged['Inactive'],
             color=MAIN_BLUE, linewidth=2.5, marker='s', markersize=6,
             label='Inactivity')
    ax2.set_ylabel('Inactivity Prevalence (%)', color=MAIN_BLUE)
    ax2.tick_params(axis='y', labelcolor=MAIN_BLUE)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))

    ax1.set_title('Q4 - Obesity vs Physical Inactivity Over Time (2011-2024)',
                  fontweight='bold', pad=12)
    ax1.set_xlabel('Year')
    ax1.set_xticks(merged['YearStart'])
    ax1.tick_params(axis='x', rotation=45)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc='upper left')

    corr = merged['Obesity'].corr(merged['Inactive'])
    ax1.text(0.98, 0.05, f'Correlation: {corr:.2f}',
             transform=ax1.transAxes, ha='right', fontsize=10,
             color='#333333', style='italic')
    save(fig, os.path.join(out_dir, 'activity_q4_vs_obesity.png'))
    print(f"      Correlation Obesity vs Inactive (Total, over time): {corr:.3f}")


# ── PART 3 - DIET ─────────────────────────────────────────────────────────────

def eda_diet(diet, obesity, out_dir):
    print("\n" + "="*60)
    print("PART 3 - DIET")
    print("="*60)

    total_diet = total_only(diet)
    total_obs  = total_only(obesity)

    # Q1 - LowFruit, LowVeg, PoorDiet rates
    print("\n  Q1: National diet rates (2017, 2019, 2021)")
    print(total_diet[['YearStart', 'LowFruit', 'LowVeg', 'PoorDiet']].to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col, color, title in zip(
        axes,
        ['LowFruit', 'LowVeg', 'PoorDiet'],
        [MAIN_BLUE, GREEN, ORANGE],
        ['Low Fruit Intake (%)', 'Low Vegetable Intake (%)', 'PoorDiet Score (%)']
    ):
        sns.barplot(data=total_diet, x='YearStart', y=col,
                    color=color, ax=ax, edgecolor='white')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('%')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))

    fig.suptitle('Q1 - National Diet Indicators (Total Population, Available Years)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    save(fig, os.path.join(out_dir, 'diet_q1_rates.png'))

    # Q2 - PoorDiet by demographic group
    print("\n  Q2: PoorDiet by demographic group")
    categories = [c for c in STRAT_ORDER.keys()
                  if c in diet['StratificationCategory1'].unique()]

    for cat in categories:
        sub = strat_only(diet, cat)
        if sub.empty:
            continue
        order = STRAT_ORDER.get(cat)
        if order is None:
            order = (sub.groupby('Stratification1')['PoorDiet']
                     .median().sort_values(ascending=False).index.tolist())
        order = [g for g in order if g in sub['Stratification1'].unique()]

        fig, ax = plt.subplots(figsize=(11, 4.5))
        sns.boxplot(data=sub, x='Stratification1', y='PoorDiet',
                    order=order, palette='Oranges_r', ax=ax, linewidth=0.8,
                    flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.4})
        ax.set_title(f'Q2 - PoorDiet Score by {cat} (2017, 2019, 2021)',
                     fontweight='bold', pad=12)
        ax.set_xlabel(cat)
        ax.set_ylabel('PoorDiet (%)')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
        plt.xticks(rotation=30, ha='right')
        fname = f"diet_q2_{cat.lower().replace('/', '_').replace(' ', '_')}.png"
        save(fig, os.path.join(out_dir, fname))
        print(f"      {cat} done")

    # Q3 - PoorDiet vs Obesity (overlapping years)
    print("\n  Q3: PoorDiet vs Obesity correlation")
    diet_obs = total_diet.merge(
        total_obs[['YearStart', 'Obesity']],
        on='YearStart', how='inner'
    )

    if diet_obs.empty:
        print("      No overlapping years between diet and obesity - skipping")
        return

    print(f"      Overlapping years: {sorted(diet_obs['YearStart'].unique().tolist())}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, col, color, label in zip(
        axes,
        ['PoorDiet', 'Inactive' if 'Inactive' in diet_obs.columns else 'PoorDiet'],
        [ORANGE, MAIN_BLUE],
        ['PoorDiet Score (%)', 'Physical Inactivity (%)']
    ):
        if col not in diet_obs.columns:
            continue
        ax.scatter(diet_obs[col], diet_obs['Obesity'],
                   color=color, s=80, alpha=0.8, edgecolors='white', linewidth=0.5)
        # Add year labels
        for _, row in diet_obs.iterrows():
            ax.annotate(str(int(row['YearStart'])),
                        (row[col], row['Obesity']),
                        textcoords='offset points', xytext=(6, 4), fontsize=8)
        z = np.polyfit(diet_obs[col], diet_obs['Obesity'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(diet_obs[col].min(), diet_obs[col].max(), 100)
        ax.plot(x_line, p(x_line), color=color, linestyle='--', linewidth=1.5, alpha=0.7)
        corr = diet_obs['Obesity'].corr(diet_obs[col])
        ax.set_title(f'Obesity vs {label}\nr = {corr:.2f}', fontweight='bold')
        ax.set_xlabel(label)
        ax.set_ylabel('Obesity (%)')

    fig.suptitle('Q3 - Diet vs Obesity (Total Population, Overlapping Years)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    save(fig, os.path.join(out_dir, 'diet_q3_vs_obesity.png'))


# ── PART 4 - COMBINED ─────────────────────────────────────────────────────────

def eda_combined(obesity, activity, diet, out_dir):
    print("\n" + "="*60)
    print("PART 4 - COMBINED (Obesity + Activity + Diet)")
    print("="*60)

    # Join all three on the full common key - all strata, diet years only
    # This gives 84 rows (all stratification groups for 2017, 2019, 2021)
    INDEX = ['YearStart', 'StratificationCategory1', 'Stratification1']

    combined = (
        obesity[INDEX + ['Obesity']]
        .merge(activity[INDEX + ['Inactive']], on=INDEX, how='inner')
        .merge(diet[INDEX + ['PoorDiet', 'LowFruit', 'LowVeg']], on=INDEX, how='inner')
        .dropna()
    )

    print(f"\n  Combined dataset: {len(combined)} rows")
    print(f"  Years: {sorted(combined['YearStart'].unique().astype(int).tolist())}")
    print(f"  Strata: {combined['StratificationCategory1'].unique().tolist()}")
    print(f"\n  Summary:")
    print(combined[['Obesity', 'Inactive', 'PoorDiet']].describe().round(2).to_string())

    if combined.empty or len(combined) < 3:
        print("  Not enough overlapping data for combined analysis")
        return

    # Q1 - Correlation matrix
    print("\n  Q1: Correlation between Obesity, Inactive, PoorDiet")
    corr = combined[['Obesity', 'Inactive', 'PoorDiet']].corr()
    print(corr.round(3).to_string())

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                vmin=-1, vmax=1, linewidths=0.5, linecolor='white',
                ax=ax, annot_kws={'size': 13})
    ax.set_title('Q1 - Correlation Matrix\n(All Strata, 84 rows, 2017/2019/2021)',
                 fontweight='bold', pad=12)
    plt.xticks(rotation=20, ha='right')
    plt.yticks(rotation=0)
    save(fig, os.path.join(out_dir, 'combined_q1_correlation.png'))

    # Q2 - VIF
    print("\n  Q2: VIF - multicollinearity check")
    predictors = [c for c in ['Inactive', 'PoorDiet'] if c in combined.columns]
    X = combined[predictors].dropna()

    if len(X) >= len(predictors) + 1:
        vif_data = pd.DataFrame({
            'Feature': predictors,
            'VIF': [variance_inflation_factor(X.values, i)
                    for i in range(len(predictors))],
        })
        print(vif_data.to_string(index=False))

        fig, ax = plt.subplots(figsize=(6, 3))
        colors = [RED if v > 10 else ORANGE if v > 5 else MAIN_BLUE
                  for v in vif_data['VIF']]
        bars = ax.barh(vif_data['Feature'], vif_data['VIF'],
                       color=colors, edgecolor='white')
        ax.axvline(5,  color=ORANGE, linestyle='--', linewidth=1.2,
                   label='Warning (VIF=5)')
        ax.axvline(10, color=RED,    linestyle='--', linewidth=1.2,
                   label='Problem (VIF=10)')
        for bar, val in zip(bars, vif_data['VIF']):
            ax.text(bar.get_width() + 0.05,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.2f}', va='center', fontsize=10)
        ax.set_xlabel('Variance Inflation Factor')
        ax.set_title('Q2 - Multicollinearity Check (VIF)',
                     fontweight='bold', pad=12)
        ax.legend(frameon=False, fontsize=9)
        save(fig, os.path.join(out_dir, 'combined_q2_vif.png'))
    else:
        print(f"  Only {len(X)} rows - insufficient for VIF calculation")

    # Scatter matrix of all three
    fig = plt.figure(figsize=(9, 7))
    pd.plotting.scatter_matrix(
        combined[['Obesity', 'Inactive', 'PoorDiet']],
        ax=fig.add_subplot(111),
        figsize=(9, 7),
        alpha=0.7,
        color=MAIN_BLUE,
        diagonal='kde',
    )
    fig.suptitle('Combined - Scatter Matrix (All Strata, 84 rows, 2017/2019/2021)',
                 fontsize=12, fontweight='bold')
    save(fig, os.path.join(out_dir, 'combined_scatter_matrix.png'))


# ── Summary Statistics & Correlations ────────────────────────────────────────

def compute_summary_stats(obesity, activity, diet, out_dir):
    """
    Compute and save summary statistics and correlations on the
    correct full datasets - not filtered to Total only.

    Dataset 1: Obesity + Activity joined (391 rows)
    Dataset 2: Obesity + Activity + Diet joined (84 rows)
    """
    from scipy import stats as scipy_stats

    INDEX = ['YearStart', 'StratificationCategory1', 'Stratification1']

    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS & CORRELATIONS")
    print(f"{'='*60}")

    # ── Summary statistics per indicator ──────────────────────────────────
    print("\n[Stats 1/4] Summary statistics per indicator")
    rows = []
    for label, df, col in [
        ('Obesity',  obesity,  'Obesity'),
        ('Inactive', activity, 'Inactive'),
        ('PoorDiet', diet,     'PoorDiet'),
        ('LowFruit', diet,     'LowFruit'),
        ('LowVeg',   diet,     'LowVeg'),
    ]:
        s = df[col].dropna()
        rows.append({
            'Indicator': label,
            'N':      len(s),
            'Mean':   round(s.mean(), 2),
            'Median': round(s.median(), 2),
            'Std':    round(s.std(), 2),
            'Min':    round(s.min(), 2),
            'Max':    round(s.max(), 2),
            'Q1':     round(s.quantile(0.25), 2),
            'Q3':     round(s.quantile(0.75), 2),
        })
    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(out_dir, 'stats_summary.csv'), index=False)
    print(f"      Saved -> {os.path.join(out_dir, 'stats_summary.csv')}")

    # ── Summary by stratification category ────────────────────────────────
    print("\n[Stats 2/4] Obesity summary by stratification category")
    strat_rows = []
    for cat, group in obesity.groupby('StratificationCategory1'):
        s = group['Obesity']
        strat_rows.append({
            'Category': cat,
            'N':        len(s),
            'Mean':     round(s.mean(), 2),
            'Std':      round(s.std(), 2),
            'Min':      round(s.min(), 2),
            'Max':      round(s.max(), 2),
        })
    strat_df = pd.DataFrame(strat_rows)
    print(strat_df.to_string(index=False))
    strat_df.to_csv(os.path.join(out_dir, 'stats_obesity_by_strat.csv'), index=False)

    # ── Correlations - Dataset 1: Obesity + Activity (391 rows) ───────────
    print("\n[Stats 3/4] Correlations - Obesity + Activity (391 rows)")
    oa = obesity[INDEX + ['Obesity']].merge(
        activity[INDEX + ['Inactive']], on=INDEX, how='inner'
    ).dropna()
    print(f"      Joined rows: {len(oa):,}")

    r, p = scipy_stats.pearsonr(oa['Obesity'], oa['Inactive'])
    print(f"\n      Obesity vs Inactive (all strata, 391 rows):")
    print(f"        Pearson r = {r:.4f}   p = {p:.6f}   "
          f"{'Significant ***' if p < 0.001 else 'Significant **' if p < 0.01 else 'Significant *' if p < 0.05 else 'Not significant'}")

    print(f"\n      By stratification category:")
    corr_rows = []
    for cat, group in oa.groupby('StratificationCategory1'):
        if len(group) >= 5:
            r2, p2 = scipy_stats.pearsonr(group['Obesity'], group['Inactive'])
            sig = '***' if p2 < 0.001 else '**' if p2 < 0.01 else '*' if p2 < 0.05 else ''
            print(f"        {cat:<25}  r={r2:+.3f}  p={p2:.4f}  n={len(group)}  {sig}")
            corr_rows.append({'Category': cat, 'r': round(r2, 4),
                              'p_value': round(p2, 6), 'n': len(group)})

    # Time trend at Total level
    total_oa = oa[oa['StratificationCategory1'] == 'Total'].sort_values('YearStart')
    if len(total_oa) >= 5:
        r3, p3 = scipy_stats.pearsonr(total_oa['YearStart'], total_oa['Obesity'])
        r4, p4 = scipy_stats.pearsonr(total_oa['YearStart'], total_oa['Inactive'])
        r5, p5 = scipy_stats.pearsonr(total_oa['Obesity'],   total_oa['Inactive'])
        print(f"\n      Total population time trends ({len(total_oa)} years):")
        print(f"        Year vs Obesity:          r={r3:+.4f}  p={p3:.6f}")
        print(f"        Year vs Inactive:         r={r4:+.4f}  p={p4:.6f}")
        print(f"        Obesity vs Inactive:      r={r5:+.4f}  p={p5:.6f}")

    pd.DataFrame(corr_rows).to_csv(
        os.path.join(out_dir, 'stats_corr_obesity_inactive.csv'), index=False)

    # ── Correlations - Dataset 2: All three joined (84 rows) ──────────────
    print("\n[Stats 4/4] Correlations - Obesity + Activity + Diet (84 rows)")
    oad = obesity[INDEX + ['Obesity']].merge(
        activity[INDEX + ['Inactive']], on=INDEX, how='inner'
    ).merge(
        diet[INDEX + ['PoorDiet', 'LowFruit', 'LowVeg']], on=INDEX, how='inner'
    ).dropna()
    print(f"      Joined rows: {len(oad):,}")
    print(f"      Years: {sorted(oad['YearStart'].unique().astype(int).tolist())}")

    pairs = [
        ('Obesity',  'Inactive'),
        ('Obesity',  'PoorDiet'),
        ('Obesity',  'LowFruit'),
        ('Obesity',  'LowVeg'),
        ('Inactive', 'PoorDiet'),
        ('LowFruit', 'LowVeg'),
    ]
    print(f"\n      Pairwise correlations:")
    pair_rows = []
    for col1, col2 in pairs:
        r, p = scipy_stats.pearsonr(oad[col1], oad[col2])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"        {col1:<12} vs {col2:<12}  r={r:+.4f}  p={p:.6f}  {sig}")
        pair_rows.append({'Var1': col1, 'Var2': col2,
                          'r': round(r, 4), 'p_value': round(p, 6)})
    pd.DataFrame(pair_rows).to_csv(
        os.path.join(out_dir, 'stats_corr_combined.csv'), index=False)

    # VIF
    print(f"\n      VIF (Inactive + PoorDiet on {len(oad)} rows):")
    X = oad[['Inactive', 'PoorDiet']].values
    for i, col in enumerate(['Inactive', 'PoorDiet']):
        vif = variance_inflation_factor(X, i)
        print(f"        {col}: VIF = {vif:.2f}")

    # Normality
    print(f"\n      Normality check (Shapiro-Wilk):")
    from scipy.stats import shapiro
    for label, data in [
        ('Obesity',  obesity['Obesity'].dropna()),
        ('Inactive', activity['Inactive'].dropna()),
        ('PoorDiet', diet['PoorDiet'].dropna()),
    ]:
        sample = data.sample(min(500, len(data)), random_state=42)
        w, p = shapiro(sample)
        print(f"        {label:<12}  W={w:.4f}  p={p:.6f}  "
              f"{'Normal (p>0.05)' if p > 0.05 else 'Non-normal (p<0.05)'}")

    print(f"\n      CSVs saved to: {out_dir}/")
    print(f"        stats_summary.csv")
    print(f"        stats_obesity_by_strat.csv")
    print(f"        stats_corr_obesity_inactive.csv")
    print(f"        stats_corr_combined.csv")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='BRFSS EDA pipeline')
    parser.add_argument('--input_dir',  required=False, default='clean_outputs',
                        help='Directory containing cleaned parquet files')
    parser.add_argument('--output_dir', required=False, default='eda_outputs',
                        help='Directory to save EDA charts')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    apply_style()

    print(f"\nBRFSS EDA Pipeline")
    print(f"{'='*60}")
    print(f"Input:  {args.input_dir}/")
    print(f"Output: {args.output_dir}/")

    # Load national datasets
    obesity  = pd.read_parquet(os.path.join(args.input_dir, 'obesity_national.parquet'))
    activity = pd.read_parquet(os.path.join(args.input_dir, 'activity_national.parquet'))
    diet     = pd.read_parquet(os.path.join(args.input_dir, 'diet_national.parquet'))

    print(f"\nLoaded:")
    print(f"  Obesity:  {obesity.shape[0]:,} rows  |  Years: {sorted(obesity['YearStart'].unique().astype(int).tolist())}")
    print(f"  Activity: {activity.shape[0]:,} rows  |  Years: {sorted(activity['YearStart'].unique().astype(int).tolist())}")
    print(f"  Diet:     {diet.shape[0]:,} rows  |  Years: {sorted(diet['YearStart'].unique().astype(int).tolist())}")

    # Run EDA - charts
    eda_obesity(obesity, args.output_dir)
    eda_activity(activity, obesity, args.output_dir)
    eda_diet(diet, obesity, args.output_dir)
    eda_combined(obesity, activity, diet, args.output_dir)

    # Run EDA - statistics and correlations
    compute_summary_stats(obesity, activity, diet, args.output_dir)

    print(f"\n{'='*60}")
    print(f"EDA complete. All outputs saved to: {args.output_dir}/")
    print(f"\nCharts produced:")
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith('.png'):
            print(f"  {f}")
    print(f"\nCSVs produced:")
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith('.csv'):
            print(f"  {f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()