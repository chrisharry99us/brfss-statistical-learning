"""
BRFSS Regression Modeling Pipeline
=====================================
Runs 3 regression models:

Model 1: Obesity ~ Inactive                (OLS, 391 rows, all strata)
Model 2: Obesity ~ PoorDiet               (OLS, 84 rows, all strata)
Model 3: Obesity ~ Inactive + PoorDiet    (Ridge, 84 rows, all strata)

Usage:
    python brfss_model.py --input_dir clean_outputs/ --output_dir model_outputs/

Requirements:
    pip install pandas numpy scikit-learn statsmodels matplotlib seaborn pyarrow
"""

import argparse
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings('ignore')

INDEX = ['YearStart', 'StratificationCategory1', 'Stratification1']
MAIN_BLUE = '#2E75B6'
DARK_BLUE = '#1F3864'
RED       = '#C00000'
ORANGE    = '#ED7D31'
GREEN     = '#70AD47'


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


def total_only(df):
    return df[df['StratificationCategory1'] == 'Total'].copy()


def ols_summary(name, X, y, feature_names):
    """Fit OLS with statsmodels and print clean summary."""
    X_const = sm.add_constant(X)
    model   = sm.OLS(y, X_const).fit()

    print(f"\n  {'-'*50}")
    print(f"  {name}")
    print(f"  {'-'*50}")
    print(f"  n = {len(y)}  |  R^2 = {model.rsquared:.4f}  |  "
          f"Adj R^2 = {model.rsquared_adj:.4f}  |  "
          f"F-stat p = {model.f_pvalue:.6f}")
    print(f"\n  Coefficients:")
    print(f"  {'Feature':<20} {'Coef':>10} {'Std Err':>10} {'t':>8} {'p':>10} {'Sig':>5}")
    print(f"  {'-'*65}")
    for i, fname in enumerate(['Intercept'] + list(feature_names)):
        b   = model.params[i]
        se  = model.bse[i]
        t   = model.tvalues[i]
        p   = model.pvalues[i]
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {fname:<20} {b:>10.4f} {se:>10.4f} {t:>8.3f} {p:>10.6f} {sig:>5}")

    return model


def residual_plots(model, name, out_dir):
    """Fitted vs residuals and Q-Q plot."""
    fitted    = model.fittedvalues
    residuals = model.resid

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Fitted vs Residuals
    ax = axes[0]
    ax.scatter(fitted, residuals, color=MAIN_BLUE, alpha=0.5, s=20, edgecolors='white')
    ax.axhline(0, color=RED, linestyle='--', linewidth=1.2)
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Fitted', fontweight='bold')

    # Q-Q plot
    ax = axes[1]
    sm.qqplot(residuals, line='s', ax=ax, alpha=0.5,
              markerfacecolor=MAIN_BLUE, markeredgewidth=0.3)
    ax.set_title('Normal Q-Q Plot', fontweight='bold')

    fig.suptitle(f'Residual Diagnostics - {name}',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    fname = name.lower().replace(' ', '_').replace('~', 'by').replace('+', 'and')
    save(fig, os.path.join(out_dir, f'residuals_{fname}.png'))


# ── Model 1 - Obesity ~ Inactive ─────────────────────────────────────────────

def model1_inactive(obesity, activity, out_dir):
    print(f"\n{'='*60}")
    print("MODEL 1 - Obesity ~ Inactive (Simple OLS)")
    print("n=391, all strata, 2011-2024")
    print(f"{'='*60}")

    df = obesity[INDEX + ['Obesity']].merge(
        activity[INDEX + ['Inactive']], on=INDEX, how='inner'
    ).dropna()

    X = df[['Inactive']].values
    y = df['Obesity'].values

    model = ols_summary('Obesity ~ Inactive', X, y, ['Inactive'])
    residual_plots(model, 'Model 1 Obesity~Inactive', out_dir)

    # Scatter with fit line
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(df['Inactive'], df['Obesity'],
               color=MAIN_BLUE, alpha=0.4, s=15, edgecolors='white')
    x_range = np.linspace(df['Inactive'].min(), df['Inactive'].max(), 100)
    y_pred  = model.params[0] + model.params[1] * x_range
    ax.plot(x_range, y_pred, color=RED, linewidth=2,
            label=f'b={model.params[1]:.3f}, R^2={model.rsquared:.3f}')
    ax.set_title('Model 1 - Obesity ~ Physical Inactivity (all strata)',
                 fontweight='bold')
    ax.set_xlabel('Physical Inactivity (%)')
    ax.set_ylabel('Obesity (%)')
    ax.legend(frameon=False)
    save(fig, os.path.join(out_dir, 'model1_inactive.png'))

    return model


# ── Model 2 - Obesity ~ PoorDiet ─────────────────────────────────────────────

def model2_diet(obesity, diet, out_dir):
    print(f"\n{'='*60}")
    print("MODEL 2 - Obesity ~ PoorDiet (Simple OLS)")
    print("n=84, all strata, 2017/2019/2021")
    print(f"{'='*60}")

    df = obesity[INDEX + ['Obesity']].merge(
        diet[INDEX + ['PoorDiet']], on=INDEX, how='inner'
    ).dropna()

    X = df[['PoorDiet']].values
    y = df['Obesity'].values

    model = ols_summary('Obesity ~ PoorDiet', X, y, ['PoorDiet'])
    residual_plots(model, 'Model 2 Obesity~PoorDiet', out_dir)

    # Scatter with fit
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(df['PoorDiet'], df['Obesity'],
               color=ORANGE, alpha=0.5, s=20, edgecolors='white')
    x_range = np.linspace(df['PoorDiet'].min(), df['PoorDiet'].max(), 100)
    y_pred  = model.params[0] + model.params[1] * x_range
    ax.plot(x_range, y_pred, color=RED, linewidth=2,
            label=f'b={model.params[1]:.3f}, R^2={model.rsquared:.3f}')
    ax.set_title('Model 2 - Obesity ~ PoorDiet (all strata, diet years)',
                 fontweight='bold')
    ax.set_xlabel('PoorDiet Score (%)')
    ax.set_ylabel('Obesity (%)')
    ax.legend(frameon=False)
    save(fig, os.path.join(out_dir, 'model2_diet.png'))

    return model


# ── Model 3 - Obesity ~ Inactive + PoorDiet (Ridge) ──────────────────────────

def model3_ridge(obesity, activity, diet, out_dir):
    print(f"\n{'='*60}")
    print("MODEL 3 - Obesity ~ Inactive + PoorDiet (Ridge Regression)")
    print("n=84, all strata, 2017/2019/2021")
    print("Ridge used due to VIF=28.15 between predictors")
    print(f"{'='*60}")

    df = obesity[INDEX + ['Obesity']].merge(
        activity[INDEX + ['Inactive']], on=INDEX, how='inner'
    ).merge(
        diet[INDEX + ['PoorDiet']], on=INDEX, how='inner'
    ).dropna()

    X_raw = df[['Inactive', 'PoorDiet']].values
    y     = df['Obesity'].values

    # Standardize predictors
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Select best alpha via cross-validation
    alphas   = np.logspace(-3, 3, 100)
    ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='r2')
    ridge_cv.fit(X_scaled, y)
    best_alpha = ridge_cv.alpha_

    print(f"\n  Best alpha (CV): {best_alpha:.4f}")

    # Fit final Ridge model
    ridge = Ridge(alpha=best_alpha)
    ridge.fit(X_scaled, y)
    y_pred = ridge.predict(X_scaled)
    r2     = r2_score(y, y_pred)
    rmse   = np.sqrt(mean_squared_error(y, y_pred))

    # Cross-validated R^2
    kf      = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2   = cross_val_score(Ridge(alpha=best_alpha), X_scaled, y,
                               cv=kf, scoring='r2')

    print(f"\n  Results:")
    print(f"  {'Feature':<15} {'Std Coef':>12}")
    print(f"  {'-'*30}")
    for name, coef in zip(['Inactive', 'PoorDiet'], ridge.coef_):
        print(f"  {name:<15} {coef:>12.4f}")
    print(f"\n  Intercept:      {ridge.intercept_:.4f}")
    print(f"  R^2 (train):     {r2:.4f}")
    print(f"  RMSE:           {rmse:.4f}")
    print(f"  CV R^2 (5-fold): {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")

    # Compare with OLS single predictors
    ols_inactive = sm.OLS(y, sm.add_constant(X_raw[:, 0])).fit()
    ols_diet     = sm.OLS(y, sm.add_constant(X_raw[:, 1])).fit()
    print(f"\n  Comparison (R^2):")
    print(f"    OLS Inactive only:  {ols_inactive.rsquared:.4f}")
    print(f"    OLS PoorDiet only:  {ols_diet.rsquared:.4f}")
    print(f"    Ridge combined:     {r2:.4f}")

    # Coefficient chart
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = [MAIN_BLUE, ORANGE]
    bars = ax.barh(['Inactive', 'PoorDiet'], ridge.coef_,
                   color=colors, edgecolor='white')
    ax.axvline(0, color='black', linewidth=0.8)
    for bar, val in zip(bars, ridge.coef_):
        ax.text(val + (0.01 if val >= 0 else -0.01),
                bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center',
                ha='left' if val >= 0 else 'right', fontsize=10)
    ax.set_xlabel('Standardized coefficient')
    ax.set_title(f'Model 3 - Ridge Coefficients\n'
                 f'(alpha={best_alpha:.3f}, R^2={r2:.3f}, CV R^2={cv_r2.mean():.3f})',
                 fontweight='bold')
    save(fig, os.path.join(out_dir, 'model3_ridge_coefficients.png'))

    # Predicted vs Actual
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y, y_pred, color=MAIN_BLUE, alpha=0.5, s=20, edgecolors='white')
    ax.plot([y.min(), y.max()], [y.min(), y.max()],
            color=RED, linestyle='--', linewidth=1.5, label='Perfect fit')
    ax.set_xlabel('Actual Obesity (%)')
    ax.set_ylabel('Predicted Obesity (%)')
    ax.set_title(f'Model 3 - Predicted vs Actual (Ridge, R^2={r2:.3f})',
                 fontweight='bold')
    ax.legend(frameon=False)
    save(fig, os.path.join(out_dir, 'model3_predicted_vs_actual.png'))

    return ridge, best_alpha, r2, cv_r2


# ── Model summary table ───────────────────────────────────────────────────────

def print_model_summary(m1, m2, ridge_r2, cv_r2):
    print(f"\n{'='*60}")
    print("MODEL SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'Model':<45} {'n':>5} {'R^2':>8} {'Key finding'}")
    print(f"  {'-'*90}")
    print(f"  {'M1: Obesity ~ Inactive (all strata)':<45} {'391':>5} {m1.rsquared:>8.4f} "
          f"  b={m1.params[1]:.3f}, p={m1.pvalues[1]:.6f}")
    print(f"  {'M2: Obesity ~ PoorDiet':<45} {'84':>5} {m2.rsquared:>8.4f} "
          f"  b={m2.params[1]:.3f}, p={m2.pvalues[1]:.6f}")
    print(f"  {'M3: Ridge (Inactive + PoorDiet)':<45} {'84':>5} {ridge_r2:>8.4f} "
          f"  CV R^2={cv_r2.mean():.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='BRFSS regression modeling')
    parser.add_argument('--input_dir',  default='clean_outputs')
    parser.add_argument('--output_dir', default='model_outputs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    apply_style()

    print(f"\nBRFSS Regression Modeling Pipeline")
    print(f"{'='*60}")

    # Load
    obesity  = pd.read_parquet(os.path.join(args.input_dir, 'obesity_national.parquet'))
    activity = pd.read_parquet(os.path.join(args.input_dir, 'activity_national.parquet'))
    diet     = pd.read_parquet(os.path.join(args.input_dir, 'diet_national.parquet'))

    # Run models
    m1 = model1_inactive(obesity, activity, args.output_dir)
    m2 = model2_diet(obesity, diet, args.output_dir)
    ridge, alpha, r2, cv_r2 = model3_ridge(obesity, activity, diet, args.output_dir)

    # Summary
    print_model_summary(m1, m2, r2, cv_r2)

    print(f"\n{'='*60}")
    print(f"Modeling complete. Outputs saved to: {args.output_dir}/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()