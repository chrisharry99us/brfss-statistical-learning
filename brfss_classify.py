"""
BRFSS Classification Pipeline
================================
Classifies obesity risk (High vs Low) based on behavioral indicators
and demographic stratification category.

Binary outcome:
    High risk (1) : Obesity > median
    Low risk  (0) : Obesity <= median

Two datasets:
    Dataset A: Inactive + StratCat1          (391 rows, 2011-2024)
    Dataset B: Inactive + PoorDiet + StratCat1 (84 rows, 2017/2019/2021)

Four classifiers per dataset:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Linear SVM

Evaluation:
    - Accuracy, Precision, Recall, F1
    - Confusion matrix
    - ROC curve and AUC
    - 5-fold cross-validation
    - Feature importance (Random Forest)

Usage:
    python brfss_classify.py --input_dir clean_outputs/ --output_dir classify_outputs/

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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_curve, auc)

warnings.filterwarnings('ignore')

INDEX    = ['YearStart', 'StratificationCategory1', 'Stratification1']
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


def build_dataset(obesity, activity, diet=None):
    """
    Join datasets, create binary outcome, one-hot encode StratCat1.
    Returns X (features), y (binary target), feature_names.
    """
    # Join
    df = obesity[INDEX + ['Obesity']].merge(
        activity[INDEX + ['Inactive']], on=INDEX, how='inner'
    )
    if diet is not None:
        df = df.merge(diet[INDEX + ['PoorDiet']], on=INDEX, how='inner')
    df = df.dropna().reset_index(drop=True)

    # Binary outcome: High(1) vs Low(0) based on dataset median obesity
    median_obs = df['Obesity'].median()
    df['HighRisk'] = (df['Obesity'] > median_obs).astype(int)
    print(f"      Median obesity: {median_obs:.2f}%")
    print(f"      High risk: {df['HighRisk'].sum()} ({df['HighRisk'].mean():.1%})  "
          f"Low risk: {(1-df['HighRisk']).sum()} ({(1-df['HighRisk']).mean():.1%})")

    # One-hot encode StratificationCategory1
    strat_dummies = pd.get_dummies(
        df['StratificationCategory1'], prefix='Strat', drop_first=True
    )

    # Build feature matrix
    behavioral_cols = ['Inactive'] + (['PoorDiet'] if diet is not None else [])
    X = pd.concat([df[behavioral_cols], strat_dummies], axis=1)
    y = df['HighRisk'].values
    feature_names = list(X.columns)

    print(f"      Features: {feature_names}")
    print(f"      X shape: {X.shape}")
    return X.values, y, feature_names, median_obs


def evaluate_classifier(name, clf, X_scaled, y, feature_names, cv, out_dir, prefix):
    """
    Fit, cross-validate, and evaluate a classifier.
    Returns dict of metrics.
    """
    # Cross-validated scores
    cv_acc  = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
    cv_f1   = cross_val_score(clf, X_scaled, y, cv=cv, scoring='f1')
    cv_auc  = cross_val_score(clf, X_scaled, y, cv=cv, scoring='roc_auc')

    # Fit on full data for confusion matrix and feature importance
    clf.fit(X_scaled, y)
    y_pred = clf.predict(X_scaled)

    acc  = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec  = recall_score(y, y_pred, zero_division=0)
    f1   = f1_score(y, y_pred, zero_division=0)
    cm   = confusion_matrix(y, y_pred)

    print(f"\n  {name}")
    print(f"    CV Accuracy:  {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")
    print(f"    CV F1:        {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")
    print(f"    CV AUC:       {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}")
    print(f"    Train Acc:    {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}")

    return {
        'name':       name,
        'cv_acc':     cv_acc.mean(),
        'cv_acc_std': cv_acc.std(),
        'cv_f1':      cv_f1.mean(),
        'cv_f1_std':  cv_f1.std(),
        'cv_auc':     cv_auc.mean(),
        'cv_auc_std': cv_auc.std(),
        'train_acc':  acc,
        'precision':  prec,
        'recall':     rec,
        'f1':         f1,
        'cm':         cm,
        'clf':        clf,
    }


def plot_confusion_matrices(results, out_dir, prefix):
    """Plot all confusion matrices side by side."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        cm = res['cm']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Low', 'High'],
                    yticklabels=['Low', 'High'],
                    ax=ax, cbar=False, linewidths=0.5)
        ax.set_title(f"{res['name']}\nAcc={res['train_acc']:.3f}  F1={res['f1']:.3f}",
                     fontweight='bold', fontsize=11)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    fig.suptitle(f'Confusion Matrices - {prefix}',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save(fig, os.path.join(out_dir, f'{prefix}_confusion_matrices.png'))


def plot_roc_curves(results, X_scaled, y, out_dir, prefix):
    """Plot ROC curves for all classifiers."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = [MAIN_BLUE, DARK_BLUE, GREEN, ORANGE]

    for res, color in zip(results, colors):
        clf = res['clf']
        # Get probability or decision function
        if hasattr(clf, 'predict_proba'):
            scores = clf.predict_proba(X_scaled)[:, 1]
        elif hasattr(clf, 'decision_function'):
            scores = clf.decision_function(X_scaled)
        else:
            continue
        fpr, tpr, _ = roc_curve(y, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{res['name']} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color='grey', linestyle='--',
            linewidth=1, label='Random (AUC=0.500)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves - {prefix}', fontweight='bold')
    ax.legend(frameon=False, fontsize=9)
    save(fig, os.path.join(out_dir, f'{prefix}_roc_curves.png'))


def plot_cv_comparison(results, out_dir, prefix):
    """Bar chart comparing CV metrics across classifiers."""
    names    = [r['name'] for r in results]
    cv_acc   = [r['cv_acc'] for r in results]
    cv_f1    = [r['cv_f1']  for r in results]
    cv_auc   = [r['cv_auc'] for r in results]
    acc_std  = [r['cv_acc_std'] for r in results]
    f1_std   = [r['cv_f1_std']  for r in results]
    auc_std  = [r['cv_auc_std'] for r in results]

    x = np.arange(len(names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 4.5))
    b1 = ax.bar(x - w, cv_acc, w, label='CV Accuracy', color=MAIN_BLUE,
                yerr=acc_std, capsize=3, edgecolor='white')
    b2 = ax.bar(x,     cv_f1,  w, label='CV F1',       color=DARK_BLUE,
                yerr=f1_std,  capsize=3, edgecolor='white')
    b3 = ax.bar(x + w, cv_auc, w, label='CV AUC',      color=GREEN,
                yerr=auc_std, capsize=3, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title(f'Cross-Validated Performance - {prefix}', fontweight='bold')
    ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.2f}'))
    save(fig, os.path.join(out_dir, f'{prefix}_cv_comparison.png'))


def plot_feature_importance(rf_result, feature_names, out_dir, prefix):
    """Plot Random Forest feature importance."""
    rf = rf_result['clf']
    importances = rf.feature_importances_
    idx = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(feature_names) * 0.5)))
    colors = [GREEN if importances[i] == importances.max() else MAIN_BLUE
              for i in idx]
    bars = ax.barh([feature_names[i] for i in idx],
                   [importances[i] for i in idx],
                   color=colors, edgecolor='white')
    for bar, val in zip(bars, [importances[i] for i in idx]):
        ax.text(bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9)
    ax.set_xlabel('Feature Importance (Gini)')
    ax.set_title(f'Random Forest Feature Importance - {prefix}',
                 fontweight='bold')
    save(fig, os.path.join(out_dir, f'{prefix}_feature_importance.png'))


def plot_decision_tree(dt_result, feature_names, out_dir, prefix):
    """Plot Decision Tree structure (limited depth for readability)."""
    dt = dt_result['clf']
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_tree(dt, feature_names=feature_names,
              class_names=['Low Risk', 'High Risk'],
              filled=True, rounded=True, max_depth=3,
              fontsize=8, ax=ax)
    ax.set_title(f'Decision Tree Structure (max depth shown: 3) - {prefix}',
                 fontweight='bold')
    save(fig, os.path.join(out_dir, f'{prefix}_decision_tree.png'))


def run_dataset(label, X, y, feature_names, out_dir):
    """Run all four classifiers on one dataset."""
    print(f"\n{'='*60}")
    print(f"DATASET: {label}")
    print(f"{'='*60}")

    # Scale features
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    classifiers = [
        ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
        ('Decision Tree',       DecisionTreeClassifier(max_depth=5, random_state=42)),
        ('Random Forest',       RandomForestClassifier(n_estimators=100, random_state=42)),
        ('Linear SVM',          LinearSVC(max_iter=2000, random_state=42)),
    ]

    results = []
    for name, clf in classifiers:
        res = evaluate_classifier(name, clf, X_scaled, y,
                                  feature_names, cv, out_dir, label)
        results.append(res)

    prefix = label.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'and')

    # Plots
    plot_confusion_matrices(results, out_dir, prefix)
    plot_roc_curves(results, X_scaled, y, out_dir, prefix)
    plot_cv_comparison(results, out_dir, prefix)

    # Random Forest feature importance
    rf_res = next(r for r in results if r['name'] == 'Random Forest')
    plot_feature_importance(rf_res, feature_names, out_dir, prefix)

    # Decision Tree structure
    dt_res = next(r for r in results if r['name'] == 'Decision Tree')
    plot_decision_tree(dt_res, feature_names, out_dir, prefix)

    # Save results CSV
    summary = pd.DataFrame([{
        'Classifier':    r['name'],
        'CV_Accuracy':   round(r['cv_acc'], 4),
        'CV_Acc_Std':    round(r['cv_acc_std'], 4),
        'CV_F1':         round(r['cv_f1'], 4),
        'CV_F1_Std':     round(r['cv_f1_std'], 4),
        'CV_AUC':        round(r['cv_auc'], 4),
        'CV_AUC_Std':    round(r['cv_auc_std'], 4),
        'Train_Acc':     round(r['train_acc'], 4),
        'Precision':     round(r['precision'], 4),
        'Recall':        round(r['recall'], 4),
        'F1':            round(r['f1'], 4),
    } for r in results])

    csv_path = os.path.join(out_dir, f'{prefix}_results.csv')
    summary.to_csv(csv_path, index=False)
    print(f"\n  Results saved -> {csv_path}")
    print(f"\n  Summary:")
    print(summary.to_string(index=False))

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='BRFSS classification pipeline')
    parser.add_argument('--input_dir',  default='clean_outputs')
    parser.add_argument('--output_dir', default='classify_outputs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    apply_style()

    print(f"\nBRFSS Classification Pipeline")
    print(f"{'='*60}")

    # Load
    obesity  = pd.read_parquet(os.path.join(args.input_dir, 'obesity_national.parquet'))
    activity = pd.read_parquet(os.path.join(args.input_dir, 'activity_national.parquet'))
    diet     = pd.read_parquet(os.path.join(args.input_dir, 'diet_national.parquet'))

    # ── Dataset A: Inactive + StratCat1 (391 rows) ────────────────────────────
    print(f"\nBuilding Dataset A (Inactive + StratCat1, 391 rows)...")
    X_a, y_a, feat_a, med_a = build_dataset(obesity, activity, diet=None)
    results_a = run_dataset('Dataset A (Inactive + StratCat1)', X_a, y_a, feat_a, args.output_dir)

    # ── Dataset B: Inactive + PoorDiet + StratCat1 (84 rows) ─────────────────
    print(f"\nBuilding Dataset B (Inactive + PoorDiet + StratCat1, 84 rows)...")
    X_b, y_b, feat_b, med_b = build_dataset(obesity, activity, diet=diet)
    results_b = run_dataset('Dataset B (Inactive + PoorDiet + StratCat1)', X_b, y_b, feat_b, args.output_dir)

    print(f"\n{'='*60}")
    print(f"Classification complete. Outputs saved to: {args.output_dir}/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()