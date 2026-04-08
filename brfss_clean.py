"""
BRFSS Data Cleaning Pipeline
==============================
Cleans the raw BRFSS dataset and splits it into three independent
class-based datasets via pivot.

Study design:
    Dependent variable : Obesity (Dataset 1 — Obesity/Weight Status)
    Predictor 1        : Inactive (Dataset 2 — Physical Activity)
    Predictor 2        : PoorDiet = (LowFruit + LowVeg) / 2 (Dataset 3 — Fruits & Vegetables)
    Unit of analysis   : National weighted average per Year x Stratification group
    Years              : 2011-2024

Pipeline:
    1. Load raw data
    2. Validate
    3. Drop redundant/constant/surrogate columns
    4. Filter Sample_Size >= 50
    5. Split into 3 class datasets
    6. Pivot each class independently
    7. Aggregate each to national level (weighted by Sample_Size)
    8. Save all datasets

Usage:
    python brfss_clean.py --input brfss_raw.csv --output_dir clean_outputs/

Requirements:
    pip install pandas pyarrow
"""

import argparse
import os
import sys
import pandas as pd
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────────────

COLUMNS_TO_DROP = {
    'YearEnd':                    'Always equal to YearStart; single-year survey cycles',
    'LocationAbbr':               'Redundant with LocationDesc',
    'Datasource':                 'Constant value ("BRFSS") across all rows',
    'Topic':                      'Duplicate of Class',
    'Data_Value_Unit':            'Constant value (%) across all rows',
    'Data_Value_Type':            'Constant value ("Value") across all rows',
    'Data_Value_Alt':             'Numeric duplicate of Data_Value',
    'Data_Value_Footnote_Symbol': 'Sparse metadata; handled by Sample_Size filter',
    'Data_Value_Footnote':        'Sparse metadata; handled by Sample_Size filter',
    'Total':                      'Mostly null; duplicates Sample_Size where populated',
    'Low_Confidence_Limit':       'Reliability addressed by Sample_Size filter',
    'High_Confidence_Limit':      'Reliability addressed by Sample_Size filter',
    'GeoLocation':                'State-level analysis uses LocationDesc',
    'Age(years)':                 'Captured in Stratification1',
    'Education':                  'Captured in Stratification1',
    'Sex':                        'Captured in Stratification1',
    'Income':                     'Captured in Stratification1',
    'Race/Ethnicity':             'Captured in Stratification1',
    'ClassID':                    'Encoded surrogate for Class',
    'TopicID':                    'Encoded surrogate for Topic',
    'QuestionID':                 'Encoded surrogate for Question',
    'DataValueTypeID':            'Encoded surrogate for Data_Value_Type',
    'LocationID':                 'Encoded surrogate for LocationDesc',
    'StratificationCategoryID1':  'Encoded surrogate for StratificationCategory1',
    'StratificationID1':          'Encoded surrogate for Stratification1',
}

# Questions to keep per class
OBESITY_QUESTIONS = [
    'Percent of adults aged 18 years and older who have obesity',
]

ACTIVITY_QUESTIONS = [
    'Percent of adults who engage in no leisure-time physical activity',
]

DIET_QUESTIONS = [
    'Percent of adults who report consuming fruit less than one time daily',
    'Percent of adults who report consuming vegetables less than one time daily',
]

# Short column names after pivoting
OBESITY_RENAME = {
    'Percent of adults aged 18 years and older who have obesity': 'Obesity',
}
ACTIVITY_RENAME = {
    'Percent of adults who engage in no leisure-time physical activity': 'Inactive',
}
DIET_RENAME = {
    'Percent of adults who report consuming fruit less than one time daily':      'LowFruit',
    'Percent of adults who report consuming vegetables less than one time daily': 'LowVeg',
}

MIN_SAMPLE_SIZE = 50
PIVOT_INDEX     = ['YearStart', 'LocationDesc', 'StratificationCategory1', 'Stratification1']
NATIONAL_INDEX  = ['YearStart', 'StratificationCategory1', 'Stratification1']


# ── Pipeline steps ─────────────────────────────────────────────────────────────

def load(path: str) -> pd.DataFrame:
    print(f"[1/8] Loading data from: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"      Raw shape:  {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"      Classes:    {df['Class'].unique().tolist()}")
    print(f"      Locations:  {df['LocationDesc'].nunique()} unique")
    print(f"      Years:      {sorted(df['YearStart'].dropna().unique().astype(int).tolist())}")
    return df


def validate(df: pd.DataFrame) -> None:
    print("\n[2/8] Validating...")
    errors = []
    if not df['Data_Value'].dropna().between(0, 100).all():
        errors.append("Data_Value contains values outside [0, 100]")
    if (df['Sample_Size'].dropna() < 0).any():
        errors.append("Sample_Size contains negative values")
    if not df['YearStart'].dropna().between(2000, 2030).all():
        errors.append("YearStart contains unexpected values")
    if 'Class' not in df.columns:
        errors.append("Class column not found — cannot split by class")
    if errors:
        for e in errors:
            print(f"      ERROR: {e}")
        sys.exit("Validation failed.")
    print("      All checks passed.")


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[3/8] Dropping redundant columns...")
    present = [c for c in COLUMNS_TO_DROP if c in df.columns]
    missing = [c for c in COLUMNS_TO_DROP if c not in df.columns]
    if missing:
        print(f"      Note: {len(missing)} expected columns not found: {missing}")
    df = df.drop(columns=present)
    print(f"      Dropped {len(present)} columns. Remaining: {df.shape[1]}")
    print(f"      Columns kept: {list(df.columns)}")
    return df


def filter_sample_size(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[4/8] Filtering Sample_Size < {MIN_SAMPLE_SIZE}...")
    n = len(df)
    df = df.loc[df['Sample_Size'] >= MIN_SAMPLE_SIZE].copy()
    print(f"      Removed {n - len(df):,} records. Remaining: {len(df):,} rows")
    return df.reset_index(drop=True)


def split_by_class(df: pd.DataFrame) -> tuple:
    """Split the cleaned dataset into three class subsets."""
    print("\n[5/8] Splitting by class...")

    obesity_raw  = df[df['Class'].str.contains('Obesity',           case=False, na=False)].copy()
    activity_raw = df[df['Class'].str.contains('Physical Activity',  case=False, na=False)].copy()
    diet_raw     = df[df['Class'].str.contains('Fruits',             case=False, na=False)].copy()

    print(f"      Obesity class:           {len(obesity_raw):,} rows")
    print(f"      Physical Activity class: {len(activity_raw):,} rows")
    print(f"      Fruits & Veg class:      {len(diet_raw):,} rows")
    print(f"      Total:                   {len(obesity_raw) + len(activity_raw) + len(diet_raw):,} rows")

    return obesity_raw, activity_raw, diet_raw


def pivot_class(df: pd.DataFrame, questions: list,
                rename: dict, label: str) -> pd.DataFrame:
    """Filter to target questions and pivot a single class dataset."""
    print(f"\n[6/8] Pivoting {label}...")

    # Filter to questions we want
    df = df[df['Question'].isin(questions)].copy()
    print(f"      After question filter: {len(df):,} rows")

    if df.empty:
        print(f"      WARNING: No rows found for {label}")
        return pd.DataFrame()

    # Sample_Size aggregation:
    # For classes with ONE question per index (Obesity, Activity) — sum == mean == the value.
    # For classes with MULTIPLE questions per index (Diet — LowFruit and LowVeg) —
    # the same respondents answered both questions, so summing would double-count.
    # We use MEAN which gives the correct per-respondent sample size regardless
    # of how many questions share the same index.
    ss = (df.groupby(PIVOT_INDEX)['Sample_Size']
            .mean()
            .round(0)
            .astype(int)
            .reset_index())

    # Pivot
    pivoted = df.pivot_table(
        index=PIVOT_INDEX,
        columns='Question',
        values='Data_Value',
        aggfunc='mean',
    ).reset_index()
    pivoted.columns.name = None
    pivoted = pivoted.rename(columns=rename)

    # Merge Sample_Size back
    pivoted = pivoted.merge(ss, on=PIVOT_INDEX, how='left')

    print(f"      Pivoted shape: {pivoted.shape[0]:,} rows x {pivoted.shape[1]} columns")
    print(f"      Columns: {list(pivoted.columns)}")
    return pivoted


def aggregate_national(df: pd.DataFrame,
                        value_cols: list, label: str) -> pd.DataFrame:
    """
    Aggregate state-level records to a national weighted average
    per Year x Stratification group.
    Weight = Sample_Size so larger state surveys count more.
    """
    print(f"\n[7/8] Aggregating {label} to national level...")

    if df.empty:
        return df

    agg_rows = []
    for keys, group in df.groupby(NATIONAL_INDEX):
        row = dict(zip(NATIONAL_INDEX, keys if isinstance(keys, tuple) else [keys]))
        row['N_States']          = group['LocationDesc'].nunique()
        row['Total_Sample_Size'] = group['Sample_Size'].sum()

        for col in value_cols:
            if col in group.columns:
                valid = group[['Sample_Size', col]].dropna()
                if not valid.empty:
                    row[col] = (
                        (valid[col] * valid['Sample_Size']).sum()
                        / valid['Sample_Size'].sum()
                    )
                else:
                    row[col] = None
        agg_rows.append(row)

    result = pd.DataFrame(agg_rows).sort_values(NATIONAL_INDEX).reset_index(drop=True)
    print(f"      National shape: {result.shape[0]:,} rows x {result.shape[1]} columns")
    print(f"      Years covered:  {sorted(result['YearStart'].unique().astype(int).tolist())}")
    print(f"      Strata:         {result['StratificationCategory1'].unique().tolist()}")
    return result


def save(df: pd.DataFrame, path: str, label: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"      Saved {label}: {path}  ({df.shape[0]:,} rows x {df.shape[1]} cols)")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='BRFSS cleaning pipeline')
    parser.add_argument('--input',      required=True,  help='Path to raw CSV')
    parser.add_argument('--output_dir', required=False, default='clean_outputs',
                        help='Directory to save output parquet files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nBRFSS Cleaning Pipeline")
    print(f"{'='*60}")

    # Steps 1-4: load, validate, drop columns, filter sample size
    df = load(args.input)
    validate(df)
    df = drop_columns(df)
    df = filter_sample_size(df)

    # Step 5: split by class
    obesity_raw, activity_raw, diet_raw = split_by_class(df)

    # Step 6: pivot each class independently
    obesity_state  = pivot_class(obesity_raw,  OBESITY_QUESTIONS,  OBESITY_RENAME,  'Obesity')
    activity_state = pivot_class(activity_raw, ACTIVITY_QUESTIONS, ACTIVITY_RENAME, 'Physical Activity')
    diet_state     = pivot_class(diet_raw,     DIET_QUESTIONS,     DIET_RENAME,     'Fruits & Vegetables')

    # Step 7: aggregate each to national level
    obesity_nat  = aggregate_national(obesity_state,  ['Obesity'],           'Obesity')
    activity_nat = aggregate_national(activity_state, ['Inactive'],          'Physical Activity')
    diet_nat     = aggregate_national(diet_state,     ['LowFruit', 'LowVeg'],'Fruits & Vegetables')

    # Compute PoorDiet score
    diet_nat['PoorDiet'] = diet_nat[['LowFruit', 'LowVeg']].mean(axis=1)

    # Step 8: save all datasets
    print(f"\n[8/8] Saving datasets...")
    save(obesity_state,  os.path.join(args.output_dir, 'obesity_state.parquet'),    'Obesity (state level)')
    save(activity_state, os.path.join(args.output_dir, 'activity_state.parquet'),   'Activity (state level)')
    save(diet_state,     os.path.join(args.output_dir, 'diet_state.parquet'),       'Diet (state level)')
    save(obesity_nat,    os.path.join(args.output_dir, 'obesity_national.parquet'), 'Obesity (national)')
    save(activity_nat,   os.path.join(args.output_dir, 'activity_national.parquet'),'Activity (national)')
    save(diet_nat,       os.path.join(args.output_dir, 'diet_national.parquet'),    'Diet (national)')

    print(f"\n{'='*60}")
    print(f"Pipeline complete. Outputs saved to: {args.output_dir}/")
    print(f"\nState-level datasets  — use for geographic and demographic analysis")
    print(f"National datasets     — use for trend analysis and modeling")
    print(f"  Obesity ~ Inactive + PoorDiet")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()