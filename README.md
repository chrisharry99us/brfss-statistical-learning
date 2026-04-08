# BRFSS Statistical Learning Analysis

**Can national physical activity levels and dietary habits predict obesity rates across U.S. demographic groups from 2011 to 2024?**

Illinois Institute of Technology · MAS Data Science · Statistical Learning

---

## Overview

This project applies regression, classification, and unsupervised clustering to 14 years of CDC Behavioral Risk Factor Surveillance System (BRFSS) data to investigate the relationship between behavioral indicators and obesity prevalence across U.S. demographic groups.

> **Important:** Every value in this dataset is a **population-level percentage** — the proportion of adults in a given state, year, and demographic group who meet a criterion. This is an ecological analysis, not individual-level data.

---

## Research Question

> Can national physical activity levels and dietary habits predict obesity rates across U.S. demographic groups from 2011 to 2024?

---

## Dataset

| Attribute | Details |
|-----------|---------|
| Source | CDC Behavioral Risk Factor Surveillance System (BRFSS) |
| Period | 2011–2024 (14 annual survey cycles) |
| Raw size | 110,880 rows × 33 columns |
| Geographic coverage | 55 locations (50 states + D.C., Puerto Rico, Virgin Islands, Guam, National) |
| Unit of analysis | National weighted average per Year × Demographic group |
| Key variables | Obesity (%), Inactive (%), PoorDiet = (LowFruit + LowVeg) / 2 |

**Download:** [data.gov — Nutrition, Physical Activity and Obesity](https://catalog.data.gov/dataset/nutrition-physical-activity-and-obesity-behavioral-risk-factor-surveillance-system)

Save as `brfss_raw.csv` in the project root, then run the pipeline in order.

---

## Project Structure

```
brfss_clean.py          # ETL pipeline → clean_outputs/
brfss_eda.py            # Exploratory data analysis → eda_outputs/
brfss_model.py          # Regression models → model_outputs/
brfss_classify.py       # Classification → classify_outputs/
brfss_cluster.py        # K-Means clustering → cluster_outputs/
app.py                  # Streamlit interactive dashboard
requirements.txt        # Python dependencies
clean_outputs/          # Cleaned parquet files (generated)
eda_outputs/            # EDA charts (generated)
model_outputs/          # Regression charts (generated)
classify_outputs/       # Classification charts and results (generated)
cluster_outputs/        # Cluster charts and profiles (generated)
```

---

## Pipeline

Run scripts in order after downloading `brfss_raw.csv`:

```bash
pip install -r requirements.txt

python brfss_clean.py    --input brfss_raw.csv --output_dir clean_outputs/
python brfss_eda.py      --input_dir clean_outputs/ --output_dir eda_outputs/
python brfss_model.py    --input_dir clean_outputs/ --output_dir model_outputs/
python brfss_classify.py --input_dir clean_outputs/ --output_dir classify_outputs/
python brfss_cluster.py  --input_dir clean_outputs/ --output_dir cluster_outputs/

streamlit run app.py
```

---

## Key Findings

### Regression
| Model | n | R² | Key Result |
|-------|---|----|------------|
| OLS: Obesity ~ Inactive | 391 | 0.213 | β=0.445, p<0.001 |
| OLS: Obesity ~ PoorDiet | 84 | 0.170 | β=0.658, p<0.001 |
| Ridge: Obesity ~ Inactive + PoorDiet | 84 | 0.237 | CV R²=0.161 |

### Classification (5-Fold CV Accuracy)
| Model | Dataset A | Dataset B |
|-------|-----------|-----------|
| Logistic Regression | 66.5% | 70.2% |
| Decision Tree | 60.9% | 70.1% |
| Random Forest | 67.8% | 70.3% |
| Linear SVM | 66.5% | 70.2% |

### Clustering
- **Optimal k = 4** (elbow + silhouette = 0.43)
- Four clusters: High Risk, Low Risk, Moderate, **Young Adult Paradox** (low obesity despite poor diet)
- Education and income drive the extreme clusters; socioeconomic gradients dominate

### Key Insight
Obesity rose **+5.8 percentage points** nationally from 2011–2024 (r=+0.98 with Year), while physical inactivity showed **no significant trend** (r=-0.39, p=0.17). Behavioral indicators explain cross-sectional variation better than temporal trends — something beyond inactivity and diet is driving the national obesity increase.

---

## Skills Demonstrated

`pandas` `numpy` `scikit-learn` `statsmodels` `matplotlib` `seaborn` `streamlit` `plotly`  
OLS regression · Ridge regression · VIF analysis · Logistic Regression · Decision Tree · Random Forest · Linear SVM · K-Means · PCA · Silhouette analysis · Cross-validation · Ecological data analysis

---

## Author

**Chris Rodrigues** · Illinois Institute of Technology · MAS Data Science  
Course: Statistical Learning
