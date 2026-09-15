# Random Survival Forest Quantifies Intra-Stage Heterogeneity of Recurrence-Free Survival After Curative Gastrectomy

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![scikit-survival](https://img.shields.io/badge/scikit--survival-0.22-orange.svg)](https://scikit-survival.readthedocs.io/)
[![TRIPOD+AI](https://img.shields.io/badge/reporting-TRIPOD%2BAI-green.svg)](https://doi.org/10.1136/bmj-2023-078378)
[![STROBE](https://img.shields.io/badge/reporting-STROBE-green.svg)](https://www.strobe-statement.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Goshi Fujimoto, MD; Hiroshi Kusanagi, MD, PhD**  
Department of Gastroenterological Surgery, Kameda Medical Center, Kamogawa, Japan

> Submitted to *International Journal of Clinical Oncology* (IJCO)  
> Trial registration: UMIN Clinical Trial Registry, UMIN000061288

---

## Overview

This repository contains the complete analysis code for a single-center retrospective cohort study that:

1. Developed a **Random Survival Forest (RSF)** model predicting recurrence-free survival (RFS) after curative R0 gastrectomy for pathological Stage IIA–IIIC gastric cancer in the ACTS-GC era (2008–2020)
2. Performed **temporal validation** (development 2008–2015, n = 330; validation 2016–2020, n = 156)
3. **Quantified intra-stage recurrence risk heterogeneity** within each pathological stage group

### Key Findings

| Metric | RSF (Development) | RSF (Validation, 95% CI) | Cox pStage-only (Validation) |
|--------|-------------------|--------------------------|-------------------------------|
| Harrell's C-index | 0.801 | **0.729 (0.653–0.809)** | 0.687 (0.616–0.765) |
| td-AUC 3-year | — | **0.764 (0.673–0.851)** | 0.711 (0.619–0.802) |
| td-AUC 5-year | — | **0.777 (0.688–0.863)** | 0.731 (0.637–0.823) |
| NRI (continuous) | — | 0.065 (−0.126 to 0.242) | Reference |
| IDI | — | 0.017 (−0.005 to 0.039) | Reference |

#### Intra-Stage Heterogeneity (combined cohort, n = 486)

| Stage Group | N | Events | IQR (pp) | CV | Log-rank p |
|-------------|---|--------|----------|----|------------|
| Stage II (IIA+IIB) | 243 | 40 | 10.1 | 0.401 | **< 0.001** |
| Stage IIIA | 108 | 40 | 13.6 | 0.355 | **< 0.001** |
| Stage IIIB | 66 | 25 | 14.7 | 0.268 | **< 0.001** |
| Stage IIIC | 69 | 48 | 15.7 | **0.180** | 0.678 |

*IQR = interquartile range of predicted 3-year cumulative recurrence probability (pp = percentage points); CV = coefficient of variation. Low/High risk split at development-cohort stage-specific median predicted probability.*

> **Stage IIIC** shows a narrow predicted-risk distribution (CV = 0.180 vs 0.355–0.401 for Stages II–IIIA), indicating that TNM staging alone adequately characterises the uniformly high-risk burden in this subgroup.

#### Variable Importance (Permutation, RSF Development Cohort)

| Rank | Predictor | ΔC-index |
|------|-----------|----------|
| 1 | Positive lymph node count | 0.111 |
| 2 | pT-category (ordinal) | 0.038 |
| 3 | Age | 0.021 |
| 4 | CA19-9 | 0.019 |
| 5 | CEA | 0.017 |
| 6 | Tumour size | 0.015 |
| 7 | LVI grade (0–3) | 0.014 |
| 8 | CRP | 0.009 |
| 9 | Albumin | 0.007 |
| 10 | S-1 adjuvant (ACTS-GC era) | 0.003 |

---

## Study Design

```
Inclusion: R0 gastrectomy, pStage IIA–IIIC, 2008–2020, no neoadjuvant chemotherapy
                              n = 486
                    ┌─────────────────────┐
          Development (2008–2015)   Temporal Validation (2016–2020)
               n = 330                      n = 156
             events = 102                 events = 51
             EPV = 10.2                        —
```

**Outcome**: Recurrence-free survival (RFS); death without recurrence = competing event  
**Prediction horizon**: 4–6 weeks post-surgery (pathological results finalised)  
**Comparison**: RSF vs. Cox proportional hazards (pStage-only, null model)

---

## Repository Structure

```
gc_recurrence_rsf/
├── README.md
├── requirements.txt
├── LICENSE
│
├── config.py            # Study constants, predictor names, colour palette
├── prepare_data.py      # Data loading, feature engineering, MICE imputation,
│                        # temporal split → returns dev/val DataFrames
├── models.py            # RSF + Cox fitting, C-index, td-AUC, bootstrap CI
├── run_analysis.py      # Full pipeline: models → figures → results summary
│
└── figures/             # Output directory (created at runtime)
    ├── fig1_cindex_comparison.png
    ├── fig2_calibration_3yr5yr.png
    ├── fig3_dca.png
    ├── fig4_intra_stage_heterogeneity.png
    └── fig5_shap_permutation_importance.png
```

---

## Prerequisites

### Python version
Python ≥ 3.10 (developed with 3.13)

### Installation

```bash
git clone https://github.com/GoshiFujimoto/gc_recurrence_rsf.git
cd gc_recurrence_rsf
pip install -r requirements.txt
```

### Data

This study uses a single-centre institutional dataset that cannot be shared publicly due to privacy regulations (Kameda Medical Center IRB approval No. 25-003). The dataset contains:

- **Cohort**: 486 patients with pathological Stage IIA–IIIC gastric adenocarcinoma after R0 gastrectomy (2008–2020)
- **Outcome**: Date of recurrence or censoring; competing event (death without recurrence)
- **10 Predictors**: Age, albumin, CRP, CEA, CA19-9, pT (ordinal), positive LN count, tumour size, LVI grade (0–3), adjuvant S-1 (binary)

To run the code with your own data, prepare a spreadsheet with the column names matching those in `config.py → PREDICTORS` and update the file path in `prepare_data.py`.

---

## Usage

```bash
# Run complete analysis pipeline
python run_analysis.py
```

This will:
1. Load and preprocess data (MICE imputation, temporal split)
2. Fit RSF (n_estimators=500, min_samples_leaf=15) and Cox pStage-only
3. Compute C-index, td-AUC (3yr, 5yr) with 1,000 bootstrap CIs
4. Generate calibration plots (3yr, 5yr)
5. Run Decision Curve Analysis (DCA), NRI, IDI
6. Compute intra-stage heterogeneity (violin plots, KM curves)
7. Compute permutation importance (30 repeats) and SHAP (proxy GBC)
8. Save all figures to `figures/`

### Step-by-step

```python
from prepare_data import load_and_prepare, surv_array
from config import PREDICTORS

# 1. Load and impute
dev, val, imputer = load_and_prepare()

# 2. Fit RSF
from sksurv.ensemble import RandomSurvivalForest
rsf = RandomSurvivalForest(n_estimators=500, min_samples_leaf=15,
                            max_features='sqrt', n_jobs=-1, random_state=42)
rsf.fit(dev[PREDICTORS].values, surv_array(dev))

# 3. Predict 3-year cumulative recurrence
from models import pred_ci
ci3_val = pred_ci(rsf, val[PREDICTORS].values, t_days=3*365.25)
```

---

## Methods Summary

| Component | Implementation |
|-----------|---------------|
| **Outcome** | RFS (time-to-event); competing risk = death without recurrence |
| **Missing data** | Multivariate iterative imputation — `sklearn.impute.IterativeImputer` (Bayesian Ridge, max_iter=20); fitted on development cohort only |
| **Model** | `sksurv.ensemble.RandomSurvivalForest` (n_estimators=500, min_samples_leaf=15, max_features='sqrt', seed=42) |
| **Comparator** | Cox PH with pStage as sole ordinal predictor (`lifelines.CoxPHFitter`) |
| **Discrimination** | Harrell's C-index; td-AUC at 3 and 5 years (`sksurv.metrics.cumulative_dynamic_auc`) |
| **Calibration** | Quintile-based KM observed vs. model-predicted at 3yr and 5yr |
| **CI** | 1,000 bootstrap resamples of validation cohort |
| **DCA** | 3-year horizon, 5–70% threshold range; cause-specific KM for observed risk |
| **NRI/IDI** | Continuous reclassification; 1,000 bootstrap CIs |
| **Variable importance** | Permutation importance (30 repeats; ΔC-index); `sklearn.inspection.permutation_importance` |
| **SHAP** | Gradient-boosting classifier proxy (`sklearn.ensemble.GradientBoostingClassifier` + `shap.TreeExplainer`) for directional effect confirmation |
| **Intra-stage heterogeneity** | Violin plots + Kaplan–Meier (Low/High split at development-cohort stage-specific median) |
| **Random seed** | 42 (fixed throughout) |

**Reporting**: STROBE statement; TRIPOD+AI checklist ([Collins et al. BMJ 2024](https://doi.org/10.1136/bmj-2023-078378))

---

## Reproducibility

All random seeds are fixed at 42. The analysis was conducted with:

```
Python         3.13
scikit-learn   1.6
scikit-survival 0.22
lifelines      0.29
shap           0.46
pandas         2.2
numpy          1.26
matplotlib     3.9
scipy          1.13
```

See `requirements.txt` for pinned versions.

---

## Ethics

This study was approved by the Institutional Review Board of Kameda Medical Center (approval number: 25-003). Individual informed consent was waived due to the retrospective design; opt-out procedures were implemented and registered in the UMIN Clinical Trial Registry (UMIN000061288).

---

## Citation

If you use this code or methodology, please cite:

```bibtex
@article{fujimoto2025rsf,
  title   = {Random Survival Forest Quantifies Intra-Stage Heterogeneity 
             of Recurrence-Free Survival After Curative Gastrectomy: 
             A Single-Center Cohort Study},
  author  = {Fujimoto, Goshi and Kusanagi, Hiroshi},
  journal = {International Journal of Clinical Oncology},
  year    = {2025},
  note    = {Submitted}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

Assistance from a large language model (LLM) was utilized in the development of the analysis code. English language editing was provided by Editage (www.editage.com).
