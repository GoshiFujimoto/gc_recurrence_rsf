# Random Survival Forest for Recurrence-Free Survival After Curative Gastrectomy: Temporal Validation and Assessment of Intra-Stage Risk Heterogeneity

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![scikit-survival](https://img.shields.io/badge/scikit--survival-0.28-orange.svg)](https://scikit-survival.readthedocs.io/)
[![TRIPOD+AI](https://img.shields.io/badge/reporting-TRIPOD%2BAI-green.svg)](https://doi.org/10.1136/bmj-2023-078378)
[![STROBE](https://img.shields.io/badge/reporting-STROBE-green.svg)](https://www.strobe-statement.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Goshi Fujimoto, MD; Hiroshi Kusanagi, MD, PhD**
Department of Gastroenterological Surgery, Kameda Medical Center, Kamogawa, Japan

> Manuscript under revision at *International Journal of Clinical Oncology* (IJCO)
> Trial registration: UMIN Clinical Trial Registry, UMIN000061288

---

## Overview

This repository contains the analysis code for a single-center retrospective cohort study that:

1. Developed a **Random Survival Forest (RSF)** model predicting recurrence-free survival (RFS) after curative R0 gastrectomy for pathological Stage IIA–IIIC gastric cancer in the ACTS-GC era (2008–2020)
2. Performed **temporal validation** against a pathological stage-only Cox model (development 2008–2015, n = 330; validation 2016–2020, n = 156)
3. Described **intra-stage heterogeneity** in predicted recurrence risk and evaluated whether it translated into outcome separation in the validation cohort

### Key Findings (temporal validation cohort, n = 156)

| Metric | RSF (95% CI) | Cox pStage-only (95% CI) | Paired difference RSF − Cox (95% CI) |
|--------|--------------|--------------------------|--------------------------------------|
| Harrell's C-index | 0.729 (0.653–0.809) | 0.687 (0.616–0.765) | +0.042 (−0.019 to 0.106) |
| td-AUC 3-year | 0.764 (0.673–0.851) | 0.711 (0.619–0.802) | +0.053 (−0.015 to 0.123) |
| td-AUC 5-year | 0.777 (0.688–0.863) | 0.731 (0.637–0.823) | +0.046 (−0.028 to 0.117) |

Development-cohort (apparent) C-index of RSF: 0.801.

At 3 years (validation cohort only; inverse-probability-of-censoring weighting; comparator = pStage-only Cox):

- Continuous NRI −0.24 (95% CI −0.58 to 0.12); IDI −0.01 (95% CI −0.06 to 0.03)
- DCA: net benefit of RSF exceeded the stage-based strategy at thresholds of approximately 30%–40% but was lower at 20%–25% and 45%–60%

All paired differences included zero; superiority over stage was **not** established.

#### Intra-Stage Heterogeneity (validation cohort only)

Low/high risk groups were defined by the development-cohort stage-specific median predicted 3-year recurrence probability and applied to the validation cohort.

| Stage group | N | Events | IQR (pp) | CV | Low-risk n / events | High-risk n / events | Log-rank p |
|-------------|---|--------|----------|----|---------------------|----------------------|------------|
| Stage II (IIA+IIB) | 63 | 9 | 8.1 | 0.352 | 28 / 4 | 35 / 5 | 0.94 |
| Stage IIIA | 47 | 19 | 9.2 | 0.311 | 25 / 6 | 22 / 13 | 0.079 |
| Stage IIIB | 22 | 6 | 13.8 | 0.305 | 14 / 4 | 8 / 2 | 0.58 |
| Stage IIIC | 24 | 17 | 14.1 | 0.201 | 18 / 12 | 6 / 5 | 0.84 |

*IQR = interquartile range of predicted 3-year cumulative recurrence probability (absolute variability; pp = percentage points; ≥ 10 pp pre-specified as clinically meaningful); CV = coefficient of variation (relative variability, which is compressed when mean predicted risk is high).*

- Dichotomized risk groups were not significantly separated in any stage (6–19 events per stage).
- In a stage-stratified Cox model, continuous predicted risk was associated with recurrence (HR per 10 percentage points 1.53, 95% CI 1.10–2.11), whereas the binary high- vs low-risk HR was 1.33 (0.75–2.36).
- Stage IIIC should not be regarded as uniformly high risk: its CV was the smallest, but its absolute IQR (14.1 pp) exceeded the 10-pp threshold.

#### Sensitivity Analyses (validation cohort)

| Analysis | n (development / validation) | C-index | td-AUC 3-year | td-AUC 5-year |
|----------|------------------------------|---------|---------------|---------------|
| Primary model | 330 / 156 | 0.729 | 0.764 | 0.777 |
| Without adjuvant chemotherapy | 330 / 156 | 0.725 | 0.760 | 0.774 |
| Landmark at 42 days after surgery | 325 / 155 | 0.725 | 0.758 | 0.748 |
| Excluding follow-up < 90 days without recurrence | 324 / 153 | 0.728 | 0.763 | 0.773 |

#### Variable Importance (permutation, RSF development cohort)

| Rank | Predictor | ΔC-index |
|------|-----------|----------|
| 1 | Positive lymph node count | 0.111 |
| 2 | pT-category (ordinal) | 0.038 |
| 3 | Age | 0.021 |
| 4 | CA19-9 | 0.019 |
| 5 | CEA | 0.017 |
| 6 | Tumor size | 0.015 |
| 7 | Lymphatic invasion (ly, grade 0–3) | 0.014 |
| 8 | CRP | 0.009 |
| 9 | Albumin | 0.007 |
| 10 | Adjuvant chemotherapy | 0.003 |

Adjuvant chemotherapy is a covariate reflecting selective administration (confounding by indication); its importance or SHAP value does **not** represent a treatment effect.

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

**Outcome**: Recurrence-free survival (RFS). Primary analysis is cause-specific (death without recurrence censored at death); the Aalen–Johansen cumulative incidence was examined as a check.
**Time of prediction**: 4–6 weeks after surgery (pathological results and adjuvant decision available). All blood tests (albumin, CRP, CEA, CA19-9) are preoperative values.
**Comparator**: Cox proportional hazards model with pathological stage as the sole ordinal predictor.

---

## Repository Structure

```
gc_recurrence_rsf/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── config.py               # Study constants, predictor names, RSF hyperparameters
├── prepare_data.py         # Data loading, feature engineering, iterative imputation,
│                           # temporal split → returns dev/val DataFrames
├── models.py               # RSF + Cox fitting, C-index, td-AUC, bootstrap CI, SHAP proxy
├── run_analysis.py         # Original pipeline (initial submission): models → figures
│
├── revision_analysis.py    # Revision: sensitivity analyses (without adjuvant chemotherapy,
│                           # 42-day landmark, exclusion of < 90-day follow-up),
│                           # validation-only intra-stage analysis, validation-only
│                           # NRI/IDI/DCA (IPCW), cohort comparison
├── revision_analysis2.py   # Revision: paired bootstrap RSF − Cox, stage-stratified Cox,
│                           # Aalen–Johansen vs 1 − KM
└── make_figs.py            # Revision: Figs. 3–5 (119 mm wide, 300 dpi TIFF)
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

This study uses a single-center institutional dataset that cannot be shared publicly due to privacy regulations (Kameda Medical Center IRB approval No. 25-003). The dataset contains:

- **Cohort**: 486 patients with pathological Stage IIA–IIIC gastric adenocarcinoma after R0 gastrectomy (2008–2020)
- **Outcome**: Date of recurrence or censoring; death without recurrence
- **10 predictors**: age, albumin, CRP, CEA, CA19-9, pT (ordinal), positive lymph node count, tumor size, lymphatic invasion grade (ly, 0–3), adjuvant chemotherapy (binary; column name `adjuvant_s1` is kept for compatibility)

To run the code with your own data, prepare a spreadsheet with column names matching `config.py → PREDICTORS` and update the file path in `prepare_data.py`.

---

## Usage

```bash
# Original pipeline (initial submission)
python run_analysis.py

# Revision analyses (run in this order)
python revision_analysis.py     # writes revision_results.json / .pkl
python revision_analysis2.py    # writes revision_results2.json / .pkl
python make_figs.py             # writes figs_out/Fig3–Fig5 (.png, .tif)
```

The `.pkl` files contain patient-level data and are excluded by `.gitignore`; do not commit them.

---

## Methods Summary

| Component | Implementation |
|-----------|---------------|
| **Outcome** | RFS (time-to-event); cause-specific primary analysis; Aalen–Johansen check |
| **Missing data** | Multivariate iterative imputation — `sklearn.impute.IterativeImputer` (Bayesian Ridge, max_iter=20); fitted on the development cohort only |
| **Model** | `sksurv.ensemble.RandomSurvivalForest` (n_estimators=500, min_samples_leaf=15, max_features='sqrt', seed=42); log-rank splitting; Nelson–Aalen cumulative hazard in terminal nodes averaged across trees; predicted recurrence probability = 1 − exp(−H(t)) |
| **Hyperparameters** | Fixed; not tuned (no hyperparameter search) |
| **Comparator** | Cox PH with pStage as the sole ordinal predictor (`lifelines.CoxPHFitter`) |
| **Discrimination** | Harrell's C-index; td-AUC at 3 and 5 years (`sksurv.metrics.cumulative_dynamic_auc`); paired bootstrap RSF − Cox differences |
| **Calibration** | Quintile-based Kaplan–Meier observed vs model-predicted at 3 and 5 years |
| **CI** | 1,000 bootstrap resamples of the validation cohort |
| **DCA / NRI / IDI** | Validation cohort only, 3-year horizon, comparator = pStage-only Cox; inverse-probability-of-censoring weights (Kaplan–Meier censoring distribution); continuous NRI |
| **Intra-stage heterogeneity** | Violin plots (IQR, CV) + Kaplan–Meier by development-derived stage-specific median cutoffs applied to the validation cohort alone; stage-stratified Cox |
| **Sensitivity analyses** | RSF without adjuvant chemotherapy; 42-day landmark (time from day 42, models refitted); exclusion of follow-up < 90 days without recurrence |
| **Variable importance** | Permutation importance (30 repeats; ΔC-index); `sklearn.inspection.permutation_importance` |
| **SHAP** | Gradient-boosting classifier proxy (`sklearn.ensemble.GradientBoostingClassifier` + `shap.TreeExplainer`) |
| **Random seed** | 42 (fixed throughout) |

**Reporting**: STROBE statement; TRIPOD+AI checklist ([Collins et al. BMJ 2024](https://doi.org/10.1136/bmj-2023-078378))

---

## Reproducibility

All random seeds are fixed at 42. The revision analyses were run with:

```
Python          3.13
scikit-learn    1.9.1
scikit-survival 0.28.0
lifelines       0.30.3
shap            0.54.1
```

See `requirements.txt` for minimum versions.

---

## Ethics

This study was approved by the Institutional Review Board of Kameda Medical Center (approval number: 25-003). Individual informed consent was waived because of the retrospective design; opt-out procedures were implemented, and the study was registered in the UMIN Clinical Trial Registry (UMIN000061288).

---

## Citation

If you use this code or methodology, please cite:

```bibtex
@article{fujimoto2026rsf,
  title   = {Random Survival Forest for recurrence-free survival after curative
             gastrectomy: temporal validation and assessment of intra-stage risk
             heterogeneity in a single-center cohort study},
  author  = {Fujimoto, Goshi and Kusanagi, Hiroshi},
  journal = {International Journal of Clinical Oncology},
  year    = {2026},
  note    = {Under revision}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

Assistance from a large language model (LLM) was utilized in the development of the analysis code. English language editing was provided by Editage (www.editage.com).
