"""
config.py
==========
Central configuration for the gc_recurrence_rsf analysis.
All scripts import from here; change study parameters in one place.
"""

# ── Random seed ───────────────────────────────────────────────────────────────
SEED = 42

# ── Cohort definition ────────────────────────────────────────────────────────
TARGET_STAGES   = ['IIA', 'IIB', 'IIIA', 'IIIB', 'IIIC']   # pStage 15th-ed
DEV_CUTOFF_YEAR = 2015   # development <= 2015; validation > 2015

# Stage groups for intra-stage heterogeneity analysis
# (IIA + IIB combined: observed recurrence rates 15.3% vs 17.9%)
STAGE_GROUPS = {
    'IIA':  'Stage II',
    'IIB':  'Stage II',
    'IIIA': 'Stage IIIA',
    'IIIB': 'Stage IIIB',
    'IIIC': 'Stage IIIC',
}

# ── Predictors (10 variables, pre-specified) ─────────────────────────────────
# Available at 4-6 weeks post-surgery (pathological results finalised).
# Fixed list; no data-driven selection. Hyperparameters fixed (not tuned).
# Surgical approach and extent of lymphadenectomy excluded (changed over the study period).
# Charlson Comorbidity Index excluded: it primarily predicts death without recurrence.
# Primary analysis is cause-specific (death without recurrence censored at death);
# Aalen-Johansen cumulative incidence is examined as a check.
PREDICTORS = [
    'age',          # Pre-op  | patient age (years)
    'Alb',          # Pre-op  | serum albumin (g/dL)
    'CRP',          # Pre-op  | C-reactive protein (mg/dL)  -- 10.5% missing -> MICE
    'CEA',          # Pre-op  | carcinoembryonic antigen (ng/mL)
    'CA19_9',       # Pre-op  | CA19-9 (U/mL)  -- renamed from 'CA19-9' in raw data
    'pT_ord',       # Path.   | T-category ordinal (1a=1 ... 4b=6)
    'pN_num',       # Path.   | number of positive lymph nodes (continuous)
    'size',         # Path.   | max tumour diameter (mm)  -- 2.7% missing -> MICE
    'ly_ord',       # Path.   | lymphatic invasion (ly) grade 0-3; venous invasion not included
    'adjuvant_s1',  # Post-op | adjuvant chemotherapy, any regimen (0/1; S-1-based in 91.5%)
                    #           column name kept for compatibility. Covariate reflecting
                    #           selective administration (confounding by indication);
                    #           its importance is NOT a treatment effect
]

# ── Evaluation horizons ──────────────────────────────────────────────────────
HZ_YEARS    = [3, 5]       # td-AUC and calibration at 3 yr and 5 yr
BOOTSTRAP_N = 1_000         # bootstrap iterations for 95% CIs
IQR_THRESHOLD_PP = 10       # >=10 percentage points within a stage = clinically meaningful

# ── RSF hyperparameters ──────────────────────────────────────────────────────
RSF_PARAMS = dict(
    n_estimators    = 500,
    min_samples_leaf = 15,
    max_features    = 'sqrt',
    n_jobs          = -1,
    random_state    = SEED,
)

# ── Publication colour palette ────────────────────────────────────────────────
COL = {
    'Stage II':   '#2196F3',
    'Stage IIIA': '#FF9800',
    'Stage IIIB': '#F44336',
    'Stage IIIC': '#9C27B0',
}
COL_RISK = {'Low': '#2196F3', 'High': '#F44336'}
