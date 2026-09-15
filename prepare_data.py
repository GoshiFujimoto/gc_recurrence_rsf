"""
prepare_data.py
================
Load raw gastric cancer dataset, engineer features, run MICE imputation,
and split into development / temporal validation cohorts.

Usage
-----
    from prepare_data import load_and_prepare, surv_array
    dev, val, imputer = load_and_prepare()
    y_dev = surv_array(dev)
"""

import re
import sys
import warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from config import (SEED, DEV_CUTOFF_YEAR, TARGET_STAGES,
                    STAGE_GROUPS, PREDICTORS)

# ── pT ordinal encoding ───────────────────────────────────────────────────────
# Maps Japanese pathology strings -> ordinal 1-6
_PT_MAP = {
    '1a': 1, '1b': 2, 'm': 1, 'sm': 2, 'sm1': 2, 'sm2': 2,
    '2': 3, 'mp': 3, 'ss': 3, 'ssr': 4, '3': 4,
    'se': 5, '4a': 5, 'si': 6, '4b': 6, '0': 1,
}


def _encode_pt(val: str | float) -> float:
    if pd.isna(val):
        return np.nan
    return _PT_MAP.get(str(val).strip().lower(), np.nan)


def _encode_ly(val: str | float) -> float:
    """Extract LVI grade (0-3) from combined column such as 'ly3 v1'."""
    if pd.isna(val):
        return np.nan
    m = re.search(r'ly(\d)', str(val))
    return float(m.group(1)) if m else np.nan


# ── Main function ─────────────────────────────────────────────────────────────

def load_and_prepare(
    data_path: str = 'gastric_data.xlsx',
) -> tuple[pd.DataFrame, pd.DataFrame, IterativeImputer]:
    """
    Load, clean, engineer, impute, and split the dataset.

    Parameters
    ----------
    data_path : str
        Path to the Excel file containing the raw dataset.

    Returns
    -------
    dev : pd.DataFrame
        Development cohort (year <= DEV_CUTOFF_YEAR) with PREDICTORS imputed.
    val : pd.DataFrame
        Temporal validation cohort with PREDICTORS imputed (imputer fitted on dev).
    imputer : IterativeImputer
        Fitted imputer (for re-use on external data).
    """
    df = pd.read_excel(data_path)

    # ── Parse dates ───────────────────────────────────────────────────────────
    df['surgery_date']   = pd.to_datetime(df['術日'],       errors='coerce')
    df['recurrence_date'] = pd.to_datetime(df['再発日'],    errors='coerce')
    df['last_followup']   = pd.to_datetime(df['最終確認日'], errors='coerce')
    df['year'] = df['surgery_date'].dt.year

    # ── Inclusion / exclusion ─────────────────────────────────────────────────
    cohort = df[
        (df['R0'] == 1.0) &
        (df['pStage15版'].isin(TARGET_STAGES)) &
        (df['year'] >= 2008) & (df['year'] <= 2020) &
        (df['neo-adjuvant+'] != 1.0)
    ].copy()

    # ── Outcome variables ─────────────────────────────────────────────────────
    cohort['event_time_days'] = np.where(
        cohort['recurrence+'] == 1.0,
        (cohort['recurrence_date'] - cohort['surgery_date']).dt.days,
        (cohort['last_followup']  - cohort['surgery_date']).dt.days,
    )
    cohort['event']     = (cohort['recurrence+'] == 1.0).astype(bool)
    cohort['competing'] = (
        (cohort['death'] == 1) & (cohort['recurrence+'] == 0.0)
    ).astype(bool)

    # Drop rows with missing outcome or zero follow-up
    cohort = cohort.dropna(subset=['event_time_days'])
    cohort = cohort[cohort['event_time_days'] > 0].copy()

    # ── Stage group label ──────────────────────────────────────────────────────
    cohort['stage_group'] = cohort['pStage15版'].map(STAGE_GROUPS)

    # ── Feature engineering ───────────────────────────────────────────────────
    cohort['CA19_9']      = pd.to_numeric(cohort['CA19-9'],         errors='coerce')
    cohort['pT_ord']      = cohort['pT-factor'].apply(_encode_pt)
    cohort['pN_num']      = pd.to_numeric(cohort['リンパ節転移数'], errors='coerce')
    cohort['ly_ord']      = cohort['脈管侵襲'].apply(_encode_ly)
    cohort['adjuvant_s1'] = (cohort['adjuvant'] == 1.0).astype(float)
    cohort['size']        = pd.to_numeric(cohort['size'],            errors='coerce')
    for col in ['age', 'Alb', 'CRP', 'CEA']:
        cohort[col] = pd.to_numeric(cohort[col], errors='coerce')

    # ── Temporal split ────────────────────────────────────────────────────────
    dev = cohort[cohort['year'] <= DEV_CUTOFF_YEAR].reset_index(drop=True)
    val = cohort[cohort['year'] >  DEV_CUTOFF_YEAR].reset_index(drop=True)

    # ── MICE imputation ───────────────────────────────────────────────────────
    # Fit exclusively on development cohort to prevent data leakage.
    imputer = IterativeImputer(max_iter=20, random_state=SEED, min_value=0)
    dev = dev.copy()
    val = val.copy()
    dev[PREDICTORS] = imputer.fit_transform(dev[PREDICTORS])
    val[PREDICTORS] = imputer.transform(val[PREDICTORS])

    # Clip ordinal columns to valid ranges
    for df_ in (dev, val):
        df_['pT_ord']      = df_['pT_ord'].clip(1, 6).round()
        df_['ly_ord']      = df_['ly_ord'].clip(0, 3).round()
        df_['adjuvant_s1'] = df_['adjuvant_s1'].round().clip(0, 1)

    return dev, val, imputer


def surv_array(df: pd.DataFrame):
    """Return structured survival array (event bool, time float) for sksurv."""
    from sksurv.util import Surv
    return Surv.from_arrays(df['event'].values, df['event_time_days'].values)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    dev, val, _ = load_and_prepare()
    print(f"Development : n={len(dev):>4}  events={dev['event'].sum():>3}  "
          f"competing={dev['competing'].sum():>3}  "
          f"EPV={dev['event'].sum()/len(PREDICTORS):.1f}")
    print(f"Validation  : n={len(val):>4}  events={val['event'].sum():>3}  "
          f"competing={val['competing'].sum():>3}")
    print(f"\nStage distribution (development):")
    print(dev['stage_group'].value_counts().sort_index())
    print(f"\nMissing after MICE: {dev[PREDICTORS].isna().sum().sum()} cells")
