"""
models.py
==========
Fit RSF and pStage-only Cox; compute discrimination, calibration, DCA,
NRI/IDI, intra-stage heterogeneity, and variable importance.

Usage
-----
    from models import fit_rsf, fit_cox, pred_ci, run_metrics
"""

import sys
import warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import (concordance_index_censored,
                             cumulative_dynamic_auc)
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingClassifier
import shap

from config import (SEED, PREDICTORS, HZ_YEARS, RSF_PARAMS,
                    BOOTSTRAP_N, STAGE_GROUPS, COL_RISK)
from prepare_data import surv_array

DAYS = {y: y * 365.25 for y in HZ_YEARS}
np.random.seed(SEED)


# ── Model fitting ─────────────────────────────────────────────────────────────

def fit_rsf(X_dev: np.ndarray, y_dev) -> RandomSurvivalForest:
    """Fit Random Survival Forest on development cohort."""
    rsf = RandomSurvivalForest(**RSF_PARAMS)
    rsf.fit(X_dev, y_dev)
    return rsf


def fit_cox(dev: pd.DataFrame) -> tuple[CoxPHFitter, np.ndarray]:
    """
    Fit Cox PH with pathological stage as sole ordinal predictor (null model).
    Returns (fitted_cph, risk_scores_dev).
    """
    stage_map = {'IIA': 1, 'IIB': 2, 'IIIA': 3, 'IIIB': 4, 'IIIC': 5}
    d = dev[['event_time_days', 'event', 'pStage15版']].copy()
    d['pstage'] = d['pStage15版'].map(stage_map)
    d = d.rename(columns={'event_time_days': 'T', 'event': 'E'})
    cph = CoxPHFitter()
    cph.fit(d[['T', 'E', 'pstage']], duration_col='T', event_col='E')
    scores = cph.predict_partial_hazard(d).values
    return cph, scores


def predict_cox_scores(cph: CoxPHFitter, df: pd.DataFrame) -> np.ndarray:
    stage_map = {'IIA': 1, 'IIB': 2, 'IIIA': 3, 'IIIB': 4, 'IIIC': 5}
    d = df[['event_time_days', 'event', 'pStage15版']].copy()
    d['pstage'] = d['pStage15版'].map(stage_map)
    d = d.rename(columns={'event_time_days': 'T', 'event': 'E'})
    return cph.predict_partial_hazard(d).values


# ── Predictions ───────────────────────────────────────────────────────────────

def pred_ci(rsf: RandomSurvivalForest, X: np.ndarray, t_days: float) -> np.ndarray:
    """
    Predict cumulative incidence (1 - survival) at time t_days.

    Parameters
    ----------
    rsf     : fitted RandomSurvivalForest
    X       : feature matrix (n_samples, n_features)
    t_days  : evaluation time in days (e.g. 3 * 365.25)

    Returns
    -------
    ci : np.ndarray of shape (n_samples,)  -- values in [0, 1]
    """
    fns = rsf.predict_survival_function(X)
    ci  = np.zeros(len(X))
    for i, fn in enumerate(fns):
        idx = np.searchsorted(fn.x, t_days, side='right') - 1
        ci[i] = 0.0 if idx < 0 else 1.0 - fn.y[idx]
    return ci


# ── Performance metrics ───────────────────────────────────────────────────────

def compute_cindex(event: np.ndarray, time: np.ndarray,
                   risk: np.ndarray) -> float:
    return concordance_index_censored(event, time, risk)[0]


def bootstrap_metrics(y_dev, y_val,
                      r_val: np.ndarray, sc_val: np.ndarray,
                      e_times: np.ndarray, n: int = BOOTSTRAP_N,
                      ) -> dict:
    """
    1,000-resample bootstrap 95% CIs for C-index and td-AUC in validation cohort.
    Returns dict of lists for each metric.
    """
    from sksurv.util import Surv
    ev_v = y_val['event']
    t_v  = y_val['time']
    n_val = len(ev_v)
    rng  = np.random.default_rng(SEED)

    results = {k: [] for k in
               ['c_rsf', 'c_cox'] +
               [f'auc_rsf_{yr}' for yr in HZ_YEARS] +
               [f'auc_cox_{yr}' for yr in HZ_YEARS]}

    for _ in range(n):
        idx  = rng.integers(0, n_val, n_val)
        y_b  = Surv.from_arrays(ev_v[idx], t_v[idx])
        if y_b['event'].sum() == 0:
            continue
        try:
            results['c_rsf'].append(
                concordance_index_censored(ev_v[idx], t_v[idx], r_val[idx])[0])
            results['c_cox'].append(
                concordance_index_censored(ev_v[idx], t_v[idx], sc_val[idx])[0])
            auc_r, _ = cumulative_dynamic_auc(y_dev, y_b, r_val[idx],  e_times)
            auc_c, _ = cumulative_dynamic_auc(y_dev, y_b, sc_val[idx], e_times)
            for i, yr in enumerate(HZ_YEARS):
                results[f'auc_rsf_{yr}'].append(auc_r[i])
                results[f'auc_cox_{yr}'].append(auc_c[i])
        except Exception:
            continue

    return results


def ci95(arr: list | np.ndarray) -> tuple[float, float]:
    a = np.array([x for x in arr if not np.isnan(x)])
    if len(a) < 10:
        return np.nan, np.nan
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


# ── NRI / IDI ─────────────────────────────────────────────────────────────────

def continuous_nri_idi(pred_new: np.ndarray, pred_ref: np.ndarray,
                        event: np.ndarray) -> dict:
    """
    Continuous NRI and IDI (Pencina 2008).
    pred_new, pred_ref : predicted probabilities [0, 1]
    event              : boolean event indicator
    """
    ev_up   = np.mean(pred_new[event]  > pred_ref[event])
    ev_down = np.mean(pred_new[event]  < pred_ref[event])
    ne_up   = np.mean(pred_new[~event] > pred_ref[~event])
    ne_down = np.mean(pred_new[~event] < pred_ref[~event])
    nri = (ev_up - ev_down) + (ne_down - ne_up)
    idi = (np.mean(pred_new[event])  - np.mean(pred_ref[event])) - \
          (np.mean(pred_new[~event]) - np.mean(pred_ref[~event]))
    return {'nri': nri, 'nri_events': ev_up - ev_down,
            'nri_nonevents': ne_down - ne_up, 'idi': idi}


# ── Intra-stage heterogeneity ─────────────────────────────────────────────────

def intra_stage_analysis(all_data: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """
    Compute IQR, CV, and log-rank p per stage group.

    Parameters
    ----------
    all_data   : combined dev+val DataFrame with 'ci3', 'risk_group',
                 'stage_group', 'event', 'event_time_days' columns
    thresholds : {stage_group: median ci3 from dev cohort}

    Returns
    -------
    DataFrame with one row per stage group.
    """
    rows = []
    for sg in ['Stage II', 'Stage IIIA', 'Stage IIIB', 'Stage IIIC']:
        s   = all_data[all_data['stage_group'] == sg]
        ci  = s['ci3']
        iqr = ci.quantile(0.75) - ci.quantile(0.25)
        cv  = ci.std() / ci.mean()
        lo  = s[s['risk_group'] == 'Low']
        hi  = s[s['risk_group'] == 'High']
        lr  = logrank_test(lo['event_time_days'], hi['event_time_days'],
                           lo['event'], hi['event'])
        rows.append({
            'stage_group': sg,
            'N': len(s),
            'events': int(s['event'].sum()),
            'event_rate': s['event'].mean(),
            'median_ci3': ci.median(),
            'IQR_pp': iqr * 100,
            'CV': cv,
            'threshold': thresholds.get(sg, np.nan),
            'low_n': len(lo),  'low_events': int(lo['event'].sum()),
            'high_n': len(hi), 'high_events': int(hi['event'].sum()),
            'logrank_p': lr.p_value,
        })
    return pd.DataFrame(rows)


# ── Variable importance ───────────────────────────────────────────────────────

def get_permutation_importance(rsf: RandomSurvivalForest,
                                X_dev: np.ndarray, y_dev,
                                n_repeats: int = 30) -> pd.DataFrame:
    pi = permutation_importance(rsf, X_dev, y_dev,
                                n_repeats=n_repeats, random_state=SEED, n_jobs=-1)
    return (pd.DataFrame({
                'feature': PREDICTORS,
                'mean': pi.importances_mean,
                'std':  pi.importances_std,
            })
            .sort_values('mean', ascending=False)
            .reset_index(drop=True))


def get_shap_values(X_dev: np.ndarray, y_bin: np.ndarray) -> np.ndarray:
    """
    Compute SHAP values using a gradient-boosting classifier proxy.
    RSF is not currently compatible with shap.TreeExplainer; GBC is used
    for directional effect confirmation only.

    Parameters
    ----------
    X_dev : feature matrix (development cohort)
    y_bin : binary recurrence indicator (0/1)

    Returns
    -------
    shap_vals : np.ndarray (n_samples, n_features)
    """
    gbc = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        min_samples_leaf=15, subsample=0.8, random_state=SEED)
    gbc.fit(X_dev, y_bin)
    explainer  = shap.TreeExplainer(gbc)
    shap_array = explainer.shap_values(X_dev)
    # For binary GBC, shap_values may be a 2D array (class 1 = recurrence)
    if isinstance(shap_array, list):
        return shap_array[1]
    return shap_array
