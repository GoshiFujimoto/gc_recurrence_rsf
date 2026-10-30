"""
revision_analysis.py -- additional analyses for the IJCO revision.

1. Re-fit baseline RSF (reproduce manuscript numbers)
2. Sensitivity analysis: RSF with vs without adjuvant chemotherapy
3. Landmark sensitivity analysis at 42 days after surgery
4. Sensitivity excluding patients with <90 d follow-up and no recurrence
5. Validation-cohort-only intra-stage analysis (dev-derived cutoffs)
6. Validation-cohort NRI / IDI / DCA at 3 years (IPCW, comparator = pStage-only Cox)
7. Cohort comparison (Table 1 additions)

Requires the institutional data file (gastric_data.xlsx), which is not public.
Outputs: revision_results.json / revision_results.pkl (patient-level; do not commit).
"""
import sys, json, warnings, pickle, re
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.ensemble import RandomSurvivalForest
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from sklearn.inspection import permutation_importance
from scipy import stats

from config import SEED, PREDICTORS, RSF_PARAMS, BOOTSTRAP_N
from prepare_data import load_and_prepare, surv_array
from models import fit_rsf, fit_cox, predict_cox_scores, pred_ci, ci95

D3, D5 = 3 * 365.25, 5 * 365.25
STAGES = ['Stage II', 'Stage IIIA', 'Stage IIIB', 'Stage IIIC']
R = {}


def cidx(ev, t, risk):
    return concordance_index_censored(ev.astype(bool), t, risk)[0]


def tdauc(y_tr, y_te, risk):
    a, _ = cumulative_dynamic_auc(y_tr, y_te, risk, np.array([D3, D5]))
    return a


# ------------------------------------------------------------------ 1. baseline
dev, val, imp = load_and_prepare('gastric_data.xlsx')
X_dev = dev[PREDICTORS].values.astype(float)
X_val = val[PREDICTORS].values.astype(float)
y_dev, y_val = surv_array(dev), surv_array(val)
rsf = fit_rsf(X_dev, y_dev)
cph, sc_dev = fit_cox(dev)
sc_val = predict_cox_scores(cph, val)
r_dev, r_val = rsf.predict(X_dev), rsf.predict(X_val)
ev_v, t_v = val['event'].values, val['event_time_days'].values
ev_d, t_d = dev['event'].values, dev['event_time_days'].values
R['baseline'] = dict(
    c_rsf_dev=cidx(ev_d, t_d, r_dev), c_rsf_val=cidx(ev_v, t_v, r_val),
    c_cox_dev=cidx(ev_d, t_d, sc_dev), c_cox_val=cidx(ev_v, t_v, sc_val),
    auc_rsf=list(tdauc(y_dev, y_val, r_val)), auc_cox=list(tdauc(y_dev, y_val, sc_val)))
print('BASELINE', R['baseline'])

# ------------------------------------------------------------------ 2. adjuvant chemotherapy sensitivity
idx_s1 = PREDICTORS.index('adjuvant_s1')
keep = [i for i in range(len(PREDICTORS)) if i != idx_s1]
rsf9 = fit_rsf(X_dev[:, keep], y_dev)
r_val9 = rsf9.predict(X_val[:, keep])
r_dev9 = rsf9.predict(X_dev[:, keep])
base_c9 = cidx(ev_v, t_v, r_val9)
auc9 = tdauc(y_dev, y_val, r_val9)
rng = np.random.default_rng(SEED)
n = len(val)
dC, dA3, dA5, C9b, A39b, A59b = [], [], [], [], [], []
for _ in range(BOOTSTRAP_N):
    ix = rng.integers(0, n, n)
    yb = Surv.from_arrays(ev_v[ix], t_v[ix])
    try:
        c_f = cidx(ev_v[ix], t_v[ix], r_val[ix]); c_r = cidx(ev_v[ix], t_v[ix], r_val9[ix])
        a_f = tdauc(y_dev, yb, r_val[ix]); a_r = tdauc(y_dev, yb, r_val9[ix])
    except Exception:
        continue
    dC.append(c_f - c_r); dA3.append(a_f[0] - a_r[0]); dA5.append(a_f[1] - a_r[1])
    C9b.append(c_r); A39b.append(a_r[0]); A59b.append(a_r[1])
pi_dev = permutation_importance(rsf, X_dev, y_dev, n_repeats=30, random_state=SEED, n_jobs=-1)
pi_val = permutation_importance(rsf, X_val, y_val, n_repeats=30, random_state=SEED, n_jobs=-1)
imp_tab = pd.DataFrame({'feature': PREDICTORS, 'dev_mean': pi_dev.importances_mean,
                        'val_mean': pi_val.importances_mean}).sort_values('dev_mean', ascending=False)
imp_tab['rank_dev'] = range(1, 11)
imp_tab['rank_val'] = imp_tab['val_mean'].rank(ascending=False).astype(int)
R['s1_sens'] = dict(
    c9_val=base_c9, c9_ci=ci95(C9b), auc9=list(auc9), auc9_3ci=ci95(A39b), auc9_5ci=ci95(A59b),
    c9_dev=cidx(ev_d, t_d, r_dev9),
    dC=float(base_c9 * -1 + R['baseline']['c_rsf_val']), dC_ci=ci95(dC),
    dA3=float(R['baseline']['auc_rsf'][0] - auc9[0]), dA3_ci=ci95(dA3),
    dA5=float(R['baseline']['auc_rsf'][1] - auc9[1]), dA5_ci=ci95(dA5),
    importance=imp_tab.to_dict('records'))
print('S1 SENS', {k: v for k, v in R['s1_sens'].items() if k != 'importance'})
print(imp_tab.to_string(index=False))

# ------------------------------------------------------------------ 3. landmark 42 days
LM = 42.0
dl = dev[dev.event_time_days > LM].copy(); vl = val[val.event_time_days > LM].copy()
for d_ in (dl, vl):
    d_['event_time_days'] = d_['event_time_days'] - LM
yl_d, yl_v = surv_array(dl), surv_array(vl)
rsf_l = fit_rsf(dl[PREDICTORS].values.astype(float), yl_d)
rl_v = rsf_l.predict(vl[PREDICTORS].values.astype(float))
cph_l, _ = fit_cox(dl)
sl_v = predict_cox_scores(cph_l, vl)
D3l, D5l = D3, D5
al_r, _ = cumulative_dynamic_auc(yl_d, yl_v, rl_v, np.array([D3, D5]))
al_c, _ = cumulative_dynamic_auc(yl_d, yl_v, sl_v, np.array([D3, D5]))
R['landmark'] = dict(n_dev=len(dl), n_val=len(vl), dropped_dev=len(dev) - len(dl), dropped_val=len(val) - len(vl),
                     ev_dev=int(dl.event.sum()), ev_val=int(vl.event.sum()),
                     c_rsf=cidx(vl.event.values, vl.event_time_days.values, rl_v),
                     c_cox=cidx(vl.event.values, vl.event_time_days.values, sl_v),
                     auc_rsf=list(al_r), auc_cox=list(al_c))
print('LANDMARK', R['landmark'])

# ------------------------------------------------------------------ 4. exclude <90 d no-recurrence
import prepare_data as pdm
_orig_read = pd.read_excel


def _filter_read(*a, **k):
    df = _orig_read(*a, **k)
    return df


# replicate pipeline with the extra exclusion by post-filtering through a monkeypatched DataFrame step
def load_excl90():
    import config
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer
    all_ = pd.concat([dev.assign(_s='dev'), val.assign(_s='val')])
    # raw (pre-imputation) values are needed: rebuild without imputation
    return all_


# raw cohort (pre-imputation) rebuilt by re-running the data steps without MICE
raw = pd.read_excel('gastric_data.xlsx')
raw['surgery_date'] = pd.to_datetime(raw['術日'], errors='coerce')
raw['recurrence_date'] = pd.to_datetime(raw['再発日'], errors='coerce')
raw['last_followup'] = pd.to_datetime(raw['最終確認日'], errors='coerce')
raw['year'] = raw['surgery_date'].dt.year
co = raw[(raw['R0'] == 1.0) & (raw['pStage15版'].isin(['IIA', 'IIB', 'IIIA', 'IIIB', 'IIIC'])) &
         (raw['year'] >= 2008) & (raw['year'] <= 2020) & (raw['neo-adjuvant+'] != 1.0)].copy()
co['event_time_days'] = np.where(co['recurrence+'] == 1.0, (co['recurrence_date'] - co['surgery_date']).dt.days,
                                 (co['last_followup'] - co['surgery_date']).dt.days)
co['event'] = (co['recurrence+'] == 1.0).astype(bool)
co = co.dropna(subset=['event_time_days']); co = co[co['event_time_days'] > 0].copy()
co['stage_group'] = co['pStage15版'].map({'IIA': 'Stage II', 'IIB': 'Stage II', 'IIIA': 'Stage IIIA',
                                          'IIIB': 'Stage IIIB', 'IIIC': 'Stage IIIC'})
co['CA19_9'] = pd.to_numeric(co['CA19-9'], errors='coerce')
co['pT_ord'] = co['pT-factor'].apply(pdm._encode_pt)
co['pN_num'] = pd.to_numeric(co['リンパ節転移数'], errors='coerce')
co['ly_ord'] = co['脈管侵襲'].apply(pdm._encode_ly)
co['adjuvant_s1'] = (co['adjuvant'] == 1.0).astype(float)
co['size'] = pd.to_numeric(co['size'], errors='coerce')
for c in ['age', 'Alb', 'CRP', 'CEA']:
    co[c] = pd.to_numeric(co[c], errors='coerce')
R['n_raw_cohort'] = len(co)
# missingness per predictor (overall, dev, val) before imputation
miss = {}
for p in PREDICTORS:
    miss[p] = dict(all=float(co[p].isna().mean()), dev=float(co[co.year <= 2015][p].isna().mean()),
                   val=float(co[co.year > 2015][p].isna().mean()))
R['missing'] = miss
co2 = co[~((co.event_time_days < 90) & (~co.event))].copy()
d2 = co2[co2.year <= 2015].reset_index(drop=True); v2 = co2[co2.year > 2015].reset_index(drop=True)
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
im2 = IterativeImputer(max_iter=20, random_state=SEED, min_value=0)
d2[PREDICTORS] = im2.fit_transform(d2[PREDICTORS]); v2[PREDICTORS] = im2.transform(v2[PREDICTORS])
for d_ in (d2, v2):
    d_['pT_ord'] = d_['pT_ord'].clip(1, 6).round(); d_['ly_ord'] = d_['ly_ord'].clip(0, 3).round()
    d_['adjuvant_s1'] = d_['adjuvant_s1'].round().clip(0, 1)
y2d, y2v = surv_array(d2), surv_array(v2)
rsf2 = fit_rsf(d2[PREDICTORS].values.astype(float), y2d)
r2v = rsf2.predict(v2[PREDICTORS].values.astype(float))
cph2, _ = fit_cox(d2); s2v = predict_cox_scores(cph2, v2)
a2r = tdauc(y2d, y2v, r2v); a2c = tdauc(y2d, y2v, s2v)
R['excl90'] = dict(n_dev=len(d2), n_val=len(v2), ev_dev=int(d2.event.sum()), ev_val=int(v2.event.sum()),
                   c_rsf=cidx(v2.event.values, v2.event_time_days.values, r2v),
                   c_cox=cidx(v2.event.values, v2.event_time_days.values, s2v),
                   auc_rsf=list(a2r), auc_cox=list(a2c))
print('EXCL90', R['excl90'])

# ------------------------------------------------------------------ 5. validation-only intra-stage
dev = dev.copy(); val = val.copy()
dev['ci3'] = pred_ci(rsf, X_dev, D3); val['ci3'] = pred_ci(rsf, X_val, D3)
dev['ci5'] = pred_ci(rsf, X_dev, D5); val['ci5'] = pred_ci(rsf, X_val, D5)
thr = {sg: float(dev.loc[dev.stage_group == sg, 'ci3'].median()) for sg in STAGES}
for d_ in (dev, val):
    d_['risk_group'] = np.where(d_['ci3'] >= d_['stage_group'].map(thr), 'High', 'Low')
comb = pd.concat([dev.assign(split='Dev'), val.assign(split='Val')], ignore_index=True)


def km_rate(df, yrs):
    k = KaplanMeierFitter().fit(df['event_time_days'] / 365.25, df['event'])
    return float(1 - k.survival_function_at_times([yrs]).values[0])


def stage_block(df, sg, bs=True):
    s = df[df.stage_group == sg]
    lo, hi = s[s.risk_group == 'Low'], s[s.risk_group == 'High']
    ci = s['ci3']
    out = dict(N=len(s), events=int(s.event.sum()), median=float(ci.median()),
               IQR_pp=float((ci.quantile(.75) - ci.quantile(.25)) * 100), CV=float(ci.std() / ci.mean()),
               mean=float(ci.mean()), sd=float(ci.std()),
               low_n=len(lo), low_ev=int(lo.event.sum()), high_n=len(hi), high_ev=int(hi.event.sum()))
    if len(lo) > 1 and len(hi) > 1 and lo.event.sum() + hi.event.sum() > 0:
        out['logrank_p'] = float(logrank_test(lo.event_time_days, hi.event_time_days, lo.event, hi.event).p_value)
        d = s[['event_time_days', 'event']].copy(); d['hi'] = (s.risk_group == 'High').astype(int)
        try:
            c = CoxPHFitter().fit(d, 'event_time_days', 'event')
            out['HR_high'] = float(np.exp(c.params_['hi']))
            out['HR_lo'], out['HR_hi'] = [float(x) for x in np.exp(c.confidence_intervals_.loc['hi'].values)]
        except Exception:
            pass
        try:
            out['km3_low'] = km_rate(lo, 3); out['km3_high'] = km_rate(hi, 3)
        except Exception:
            pass
    # continuous: HR per 10 pp of predicted 3-y risk and within-stage C-index
    d = s[['event_time_days', 'event']].copy(); d['x10'] = s['ci3'] * 10
    try:
        c = CoxPHFitter(penalizer=0.0).fit(d, 'event_time_days', 'event')
        out['HR10'] = float(np.exp(c.params_['x10']))
        out['HR10_lo'], out['HR10_hi'] = [float(x) for x in np.exp(c.confidence_intervals_.loc['x10'].values)]
        out['HR10_p'] = float(c.summary.loc['x10', 'p'])
    except Exception:
        pass
    try:
        out['cidx'] = cidx(s.event.values, s.event_time_days.values, s.ci3.values)
        if bs:
            rg = np.random.default_rng(SEED); bl = []
            for _ in range(1000):
                ix = rg.integers(0, len(s), len(s))
                if s.event.values[ix].sum() == 0: continue
                try: bl.append(cidx(s.event.values[ix], s.event_time_days.values[ix], s.ci3.values[ix]))
                except Exception: pass
            out['cidx_ci'] = ci95(bl)
    except Exception:
        pass
    return out


R['thr'] = thr
R['stage_val'] = {sg: stage_block(val, sg) for sg in STAGES}
R['stage_dev'] = {sg: stage_block(dev, sg, bs=False) for sg in STAGES}
R['stage_comb'] = {sg: stage_block(comb, sg, bs=False) for sg in STAGES}
# IIIC quartiles (combined, descriptive) + validation tertile/median note
s = comb[comb.stage_group == 'Stage IIIC'].copy()
s['q'] = pd.qcut(s.ci3, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
q1, q4 = s[s.q == 'Q1'], s[s.q == 'Q4']
R['iiic_quartile_comb'] = dict(p=float(logrank_test(q1.event_time_days, q4.event_time_days, q1.event, q4.event).p_value),
                               q1_n=len(q1), q1_ev=int(q1.event.sum()), q4_n=len(q4), q4_ev=int(q4.event.sum()),
                               overall_p=float(multivariate_logrank_test(s.event_time_days, s.q, s.event).p_value))
# overall within-stage: stage-adjusted
for nm, df in [('val', val), ('comb', comb)]:
    for sg in STAGES:
        pass
print('THRESH', thr)
for k in ('stage_val', 'stage_dev', 'stage_comb'):
    print(k)
    print(pd.DataFrame(R[k]).T[['N', 'events', 'IQR_pp', 'CV', 'low_n', 'low_ev', 'high_n', 'high_ev', 'logrank_p',
                                'HR_high', 'HR10', 'HR10_lo', 'HR10_hi', 'cidx']].round(3).to_string())
print('IIIC quartile comb', R['iiic_quartile_comb'])

# ------------------------------------------------------------------ 6. NRI / IDI / DCA (validation, 3y, IPCW)
tv, ev = val.event_time_days.values, val.event.values.astype(bool)
# comparator: pStage-only Cox 3-y risk (fitted on development)
stage_map = {'IIA': 1, 'IIB': 2, 'IIIA': 3, 'IIIB': 4, 'IIIC': 5}
dd = val[['pStage15版']].copy(); dd['pstage'] = dd['pStage15版'].map(stage_map)
p_stage = 1 - cph.predict_survival_function(dd[['pstage']].rename(columns={'pstage': 'pstage'}), times=[D3]).values[0]
# note: cph was fitted with columns T,E,pstage
p_rsf = val['ci3'].values


def ipcw(t, e, horizon):
    kc = KaplanMeierFitter().fit(t, event_observed=~e)
    G = lambda x: np.clip(kc.survival_function_at_times(x).values, 1e-3, 1)
    case = e & (t <= horizon)
    ctrl = t > horizon
    w = np.zeros(len(t))
    w[case] = 1 / G(t[case]); w[ctrl] = 1 / G(np.full(ctrl.sum(), horizon))
    return w, case, ctrl


def nri_idi(pn, pr, w, case, ctrl):
    up, dn = pn > pr, pn < pr
    sw_c, sw_n = (w * case).sum(), (w * ctrl).sum()
    nri_e = ((w * case * up).sum() - (w * case * dn).sum()) / sw_c
    nri_n = ((w * ctrl * dn).sum() - (w * ctrl * up).sum()) / sw_n
    idi = ((w * case * pn).sum() / sw_c - (w * case * pr).sum() / sw_c) - \
          ((w * ctrl * pn).sum() / sw_n - (w * ctrl * pr).sum() / sw_n)
    return nri_e + nri_n, nri_e, nri_n, idi


def nb(p, w, case, ctrl, pts, n):
    out = []
    for pt in pts:
        tr = p >= pt
        out.append(((w * case * tr).sum() - (w * ctrl * tr).sum() * pt / (1 - pt)) / n)
    return np.array(out)


w, case, ctrl = ipcw(tv, ev, D3)
nri, nri_e, nri_n, idi = nri_idi(p_rsf, p_stage, w, case, ctrl)
pts = np.round(np.arange(0.05, 0.701, 0.01), 2)
nb_rsf = nb(p_rsf, w, case, ctrl, pts, len(tv)); nb_stg = nb(p_stage, w, case, ctrl, pts, len(tv))
nb_all = nb(np.ones(len(tv)), w, case, ctrl, pts, len(tv))
rg = np.random.default_rng(SEED)
bn, bi, dnb = [], [], []
sel = [i for i, p in enumerate(pts) if 0.2 <= p <= 0.5]
for _ in range(BOOTSTRAP_N):
    ix = rg.integers(0, len(tv), len(tv))
    try:
        wb, cb, kb = ipcw(tv[ix], ev[ix], D3)
        r_ = nri_idi(p_rsf[ix], p_stage[ix], wb, cb, kb)
        bn.append(r_[0]); bi.append(r_[3])
        d_nb = nb(p_rsf[ix], wb, cb, kb, pts[sel], len(ix)) - nb(p_stage[ix], wb, cb, kb, pts[sel], len(ix))
        dnb.append(d_nb)
    except Exception:
        continue
dnb = np.array(dnb)
R['nri_idi_val'] = dict(nri=float(nri), nri_ci=ci95(bn), nri_e=float(nri_e), nri_n=float(nri_n),
                        idi=float(idi), idi_ci=ci95(bi), n_cases=int(case.sum()), n_ctrl=int(ctrl.sum()),
                        n_cens_before=int(len(tv) - case.sum() - ctrl.sum()))
R['dca_val'] = dict(pts=pts.tolist(), rsf=nb_rsf.tolist(), stage=nb_stg.tolist(), all=nb_all.tolist(),
                    dnb_mean=dnb.mean(0).tolist(), dnb_lo=np.percentile(dnb, 2.5, axis=0).tolist(),
                    dnb_hi=np.percentile(dnb, 97.5, axis=0).tolist(), dnb_pts=pts[sel].tolist())
dd_ = nb_rsf - nb_stg
print('NRI/IDI val', R['nri_idi_val'])
print('DCA RSF-stage difference (val) at thresholds:')
for p_, a, b_ in zip(pts, nb_rsf, nb_stg):
    if int(round(p_ * 100)) % 5 == 0:
        print(f'  {p_:.2f} rsf={a:.4f} stage={b_:.4f} diff={a - b_:+.4f} all={nb_all[list(pts).index(p_)]:.4f}')
print('frac of 20-50% thresholds where RSF>stage:', float(np.mean(dd_[sel] > 0)),
      'bootstrap CI of diff excludes 0 at', int(np.sum((np.array(R['dca_val']['dnb_lo']) > 0))), 'of', len(sel))

# also: calibration summary in validation (calibration slope-ish) -> predicted vs observed overall 3y
R['calib_val'] = dict(mean_pred3=float(val.ci3.mean()), km_obs3=km_rate(val, 3),
                      mean_pred5=float(val.ci5.mean()), km_obs5=km_rate(val, 5))
print('CALIB val', R['calib_val'])

# ------------------------------------------------------------------ 7. cohort comparison
allc = pd.concat([dev.assign(split='Dev'), val.assign(split='Val')], ignore_index=True)


def pct(m, s): return f'{int(m.sum())} ({m.mean() * 100:.1f}%)'


def regimen(x):
    if pd.isna(x): return 'unknown'
    x = str(x)
    if re.search(r'SOX|CAPOX|CapeOX|Capox|XELOX|XEROX|cape|DTX|CDDP|CBDCA|Her|PTX', x): return 'combination/other'
    if re.search(r'TS-1|S-1', x): return 'S-1'
    return 'unknown'


allc['regimen'] = np.where(allc.adjuvant == 1, allc['術後補助化学療法'].apply(regimen), 'none')
T = {}
for sp, d_ in allc.groupby('split'):
    T[sp] = dict(
        n=len(d_), open=pct(d_['open'] == 1, d_), lap=pct(d_['laparoscopic'] == 1, d_),
        TG=pct(d_['TG'] == 1, d_), DG=pct(d_['DG'] == 1, d_), PG=pct(d_['PG'] == 1, d_),
        D2=pct(d_['D2'] == 1, d_), ASA3=pct(d_['ASA-PS'].astype(str).str.startswith(('3', '4')), d_),
        G3=pct(d_['Grade≧3'] == 1, d_), adj=pct(d_['adjuvant'] == 1, d_),
        adj_S1=pct(d_['regimen'] == 'S-1', d_), adj_comb=pct(d_['regimen'] == 'combination/other', d_),
        adj_unk=pct(d_['regimen'] == 'unknown', d_),
        rec=pct(d_['event'], d_), comp=pct(d_['competing'], d_),
        rfs3=km_rate(d_, 3), rfs5=km_rate(d_, 5),
        age75=pct(d_['age'] >= 75, d_), age_med=float(d_.age.median()),
        fu_cens_med=float((d_.loc[~d_.event, 'event_time_days'] / 365.25).median()),
        fu_cens_q=[float(x) for x in (d_.loc[~d_.event, 'event_time_days'] / 365.25).quantile([.25, .75])],
        adj_by_stage={sg: f"{int((d_[d_.pStage15版 == sg].adjuvant == 1).sum())}/{len(d_[d_.pStage15版 == sg])}"
                      for sg in ['IIA', 'IIB', 'IIIA', 'IIIB', 'IIIC']})
    # reverse KM median follow-up
    rk = KaplanMeierFitter().fit(d_.event_time_days / 365.25, ~d_.event)
    T[sp]['fu_revkm_med'] = float(rk.median_survival_time_)
# p-values
def pchi(m):
    ct = pd.crosstab(allc['split'], m); return float(stats.chi2_contingency(ct)[1]) if ct.shape == (2, 2) else np.nan
T['p'] = dict(lap=float(stats.fisher_exact(pd.crosstab(allc.split, allc['laparoscopic'] == 1))[1]),
              TG=pchi(allc['TG'] == 1), D2=pchi(allc['D2'] == 1),
              G3=pchi(allc['Grade≧3'] == 1), adj=pchi(allc['adjuvant'] == 1),
              comb=pchi(allc['regimen'] == 'combination/other'),
              comp=pchi(allc['competing']), rec=pchi(allc['event']),
              age75=pchi(allc['age'] >= 75),
              asa3=pchi(allc['ASA-PS'].astype(str).str.startswith(('3', '4'))),
              lvi=float(stats.mannwhitneyu(dev.ly_ord, val.ly_ord)[1]),
              ly0=pchi(allc['ly_ord'] == 0))
T['ly0'] = {sp: pct(d_['ly_ord'] == 0, d_) for sp, d_ in allc.groupby('split')}
T['rfs_logrank_p'] = float(logrank_test(dev.event_time_days, val.event_time_days, dev.event, val.event).p_value)
R['table_extra'] = T
print(json.dumps(T, indent=1, default=str))

# ------------------------------------------------------------------ save
pickle.dump(dict(R=R, dev=dev, val=val, comb=comb), open('revision_results.pkl', 'wb'))
json.dump(R, open('revision_results.json', 'w'), indent=1, default=str)
print('DONE')
