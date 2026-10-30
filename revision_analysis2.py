"""
revision_analysis2.py -- supporting analyses for interpretation of the validation-only results.
 a) paired bootstrap difference RSF - Cox (C-index, td-AUC) in validation
 b) stage-stratified Cox in validation (binary high/low, continuous per 10 pp)
 c) 5-fold CV out-of-fold predictions within the development cohort (honest within-stage separation)
 d) competing risk: Aalen-Johansen vs 1-KM at 3 and 5 years
 e) detail of validation stage results
"""
import sys, pickle, json, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sklearn.model_selection import KFold
from lifelines import KaplanMeierFitter, CoxPHFitter, AalenJohansenFitter
from lifelines.statistics import logrank_test
from config import SEED, PREDICTORS, BOOTSTRAP_N
from prepare_data import surv_array
from models import fit_rsf, fit_cox, predict_cox_scores, pred_ci, ci95

D3, D5 = 3 * 365.25, 5 * 365.25
STAGES = ['Stage II', 'Stage IIIA', 'Stage IIIB', 'Stage IIIC']
P = pickle.load(open('revision_results.pkl', 'rb'))
R, dev, val = P['R'], P['dev'], P['val']
X_dev = dev[PREDICTORS].values.astype(float); X_val = val[PREDICTORS].values.astype(float)
y_dev, y_val = surv_array(dev), surv_array(val)
rsf = fit_rsf(X_dev, y_dev); cph, sc_dev = fit_cox(dev); sc_val = predict_cox_scores(cph, val)
r_val = rsf.predict(X_val)
ev, t = val.event.values, val.event_time_days.values
R2 = {}

# a) paired bootstrap RSF - Cox
rng = np.random.default_rng(SEED); n = len(val)
dC, dA3, dA5 = [], [], []
for _ in range(BOOTSTRAP_N):
    ix = rng.integers(0, n, n); yb = Surv.from_arrays(ev[ix], t[ix])
    try:
        c1 = concordance_index_censored(ev[ix], t[ix], r_val[ix])[0]; c0 = concordance_index_censored(ev[ix], t[ix], sc_val[ix])[0]
        a1, _ = cumulative_dynamic_auc(y_dev, yb, r_val[ix], np.array([D3, D5]))
        a0, _ = cumulative_dynamic_auc(y_dev, yb, sc_val[ix], np.array([D3, D5]))
    except Exception:
        continue
    dC.append(c1 - c0); dA3.append(a1[0] - a0[0]); dA5.append(a1[1] - a0[1])
b = R['baseline']
R2['paired'] = dict(dC=b['c_rsf_val'] - b['c_cox_val'], dC_ci=ci95(dC),
                    dA3=b['auc_rsf'][0] - b['auc_cox'][0], dA3_ci=ci95(dA3),
                    dA5=b['auc_rsf'][1] - b['auc_cox'][1], dA5_ci=ci95(dA5),
                    p_dC_le0=float(np.mean(np.array(dC) <= 0)))
# bootstrap CIs for each model's val metrics (to restate Table 2)
print('PAIRED', R2['paired'])

# b) stratified Cox in validation
v = val.copy(); v['hi'] = (v.risk_group == 'High').astype(int); v['x10'] = v.ci3 * 10
for nm, col in [('binary', 'hi'), ('cont10', 'x10')]:
    d = v[['event_time_days', 'event', col, 'stage_group']]
    c = CoxPHFitter().fit(d, 'event_time_days', 'event', strata=['stage_group'])
    s = c.summary.loc[col]
    R2[f'strat_{nm}'] = dict(HR=float(np.exp(s['coef'])), lo=float(s['exp(coef) lower 95%']),
                             hi=float(s['exp(coef) upper 95%']), p=float(s['p']))
    print('STRATIFIED', nm, R2[f'strat_{nm}'])
# stage-stratified, Cox with stage ordinal + x10 (adjusted)
st = {'IIA': 1, 'IIB': 2, 'IIIA': 3, 'IIIB': 4, 'IIIC': 5}
v['pst'] = v['pStage15版'].map(st)
c = CoxPHFitter().fit(v[['event_time_days', 'event', 'pst', 'x10']], 'event_time_days', 'event')
R2['adj_cont10'] = dict(HR=float(np.exp(c.params_['x10'])), lo=float(np.exp(c.confidence_intervals_.loc['x10', '95% lower-bound'])),
                        hi=float(np.exp(c.confidence_intervals_.loc['x10', '95% upper-bound'])), p=float(c.summary.loc['x10', 'p']))
print('ADJ cont10 (stage-ordinal adjusted)', R2['adj_cont10'])

# c) out-of-fold within development
oof = np.zeros(len(dev)); kf = KFold(5, shuffle=True, random_state=SEED)
for tr, te in kf.split(X_dev):
    m = fit_rsf(X_dev[tr], y_dev[tr]); oof[te] = pred_ci(m, X_dev[te], D3)
dv = dev.copy(); dv['oof3'] = oof
thr_oof = {sg: float(dv.loc[dv.stage_group == sg, 'oof3'].median()) for sg in STAGES}
rows = {}
for sg in STAGES:
    s = dv[dv.stage_group == sg]; hi = s.oof3 >= thr_oof[sg]
    lo_, hi_ = s[~hi], s[hi]
    p = logrank_test(lo_.event_time_days, hi_.event_time_days, lo_.event, hi_.event).p_value
    d = s[['event_time_days', 'event']].copy(); d['hi'] = hi.astype(int).values
    c = CoxPHFitter().fit(d, 'event_time_days', 'event')
    d2 = s[['event_time_days', 'event']].copy(); d2['x10'] = s.oof3 * 10
    c2 = CoxPHFitter().fit(d2, 'event_time_days', 'event')
    rows[sg] = dict(N=len(s), ev=int(s.event.sum()), low_n=len(lo_), low_ev=int(lo_.event.sum()), high_n=len(hi_),
                    high_ev=int(hi_.event.sum()), p=float(p), HR=float(np.exp(c.params_['hi'])),
                    IQR=float((s.oof3.quantile(.75) - s.oof3.quantile(.25)) * 100), CV=float(s.oof3.std() / s.oof3.mean()),
                    HR10=float(np.exp(c2.params_['x10'])), HR10_p=float(c2.summary.loc['x10', 'p']),
                    cidx=float(concordance_index_censored(s.event.values, s.event_time_days.values, s.oof3.values)[0]))
R2['oof'] = rows; R2['oof_overall_c'] = float(concordance_index_censored(dv.event.values, dv.event_time_days.values, dv.oof3.values)[0])
print('OOF within-dev, overall C =', R2['oof_overall_c'])
print(pd.DataFrame(rows).T.round(3).to_string())
d = dv[['event_time_days', 'event', 'stage_group']].copy(); d['hi'] = (dv.oof3 >= dv.stage_group.map(thr_oof)).astype(int)
c = CoxPHFitter().fit(d, 'event_time_days', 'event', strata=['stage_group']); s = c.summary.loc['hi']
R2['oof_strat_binary'] = dict(HR=float(np.exp(s['coef'])), lo=float(s['exp(coef) lower 95%']), hi=float(s['exp(coef) upper 95%']), p=float(s['p']))
print('OOF stratified binary', R2['oof_strat_binary'])

# d) competing risk: Aalen-Johansen vs 1-KM
for nm, df in [('dev', dev), ('val', val)]:
    code = np.where(df.event, 1, np.where(df.competing, 2, 0))
    aj = AalenJohansenFitter(calculate_variance=False, seed=SEED).fit(df.event_time_days / 365.25, code, event_of_interest=1)
    cif = aj.cumulative_density_
    f = lambda y: float(cif[cif.index <= y].iloc[-1, 0])
    km = KaplanMeierFitter().fit(df.event_time_days / 365.25, df.event)
    k = lambda y: float(1 - km.survival_function_at_times([y]).values[0])
    R2[f'cif_{nm}'] = dict(cif3=f(3), km3=k(3), cif5=f(5), km5=k(5))
    print('CIF', nm, R2[f'cif_{nm}'])
# mean predicted (1-S) vs AJ CIF in validation
R2['pred_vs_cif_val'] = dict(pred3=float(val.ci3.mean()), pred5=float(val.ci5.mean()))
print('mean predicted val', R2['pred_vs_cif_val'])

# e) val detail
print('VAL DETAIL')
sv = pd.DataFrame(R['stage_val']).T
print(sv[['N', 'events', 'median', 'IQR_pp', 'CV', 'low_n', 'low_ev', 'high_n', 'high_ev', 'logrank_p', 'HR_high', 'HR_lo', 'HR_hi',
          'km3_low', 'km3_high', 'HR10', 'HR10_lo', 'HR10_hi', 'HR10_p', 'cidx', 'cidx_ci']].to_string())
# overall: validation-cohort stage mean predicted/observed
for sg in STAGES:
    s = val[val.stage_group == sg]
    print(sg, 'val mean pred3=%.3f' % s.ci3.mean(), 'KM obs3=%.3f' % (1 - KaplanMeierFitter().fit(s.event_time_days / 365.25, s.event).survival_function_at_times([3]).values[0]))
R['R2'] = R2
pickle.dump(dict(R=R, dev=dev, val=val, comb=P['comb'], dv=dv), open('revision_results2.pkl', 'wb'))
json.dump(R2, open('revision_results2.json', 'w'), indent=1, default=str)
print('DONE2')
