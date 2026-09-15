"""
run_analysis.py
================
Complete analysis pipeline for the gc_recurrence_rsf study.

Runs in order:
  1. Data preparation (MICE, temporal split)
  2. RSF + Cox model fitting
  3. C-index, td-AUC with 1,000 bootstrap 95% CIs
  4. Calibration (3yr, 5yr)
  5. Decision Curve Analysis (DCA), NRI, IDI
  6. Intra-stage heterogeneity (violin + KM)
  7. Permutation importance + SHAP proxy
  8. Save all figures to figures/

Usage
-----
    python run_analysis.py
"""

import sys
import time
import warnings
import os
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from lifelines import KaplanMeierFitter

from config import (SEED, PREDICTORS, HZ_YEARS, BOOTSTRAP_N,
                    STAGE_GROUPS, COL, COL_RISK, IQR_THRESHOLD_PP)
from prepare_data import load_and_prepare, surv_array
from models import (fit_rsf, fit_cox, predict_cox_scores, pred_ci,
                    compute_cindex, bootstrap_metrics, ci95,
                    continuous_nri_idi, intra_stage_analysis,
                    get_permutation_importance, get_shap_values)

np.random.seed(SEED)
DAYS    = {y: y * 365.25 for y in HZ_YEARS}
OUTDIR  = 'figures'
os.makedirs(OUTDIR, exist_ok=True)

PRED_LABELS = {
    'age':         'Age (yr)',
    'Alb':         'Albumin (g/dL)',
    'CRP':         'CRP (mg/dL)',
    'CEA':         'CEA (ng/mL)',
    'CA19_9':      'CA19-9 (U/mL)',
    'pT_ord':      'pT (ordinal)',
    'pN_num':      'Positive LN count',
    'size':        'Tumour size (mm)',
    'ly_ord':      'LVI grade (0-3)',
    'adjuvant_s1': 'S-1 adjuvant (ACTS-GC)',
}

plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.size':          11,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'figure.dpi':         150,
})

STAGE_ORDER = ['Stage II', 'Stage IIIA', 'Stage IIIB', 'Stage IIIC']


# ══════════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1  Data preparation")
    print("=" * 60)
    dev, val, imputer = load_and_prepare()
    X_dev = dev[PREDICTORS].values.astype(float)
    X_val = val[PREDICTORS].values.astype(float)
    y_dev = surv_array(dev)
    y_val = surv_array(val)
    print(f"  Dev: n={len(dev)}  events={dev['event'].sum()}  "
          f"EPV={dev['event'].sum()/len(PREDICTORS):.1f}")
    print(f"  Val: n={len(val)}  events={val['event'].sum()}")

    # ── 2. Models ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2  Model fitting")
    print("=" * 60)
    print("  Fitting RSF (n_estimators=500) ...")
    rsf = fit_rsf(X_dev, y_dev)
    print("  Fitting Cox (pStage-only) ...")
    cph, sc_dev = fit_cox(dev)
    sc_val = predict_cox_scores(cph, val)

    # ── 3. Discrimination ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3  C-index and td-AUC")
    print("=" * 60)
    r_dev = rsf.predict(X_dev)
    r_val = rsf.predict(X_val)
    ev_d, t_d = dev['event'].values, dev['event_time_days'].values
    ev_v, t_v = val['event'].values, val['event_time_days'].values

    c_rsf_d = compute_cindex(ev_d, t_d, r_dev)
    c_rsf_v = compute_cindex(ev_v, t_v, r_val)
    c_cox_d = compute_cindex(ev_d, t_d, sc_dev)
    c_cox_v = compute_cindex(ev_v, t_v, sc_val)

    t_max   = min(t_v.max(), t_d.max()) * 0.98
    e_times = np.array([DAYS[y] for y in HZ_YEARS if DAYS[y] < t_max])
    auc_rsf, m_rsf = cumulative_dynamic_auc(y_dev, y_val, r_val,  e_times)
    auc_cox, m_cox = cumulative_dynamic_auc(y_dev, y_val, sc_val, e_times)

    print(f"  C-index  RSF : Dev={c_rsf_d:.3f}  Val={c_rsf_v:.3f}")
    print(f"  C-index  Cox : Dev={c_cox_d:.3f}  Val={c_cox_v:.3f}")
    for t2, ar, ac in zip(e_times, auc_rsf, auc_cox):
        print(f"  td-AUC {t2/365.25:.0f}yr  RSF={ar:.3f}  Cox={ac:.3f}")

    print("  Bootstrap CIs (1,000 iterations) ...")
    bs = bootstrap_metrics(y_dev, y_val, r_val, sc_val, e_times)
    print(f"  C-index RSF val: {c_rsf_v:.3f} ({ci95(bs['c_rsf'])[0]:.3f}-"
          f"{ci95(bs['c_rsf'])[1]:.3f})")

    # ── 4. Predicted CI at 3yr / within-stage groups ──────────────────────────
    ci3_dev = pred_ci(rsf, X_dev, DAYS[3])
    ci3_val = pred_ci(rsf, X_val, DAYS[3])
    dev = dev.copy(); dev['ci3'] = ci3_dev
    val = val.copy(); val['ci3'] = ci3_val

    thresholds = {}
    for sg in STAGE_ORDER:
        med = dev.loc[dev['stage_group'] == sg, 'ci3'].median()
        thresholds[sg] = med
        for df_ in (dev, val):
            mask = df_['stage_group'] == sg
            df_.loc[mask, 'risk_group'] = np.where(
                df_.loc[mask, 'ci3'] >= med, 'High', 'Low')

    all_data = pd.concat([dev.assign(split='Dev'),
                           val.assign(split='Val')], ignore_index=True)

    # ── 5. DCA ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5  DCA, NRI, IDI")
    print("=" * 60)
    thresholds_dca = np.linspace(0.05, 0.70, 40)
    stage_risk = {'IIA': 0.153, 'IIB': 0.179, 'IIIA': 0.370,
                  'IIIB': 0.379, 'IIIC': 0.696}
    stage_ci3  = all_data['pStage15版'].map(stage_risk).values
    pred_new   = all_data['ci3'].values
    ev_all     = all_data['event'].values.astype(bool)
    t_all      = all_data['event_time_days'].values

    def net_benefit_km(pred, ev, t, t_horizon, pts):
        from lifelines import KaplanMeierFitter
        n = len(pred)
        nb_model, nb_all = [], []
        for p_t in pts:
            treat = pred >= p_t
            if treat.sum() == 0:
                nb_model.append(0.0)
            else:
                kmf = KaplanMeierFitter()
                kmf.fit(t[treat] / 365.25, event_observed=ev[treat])
                sv  = kmf.survival_function_at_times([t_horizon / 365.25]).values
                obs = 1.0 - (sv[0] if len(sv) > 0 else 0.0)
                tp  = obs * treat.sum() / n
                fp  = (1 - obs) * treat.sum() / n
                nb_model.append(tp - fp * p_t / (1 - p_t))
            kmf2 = KaplanMeierFitter()
            kmf2.fit(t / 365.25, event_observed=ev)
            sv2  = kmf2.survival_function_at_times([t_horizon / 365.25]).values
            pop  = 1.0 - (sv2[0] if len(sv2) > 0 else 0.0)
            nb_all.append(pop - (1 - pop) * p_t / (1 - p_t))
        return np.array(nb_model), np.array(nb_all)

    nb_rsf,   nb_all   = net_benefit_km(pred_new,  ev_all, t_all,
                                         DAYS[3], thresholds_dca)
    nb_stage, _        = net_benefit_km(stage_ci3, ev_all, t_all,
                                         DAYS[3], thresholds_dca)

    # NRI / IDI
    nri_res = continuous_nri_idi(pred_new, stage_ci3, ev_all)
    rng = np.random.default_rng(SEED)
    n_all = len(pred_new)
    nri_b, idi_b = [], []
    for _ in range(BOOTSTRAP_N):
        idx2 = rng.integers(0, n_all, n_all)
        r    = continuous_nri_idi(pred_new[idx2], stage_ci3[idx2], ev_all[idx2])
        nri_b.append(r['nri']); idi_b.append(r['idi'])
    nri_lo, nri_hi = ci95(nri_b)
    idi_lo, idi_hi = ci95(idi_b)
    print(f"  NRI: {nri_res['nri']:.3f} ({nri_lo:.3f}-{nri_hi:.3f})")
    print(f"  IDI: {nri_res['idi']:.3f} ({idi_lo:.3f}-{idi_hi:.3f})")

    # ── 6. Intra-stage heterogeneity ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6  Intra-stage heterogeneity")
    print("=" * 60)
    ish = intra_stage_analysis(all_data, thresholds)
    print(ish[['stage_group', 'N', 'events', 'IQR_pp', 'CV',
               'low_events', 'high_events', 'logrank_p']].to_string(index=False))

    # ── 7. Variable importance + SHAP ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7  Variable importance + SHAP proxy")
    print("=" * 60)
    print("  Permutation importance (30 repeats) ...")
    pi_df = get_permutation_importance(rsf, X_dev, y_dev)
    print(pi_df[['feature', 'mean']].to_string(index=False))

    print("  SHAP (GBC proxy) ...")
    shap_vals = get_shap_values(X_dev, dev['event'].values.astype(int))

    # ── 8. Figures ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 8  Generating figures")
    print("=" * 60)

    # -- Fig 1: C-index / td-AUC forest plot --
    fig, ax = plt.subplots(figsize=(9, 4))
    metrics = [
        ('C-index (val)', c_rsf_v, *ci95(bs['c_rsf']),
                          c_cox_v, *ci95(bs['c_cox'])),
        ('td-AUC 3yr (val)', auc_rsf[0], *ci95(bs['auc_rsf_3']),
                              auc_cox[0], *ci95(bs['auc_cox_3'])),
        ('td-AUC 5yr (val)', auc_rsf[1], *ci95(bs['auc_rsf_5']),
                              auc_cox[1], *ci95(bs['auc_cox_5'])),
    ]
    for yi, (lbl, r_pt, r_lo, r_hi, c_pt, c_lo, c_hi) in enumerate(metrics):
        ax.errorbar(r_pt, yi + 0.15, xerr=[[r_pt - r_lo], [r_hi - r_pt]],
                    fmt='o', color='#1565C0', ms=8, lw=2, capsize=4,
                    label='RSF' if yi == 0 else '')
        ax.errorbar(c_pt, yi - 0.15, xerr=[[c_pt - c_lo], [c_hi - c_pt]],
                    fmt='s', color='#B71C1C', ms=8, lw=2, capsize=4,
                    label='Cox-Stage' if yi == 0 else '')
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m[0] for m in metrics], fontsize=11)
    ax.set_xlim(0.55, 0.90)
    ax.axvline(0.7, color='grey', lw=0.8, ls=':', alpha=0.5)
    ax.set_xlabel("Performance metric (point estimate ± 95% bootstrap CI)", fontsize=10)
    ax.set_title('RSF vs pStage-only Cox — Validation Performance\n'
                 f'(n=156, events=51, 1,000 bootstrap iterations)', fontsize=11)
    ax.legend(fontsize=10, loc='lower right')
    plt.tight_layout()
    fig.savefig(f'{OUTDIR}/fig1_cindex_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Fig 1 saved")

    # -- Fig 2: Calibration 3yr and 5yr --
    ci5_dev = pred_ci(rsf, X_dev, DAYS[5])
    ci5_val = pred_ci(rsf, X_val, DAYS[5])
    dev = dev.copy(); dev['ci5'] = ci5_dev
    val = val.copy(); val['ci5'] = ci5_val

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for row, yr in enumerate([3, 5]):
        for col, (sname, df_, ci_arr) in enumerate([
            ('Development (2008-2015)',      dev, dev[f'ci{yr}'].values),
            ('Temporal Validation (2016-2020)', val, val[f'ci{yr}'].values),
        ]):
            ax = axes[row][col]
            obs_l, pred_l = [], []
            for lo_q, hi_q in zip([0, 20, 40, 60, 80], [20, 40, 60, 80, 100]):
                lo_v = np.percentile(ci_arr, lo_q)
                hi_v = np.percentile(ci_arr, hi_q)
                mask = (ci_arr >= lo_v) & (ci_arr <= hi_v)
                s    = df_[mask]
                if mask.sum() < 5:
                    continue
                kmf = KaplanMeierFitter()
                kmf.fit(s['event_time_days'] / 365.25, s['event'])
                sv  = kmf.survival_function_at_times([yr]).values
                if len(sv) == 0 or np.isnan(sv[0]):
                    continue
                obs_l.append(float(1 - sv[0]))
                pred_l.append(float(ci_arr[mask].mean()))
            ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Ideal')
            ax.scatter(pred_l, obs_l, s=70, color='#6A1B9A', zorder=5, label='Quintile')
            for px, ox in zip(pred_l, obs_l):
                ax.plot([px, px], [px, ox], 'grey', lw=0.8, alpha=0.5)
            ax.set_xlim(0, 0.95); ax.set_ylim(0, 0.95)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
            ax.set_xlabel(f'Predicted {yr}-yr recurrence', fontsize=10)
            ax.set_ylabel(f'Observed {yr}-yr recurrence (KM)', fontsize=10)
            ax.set_title(f'{yr}-Year Horizon — {sname}', fontsize=10)
            ax.legend(fontsize=9)
    plt.suptitle('RSF Model Calibration', fontsize=13)
    plt.tight_layout()
    fig.savefig(f'{OUTDIR}/fig2_calibration_3yr5yr.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Fig 2 saved")

    # -- Fig 3: DCA --
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds_dca, nb_rsf,   color='#1565C0', lw=2.5, label='RSF model')
    ax.plot(thresholds_dca, nb_stage, color='#FF9800', lw=2.0, ls='--',
            label='pStage-only (observed rate)')
    ax.plot(thresholds_dca, nb_all,   color='#616161', lw=1.5, ls=':',  label='Treat all')
    ax.axhline(0, color='k', lw=0.8, label='Treat none')
    ax.fill_between(thresholds_dca, nb_rsf, nb_stage,
                    where=nb_rsf > nb_stage, alpha=0.12, color='#1565C0')
    ax.set_xlim(0.05, 0.70); ax.set_ylim(-0.05, 0.30)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_xlabel('Threshold probability (3-year recurrence)', fontsize=11)
    ax.set_ylabel('Net benefit', fontsize=11)
    ax.set_title('Decision Curve Analysis — 3-Year Recurrence\n'
                 '(pStage IIA-IIIC, ACTS-GC era; n=486)', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    fig.savefig(f'{OUTDIR}/fig3_dca.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Fig 3 saved")

    # -- Fig 4: Intra-stage heterogeneity (violin + KM 2x2) --
    fig = plt.figure(figsize=(14, 12))
    from matplotlib.gridspec import GridSpec
    gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    # 4a: violin (top-left, wider)
    ax_v = fig.add_subplot(gs[0, :])
    data_violin = [all_data.loc[all_data['stage_group'] == sg, 'ci3'].values
                   for sg in STAGE_ORDER]
    vp = ax_v.violinplot(data_violin, positions=range(4),
                          showmedians=True, showextrema=False)
    for pc, sg in zip(vp['bodies'], STAGE_ORDER):
        pc.set_facecolor(COL[sg]); pc.set_alpha(0.72)
    vp['cmedians'].set_color('k'); vp['cmedians'].set_linewidth(2.5)
    for i, sg in enumerate(STAGE_ORDER):
        s   = all_data[all_data['stage_group'] == sg]['ci3']
        iqr = s.quantile(0.75) - s.quantile(0.25)
        ax_v.vlines(i, s.quantile(0.25), s.quantile(0.75),
                    lw=7, color=COL[sg], alpha=0.45, zorder=3)
        ax_v.text(i, s.quantile(0.75) + 0.025,
                  f'IQR={iqr*100:.1f} pp', ha='center', fontsize=9)
    ax_v.set_xticks(range(4))
    ax_v.set_xticklabels(STAGE_ORDER, fontsize=12)
    ax_v.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax_v.set_ylabel('Predicted 3-yr cumulative recurrence', fontsize=11)
    ax_v.set_title('Intra-Stage Heterogeneity of Predicted Recurrence Risk\n'
                   '(RSF; n=486; bar=IQR)', fontsize=12)

    # 4b-e: KM per stage
    for ax_idx, sg in enumerate(STAGE_ORDER):
        ax = fig.add_subplot(gs[1, ax_idx % 2] if ax_idx < 2 else None)
        if ax is None:
            break
    # Use a flat 2x2 approach for KM panels
    km_axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for ax_km, sg in zip(km_axes, STAGE_ORDER[2:]):   # IIIA and IIIB as example
        pass
    # Redo as 4-panel KM
    fig2, km_axes2 = plt.subplots(2, 2, figsize=(12, 10))
    lr_results = {}
    for ax_km, sg in zip(km_axes2.flatten(), STAGE_ORDER):
        s  = all_data[all_data['stage_group'] == sg]
        lr = ish.loc[ish['stage_group'] == sg, 'logrank_p'].values[0]
        lr_results[sg] = lr
        p_str = 'p<0.001' if lr < 0.001 else f'p={lr:.3f}'
        for grp, col, ls in [('Low', COL_RISK['Low'], '-'),
                              ('High', COL_RISK['High'], '--')]:
            df_ = s[s['risk_group'] == grp]
            kmf = KaplanMeierFitter()
            kmf.fit(df_['event_time_days'] / 365.25, df_['event'],
                    label=f'{grp} risk (n={len(df_)}, ev={int(df_["event"].sum())})')
            kmf.plot_survival_function(ax=ax_km, ci_show=True, color=col, linestyle=ls)
        ax_km.set_title(f'{sg}   {p_str}', fontsize=13, fontweight='bold')
        ax_km.set_xlabel('Time (years)', fontsize=10)
        ax_km.set_ylabel('RFS probability', fontsize=10)
        ax_km.set_xlim(0, 8); ax_km.set_ylim(0, 1)
        for xv in [3, 5]:
            ax_km.axvline(xv, color='grey', lw=0.8, ls=':', alpha=0.6)
        ax_km.legend(fontsize=9, loc='lower left')
        ax_km.text(0.98, 0.96, f'Threshold: {thresholds[sg]:.1%}',
                   transform=ax_km.transAxes, ha='right', va='top',
                   fontsize=8.5, color='grey')
    fig2.suptitle('Within-Stage Risk Stratification by RSF-Predicted 3-Year Risk\n'
                  '(ACTS-GC era 2008-2020; n=486)', fontsize=12, y=1.01)
    plt.tight_layout()
    fig2.savefig(f'{OUTDIR}/fig4_intra_stage_heterogeneity.png',
                 dpi=200, bbox_inches='tight')
    plt.close(fig); plt.close(fig2)
    print("  Fig 4 saved")

    # -- Fig 5: SHAP beeswarm + permutation importance --
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # SHAP beeswarm (left)
    feat_order = np.argsort(np.abs(shap_vals).mean(0))
    labels_ord = [PRED_LABELS[PREDICTORS[fi]] for fi in feat_order]
    for yi, fi in enumerate(feat_order):
        sv = shap_vals[:, fi]; xv = X_dev[:, fi]
        xv_n = (xv - xv.min()) / (xv.max() - xv.min() + 1e-9)
        colors = plt.cm.coolwarm(xv_n)
        ax1.scatter(sv, np.full_like(sv, yi) + np.random.uniform(-0.12, 0.12, len(sv)),
                    c=colors, s=14, alpha=0.7, linewidths=0)
    ax1.axvline(0, color='k', lw=0.8)
    ax1.set_yticks(range(len(PREDICTORS)))
    ax1.set_yticklabels(labels_ord, fontsize=10)
    ax1.set_xlabel('SHAP value (impact on recurrence risk)', fontsize=10)
    ax1.set_title('SHAP Summary\n(GBC proxy; direction of effect)', fontsize=11)
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax1, shrink=0.4, pad=0.01)
    cb.set_ticks([0, 1]); cb.set_ticklabels(['Low', 'High'], fontsize=8)

    # Permutation importance (right)
    pi_s = pi_df.sort_values('mean')
    ax2.barh([PRED_LABELS[f] for f in pi_s['feature']], pi_s['mean'],
             xerr=pi_s['std'], color='#E53935', ecolor='grey', capsize=3, alpha=0.85)
    ax2.axvline(0, color='k', lw=0.8)
    ax2.set_xlabel('Delta C-index (permutation importance, RSF)', fontsize=10)
    ax2.set_title('Variable Importance\n(RSF; n=30 repeats)', fontsize=11)
    plt.suptitle('Model Interpretability', fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(f'{OUTDIR}/fig5_shap_permutation_importance.png',
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  Fig 5 saved")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"""
Cohort
  Total N           : {len(all_data)}
  Development       : {len(dev)} (events={dev['event'].sum()})
  Temporal validation: {len(val)} (events={val['event'].sum()})

Performance (temporal validation, 95% bootstrap CI)
  C-index  RSF : {c_rsf_v:.3f} ({ci95(bs['c_rsf'])[0]:.3f}-{ci95(bs['c_rsf'])[1]:.3f})
  C-index  Cox : {c_cox_v:.3f} ({ci95(bs['c_cox'])[0]:.3f}-{ci95(bs['c_cox'])[1]:.3f})
  td-AUC 3yr RSF: {auc_rsf[0]:.3f}   Cox: {auc_cox[0]:.3f}
  td-AUC 5yr RSF: {auc_rsf[1]:.3f}   Cox: {auc_cox[1]:.3f}
  NRI : {nri_res['nri']:.3f} ({nri_lo:.3f}-{nri_hi:.3f})
  IDI : {nri_res['idi']:.3f} ({idi_lo:.3f}-{idi_hi:.3f})

Intra-stage heterogeneity (combined cohort)
{ish[['stage_group','IQR_pp','CV','logrank_p']].to_string(index=False)}

All figures saved to ./{OUTDIR}/
Elapsed: {elapsed:.0f} s
""")


if __name__ == '__main__':
    # Bootstrapping auc_rsf/auc_cox keys need to be split by year
    # Monkey-patch bootstrap_metrics to expose per-year lists
    _orig_bm = bootstrap_metrics

    def _patched_bm(y_dev, y_val, r_val, sc_val, e_times, n=BOOTSTRAP_N):
        raw = _orig_bm(y_dev, y_val, r_val, sc_val, e_times, n)
        # Rename for main() convenience
        for i, yr in enumerate(HZ_YEARS):
            raw[f'auc_rsf_{yr}'] = raw[f'auc_rsf_{yr}']
            raw[f'auc_cox_{yr}'] = raw[f'auc_cox_{yr}']
        return raw

    import models
    models.bootstrap_metrics = _patched_bm

    main()
