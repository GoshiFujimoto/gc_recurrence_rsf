"""Regenerate Fig 3 (validation DCA), Fig 4 (validation-only heterogeneity), Fig 5 (labels) in IJCO format:
119 mm wide, Liberation Sans, >=7 pt lettering at final size, 300 dpi TIFF (RGB, LZW)."""
import sys, pickle, warnings, os
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from lifelines import KaplanMeierFitter
from PIL import Image
from sklearn.inspection import permutation_importance
from config import SEED, PREDICTORS
from prepare_data import surv_array
from models import fit_rsf, get_shap_values

P = pickle.load(open('revision_results2.pkl', 'rb'))
R, dev, val = P['R'], P['dev'], P['val']
OUT = 'figs_out'; os.makedirs(OUT, exist_ok=True)
W = 119 / 25.4
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5, 'axes.linewidth': 0.8,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 100, 'lines.linewidth': 1.4})
STAGES = ['Stage II', 'Stage IIIA', 'Stage IIIB', 'Stage IIIC']
COL = {'Stage II': '#2196F3', 'Stage IIIA': '#FF9800', 'Stage IIIB': '#F44336', 'Stage IIIC': '#9C27B0'}
BLUE, RED = '#1565C0', '#B71C1C'


def save(fig, name):
    fig.savefig(f'{OUT}/{name}.png', dpi=300, facecolor='white')
    im = Image.open(f'{OUT}/{name}.png').convert('RGB')
    im.save(f'{OUT}/{name}.tif', dpi=(300, 300), compression='tiff_lzw')
    print(name, im.size, round(im.size[0] / 300 * 25.4, 1), 'mm wide', round(im.size[1] / 300 * 25.4, 1), 'mm high')
    plt.close(fig)


pct = FuncFormatter(lambda y, _: f'{y:.0%}')

# ---------------------------------------------------------------- Fig 3: validation DCA
d = R['dca_val']; pts = np.array(d['pts'])
fig, ax = plt.subplots(figsize=(W, 3.1))
ax.plot(pts, d['rsf'], color=BLUE, lw=1.8, label='RSF')
ax.plot(pts, d['stage'], color='#E65100', lw=1.5, ls='--', label='pStage-only Cox')
ax.plot(pts, d['all'], color='#616161', lw=1.2, ls=':', label='Treat all')
ax.axhline(0, color='k', lw=0.9, label='Treat none')
ax.set_xlim(0.05, 0.70); ax.set_ylim(-0.05, 0.30)
ax.xaxis.set_major_formatter(pct)
ax.set_xlabel('Threshold probability (3-year recurrence)'); ax.set_ylabel('Net benefit')
ax.legend(frameon=False, loc='upper right')
fig.tight_layout(pad=0.6)
save(fig, 'Fig3')

# ---------------------------------------------------------------- Fig 4
fig = plt.figure(figsize=(W, 7.0))
gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1, 1], hspace=0.78, wspace=0.36, left=0.15, right=0.985, top=0.955, bottom=0.085)
axa = fig.add_subplot(gs[0, :])
data = [val.loc[val.stage_group == sg, 'ci3'].values for sg in STAGES]
vp = axa.violinplot(data, positions=range(4), showmedians=False, showextrema=False, widths=0.8)
rng = np.random.default_rng(SEED)
for i, (pc, sg) in enumerate(zip(vp['bodies'], STAGES)):
    pc.set_facecolor(COL[sg]); pc.set_alpha(0.45); pc.set_edgecolor('none')
    v = data[i]
    axa.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=5, color=COL[sg], alpha=0.9, linewidths=0, zorder=3)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    axa.vlines(i, q1, q3, lw=4.5, color='k', alpha=0.55, zorder=4)
    axa.hlines(med, i - 0.2, i + 0.2, color='k', lw=1.6, zorder=5)
    axa.text(i, min(v.max(), 0.98) + 0.035, f'IQR {(q3 - q1) * 100:.1f} pp', ha='center', va='bottom', fontsize=7.5)
axa.set_xticks(range(4)); axa.set_xticklabels([f'{sg}\n(n = {len(d_)})' for sg, d_ in zip(STAGES, data)])
axa.set_ylim(0, 1.0); axa.yaxis.set_major_formatter(pct)
axa.set_ylabel('Predicted 3-year\nrecurrence probability')
axa.text(-0.17, 1.04, '(a)', transform=axa.transAxes, fontweight='bold', fontsize=9, va='bottom', ha='left')
labels = ['(b)', '(c)', '(d)', '(e)']
for k, sg in enumerate(STAGES):
    ax = fig.add_subplot(gs[1 + k // 2, k % 2])
    s = val[val.stage_group == sg]; sv = R['stage_val'][sg]
    for grp, col, ls in [('Low', BLUE, '-'), ('High', RED, '--')]:
        g = s[s.risk_group == grp]
        km = KaplanMeierFitter().fit(g.event_time_days / 365.25, g.event)
        km.plot_survival_function(ax=ax, ci_show=False, color=col, linestyle=ls, label='_nolegend_', show_censors=False, legend=False)
    for xv in (3, 5):
        ax.axvline(xv, color='grey', lw=0.7, ls=':', alpha=0.7)
    p = sv['logrank_p']; ps = 'p < 0.001' if p < 0.001 else f'p = {p:.2f}'
    ax.set_title(f'{labels[k]} {sg}  (log-rank {ps})', loc='left', fontsize=8, fontweight='bold', pad=13)
    ax.text(0.0, 1.015, f"n = {sv['low_n']} ({sv['low_ev']} events)", color=BLUE, transform=ax.transAxes, fontsize=7, va='bottom', ha='left')
    ax.text(1.0, 1.015, f"n = {sv['high_n']} ({sv['high_ev']} events)", color=RED, transform=ax.transAxes, fontsize=7, va='bottom', ha='right')
    ax.set_xlim(0, 8); ax.set_ylim(0, 1.0); ax.set_xticks([0, 2, 4, 6, 8]); ax.set_yticks([0, .2, .4, .6, .8, 1.0])
    ax.set_xlabel('Time since surgery (years)'); ax.set_ylabel('Recurrence-free probability')
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([0], [0], color=BLUE, ls='-', lw=1.6, label='Low predicted risk'),
                    Line2D([0], [0], color=RED, ls='--', lw=1.6, label='High predicted risk')],
           loc='lower center', ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.55, 0.0))
save(fig, 'Fig4')

# ---------------------------------------------------------------- Fig 5
X_dev = dev[PREDICTORS].values.astype(float); y_dev = surv_array(dev)
rsf = fit_rsf(X_dev, y_dev)
pi = permutation_importance(rsf, X_dev, y_dev, n_repeats=30, random_state=SEED, n_jobs=-1)
shap_vals = get_shap_values(X_dev, dev['event'].values.astype(int))
LAB = {'age': 'Age (yr)', 'Alb': 'Albumin (g/dL)', 'CRP': 'CRP (mg/dL)', 'CEA': 'CEA (ng/mL)', 'CA19_9': 'CA19-9 (U/mL)',
       'pT_ord': 'pT (ordinal)', 'pN_num': 'Positive LN count', 'size': 'Tumor size (mm)', 'ly_ord': 'Lymphatic invasion (0–3)',
       'adjuvant_s1': 'Adjuvant chemotherapy'}
fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 6.9), gridspec_kw={'hspace': 0.42})
order = np.argsort(np.abs(shap_vals).mean(0))
rg = np.random.default_rng(SEED)
for yi, fi in enumerate(order):
    sv = shap_vals[:, fi]; xv = X_dev[:, fi]
    xn = (xv - xv.min()) / (xv.max() - xv.min() + 1e-9)
    a1.scatter(sv, yi + rg.uniform(-0.2, 0.2, len(sv)), c=plt.cm.coolwarm(xn), s=4, alpha=0.75, linewidths=0)
a1.axvline(0, color='k', lw=0.8)
a1.set_yticks(range(10)); a1.set_yticklabels([LAB[PREDICTORS[i]] for i in order])
a1.set_xlabel('SHAP value (impact on recurrence risk)')
a1.text(-0.42, 1.03, '(a)', transform=a1.transAxes, fontweight='bold', fontsize=9, ha='left')
cax = a1.inset_axes([1.02, 0.30, 0.025, 0.40])
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(0, 1)); sm.set_array([])
cb = fig.colorbar(sm, cax=cax); cb.set_ticks([0, 1]); cb.set_ticklabels(['Low', 'High'], fontsize=7); cb.outline.set_linewidth(0.5)
cax.set_title('Feature\nvalue', fontsize=7, pad=3)
imp = pd.DataFrame({'f': PREDICTORS, 'm': pi.importances_mean, 's': pi.importances_std}).sort_values('m')
a2.barh([LAB[f] for f in imp.f], imp.m, xerr=imp.s, color='#3F7FCC', ecolor='grey', capsize=1.5, height=0.62, error_kw={'lw': 0.8})
a2.axvline(0, color='k', lw=0.8)
a2.set_xlabel('Decrease in C-index (permutation importance)')
a2.text(-0.42, 1.03, '(b)', transform=a2.transAxes, fontweight='bold', fontsize=9, ha='left')
fig.subplots_adjust(left=0.40, right=0.90, top=0.965, bottom=0.075)
save(fig, 'Fig5')
print('perm importance (dev):', imp.sort_values('m', ascending=False)[['f', 'm']].round(4).to_string(index=False))
