"""Per-timepoint archetype-number (NOC) sweep for the yoo25 NR series, in the P21 PC basis.

Scripts 61/61.v2 project the whole NR series (P6..P21) into a PC basis fit on P21 alone, and
script 62 fits archetypes ONCE, on P21, at the NOC inherited from scripts 33-36. Every earlier
age is therefore described against an adult simplex that was never fit to it. That is the right
control for "how far is P8 from the adult structure", but it cannot answer a different question:
does each age have its OWN archetype structure, and how many archetypes does it support?

This script sweeps NOC from 2 to 5 independently at every timepoint and reports the repo's
standard fit/stability metrics, so the number of archetypes becomes a measurement rather than an
assumption. It deliberately does NOT choose: no single criterion reproduces the four curated NOCs
already in the repo (33-36: L2/3=3, L4=3, L5IT=2, L6IT=3) --

    rule                                  L2/3  L4  L5IT  L6IT   matches
    curated                                  3   3     2     3      --
    argmax effEV_mean                        4   3     2     3     3/4
    argmax effEV_rep                         3   4     2     3     3/4
    largest NOC with ARV_mean < 0.05         3   2     2     3     3/4

-- so the existing NOCs were judgment calls, and this script keeps that honest. It writes the
metrics and a diagnostic figure; the NOC per (subclass, age) is then chosen by eye and hard-coded
into script 65. The three candidate rules are printed at the end as SUGGESTIONS only.

All ages are fit in the SHARED P21 basis, so vertices stay comparable across ages and overlay on
the 61.v3 grid. L5IT is excluded: 410-953 cells per age makes it the weakest fit in every
existing sweep.

THE n CONFOUND. Cell counts fall ~3x from P6 to P21 (L2/3 6612 -> 2213), and bootstrap stability
improves with n, so P6 could support a higher NOC for purely statistical reasons. The sweep is
therefore run TWICE: once on all cells (`pass='all'`), and once with every age subsampled to that
subclass's smallest age (`pass='matched_n'`). Only the matched-n pass supports a statement of the
form "early ages have more archetypes" -- a trend that appears only in the `all` pass is an n
effect, not biology.

Procedure per (subclass, age, pass), reusing the repo's machinery rather than reimplementing it:
  1) read 61.v2's coords, subset to the age, take PC1..PC5 as the feature space (NDIM=5,
     DROP_PCS=[], matching 33-36's N_ARCH_PCS and script 62)
  2) SCA(xn, types) -> setup_feature_matrix('data')
  3) common.run_noc_sweep over NOC 2..5 -> EV, bootstrap ARV, per-replicate ARV
  4) repeat the bootstrap ARV N_OUTER times for mean +/- std, as 33-36 do
  5) NREPEATS=10, N_OUTER=3 kept identical to 33-36 so the numbers are comparable

Caveats:
  - ARV_rep rests on the 2-3 Sample replicates each age has, so it is noisy; ARV_mean (bootstrap)
    is the more stable stability metric here.
  - Every age is fit in the P21 basis, so an early age whose dominant variation is not captured by
    adult PCs may look lower-dimensional than it really is. That is the price of comparability.
  - No batch correction anywhere; each age is its own set of replicates.
  - py_pcha needs PYTHONNOUSERSITE=1 -- a user-site numpy 2.2.6 in ~/.local shadows the env's
    pinned 1.26.4, and np.mat was removed in numpy 2.0.

Reads:
  local_data/res/it/61.v2.yoo25_nr_<token>_pc_coords.tsv
Outputs:
  local_data/res/it/64.yoo25_nr_per_age_noc_metrics.tsv
  local_data/fig/it/64.yoo25_nr_per_age_noc_sweep.pdf
  local_data/log/it/64.yoo25_nr_per_age_noc_sweep.log
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from common import run_noc_sweep
from scomics.main import SCA
from scomics.utils import get_relative_variation

# --- file paths ---
RES_DIR     = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
FIG_DIR     = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it')
LOG_DIR     = os.path.join(PROJECT_ROOT, 'local_data', 'log', 'it')
IN_COORDS   = os.path.join(RES_DIR, '61.v2.yoo25_nr_{token}_pc_coords.tsv')
OUT_METRICS = os.path.join(RES_DIR, '64.yoo25_nr_per_age_noc_metrics.tsv')
OUT_PDF     = os.path.join(FIG_DIR, '64.yoo25_nr_per_age_noc_sweep.pdf')
OUT_LOG     = os.path.join(LOG_DIR, '64.yoo25_nr_per_age_noc_sweep.log')

# L5IT excluded: 410-953 cells per age, the weakest fit in every existing sweep.
SUBCLASSES = [
    dict(subclass='L2/3', token='L23'),
    dict(subclass='L4',   token='L4'),
    dict(subclass='L6IT', token='L6IT'),
]

# --- parameters ---
AGE_COL      = 'Age'
SAMPLE_COL   = 'Sample'
TYPE_COL     = 'Type'
AGES         = ['P6', 'P8', 'P10', 'P12', 'P14', 'P17', 'P21']
NDIM         = 5             # PC1..PC5, matching 33-36's N_ARCH_PCS and script 62
DROP_PCS     = []            # use all NDIM dims directly, per 33-36
NOC_GRID     = np.arange(2, 6)   # 2..5, as specified
NREPEATS     = 10            # bootstrap resamples per ARV estimate   (33-36 value)
N_OUTER      = 3             # repeated ARV estimates -> mean +/- std (33-36 value)
PASSES       = ['all', 'matched_n']
SUBSAMPLE_SEED = 0
ARV_GATE     = 0.05          # for the suggestion table only, never a decision here
FIG_PANEL_W  = 2.6
FIG_PANEL_H  = 2.4
DPI          = 300

os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


class _Tee:
    """Mirror stdout/stderr to console and a log file, flushing every write (33-36 idiom)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


_log_fh = open(OUT_LOG, 'w', buffering=1)
sys.stdout = _Tee(sys.__stdout__, _log_fh)
sys.stderr = _Tee(sys.__stderr__, _log_fh)
print(f'Logging to {OUT_LOG}')
print(f'--- per-timepoint NOC sweep {NOC_GRID.min()}-{NOC_GRID.max()} in the P21 PC basis ---')

# ===================== load every subclass's coords up front =====================

pc_cols = [f'PC{i+1}' for i in range(NDIM)]
data = {}
for cfg in SUBCLASSES:
    path = IN_COORDS.format(token=cfg['token'])
    if not os.path.exists(path):
        raise FileNotFoundError(f'{cfg["subclass"]}: missing {path} -- run 61.v2 first')
    coords = pd.read_csv(path, sep='\t', index_col=0)
    ages = coords[AGE_COL].astype(str).values
    missing = sorted(set(AGES) - set(ages))
    if missing:
        raise ValueError(f'{cfg["subclass"]}: no cells at age(s) {missing}')
    n_by_age = {age: int((ages == age).sum()) for age in AGES}
    data[cfg['subclass']] = dict(coords=coords, ages=ages, n_by_age=n_by_age,
                                 n_min=min(n_by_age.values()))
    print(f'  {cfg["subclass"]:5s} ' + '  '.join(f'{a}={n_by_age[a]}' for a in AGES)
          + f'   -> matched_n subsample size {min(n_by_age.values())}')

# ===================== sweep =====================

rows = []
for pass_name in PASSES:
    print(f'\n================ pass: {pass_name} ================')
    for cfg in SUBCLASSES:
        S = cfg['subclass']
        D = data[S]
        for age in AGES:
            m = D['ages'] == age
            idx = np.flatnonzero(m)
            if pass_name == 'matched_n' and len(idx) > D['n_min']:
                # Fixed seed so the subsample is reproducible; the age with the fewest cells is
                # untouched, every other age is cut down to it.
                rng = np.random.default_rng(SUBSAMPLE_SEED)
                idx = np.sort(rng.choice(idx, D['n_min'], replace=False))
            sub = D['coords'].iloc[idx]
            xn = sub[pc_cols].values
            types = sub[TYPE_COL].astype(str).values
            samples = sub[SAMPLE_COL].astype(str).values
            print(f'\n--- {S} {age} [{pass_name}] {len(sub)} cells, '
                  f'{len(np.unique(samples))} replicate(s) ---')

            sca = SCA(xn, types)
            sca.setup_feature_matrix(method='data')
            ev_grid, _, av_rep_grid, _, _, _ = run_noc_sweep(
                sca, NOC_GRID, NDIM, NREPEATS, samples, drop_pcs=DROP_PCS)

            # repeated bootstrap ARV -> mean +/- std, exactly as 33-36 do
            for i, noc in enumerate(NOC_GRID):
                arv_reps = np.array([
                    get_relative_variation(
                        sca.bootstrap_proj_pcha(NDIM, noc, nrepeats=NREPEATS, drop_pcs=DROP_PCS))
                    for _ in range(N_OUTER)])
                effev_reps = ev_grid[i] * (1 - arv_reps)
                rows.append(dict(
                    subclass=S, age=age, **{'pass': pass_name}, n_cells=len(sub), NOC=int(noc),
                    EV=ev_grid[i],
                    ARV_mean=arv_reps.mean(), ARV_std=arv_reps.std(),
                    ARV_rep=av_rep_grid[i],
                    effEV_mean=effev_reps.mean(), effEV_std=effev_reps.std(),
                    effEV_rep=ev_grid[i] * (1 - av_rep_grid[i])))
                print(f'    NOC={noc}  EV={ev_grid[i]:.4f}  '
                      f'ARV={arv_reps.mean():.4f}+/-{arv_reps.std():.4f}  '
                      f'effEV={effev_reps.mean():.4f}+/-{effev_reps.std():.4f}  '
                      f'effEV_rep={ev_grid[i] * (1 - av_rep_grid[i]):.4f}')

metrics = pd.DataFrame(rows)
metrics.to_csv(OUT_METRICS, sep='\t', index=False)
print(f'\nSaved {OUT_METRICS}')

# ===================== figure: subclass rows x age cols, both passes overlaid ==================

plt.rcParams['pdf.fonttype'] = 42             # editable vector text
fig, axes = plt.subplots(len(SUBCLASSES), len(AGES), squeeze=False, sharey=True,
                         figsize=(FIG_PANEL_W * len(AGES), FIG_PANEL_H * len(SUBCLASSES)))

STYLE = {'all': '-', 'matched_n': '--'}
for row, cfg in enumerate(SUBCLASSES):
    S = cfg['subclass']
    for col, age in enumerate(AGES):
        ax = axes[row][col]
        for pass_name in PASSES:
            sub = metrics[(metrics['subclass'] == S) & (metrics['age'] == age)
                          & (metrics['pass'] == pass_name)].sort_values('NOC')
            ls = STYLE[pass_name]
            ax.plot(sub['NOC'], sub['EV'], ls, color='0.6', marker='o', markersize=3,
                    label='EV' if pass_name == 'all' else None)
            ax.plot(sub['NOC'], sub['effEV_mean'], ls, color='C0', marker='o', markersize=3,
                    label='effEV_mean' if pass_name == 'all' else None)
            if pass_name == 'all':
                ax.fill_between(sub['NOC'], sub['effEV_mean'] - sub['effEV_std'],
                                sub['effEV_mean'] + sub['effEV_std'], color='C0', alpha=0.18)
            ax.plot(sub['NOC'], sub['effEV_rep'], ls, color='C2', marker='o', markersize=3,
                    label='effEV_rep' if pass_name == 'all' else None)
        ax.set_xticks(NOC_GRID)
        ax.set_ylim(0, 1)
        ax.axhline(0, color='0.85', linewidth=0.8, zorder=0)
        if row == len(SUBCLASSES) - 1:
            ax.set_xlabel('NOC')
        if col == 0:
            ax.set_ylabel(f'{S}\nmetric', fontweight='bold')
        n_all = data[S]['n_by_age'][age]
        ax.set_title(f'{age}  n={n_all}', fontsize=9)
        if row == 0 and col == 0:
            ax.legend(fontsize=6, frameon=False, loc='lower left')
        sns.despine(ax=ax)

fig.suptitle('Per-timepoint NOC sweep in the P21 PC basis — solid = all cells, '
             'dashed = subsampled to the subclass\'s smallest age (the n control)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT_PDF, bbox_inches='tight', dpi=DPI)
plt.close(fig)
print(f'Saved {OUT_PDF}')

# ===================== suggestion table (NOT a decision) =====================

print('\n--- candidate NOC under three rules. SUGGESTIONS ONLY: no rule reproduces all four ---')
print('--- curated NOCs of scripts 33-36, so read the figure and hard-code 65 by hand.      ---')
print(f'{"subclass":8s} {"age":5s} {"pass":10s} {"effEV_mean":>10s} {"effEV_rep":>9s} '
      f'{"ARV<" + str(ARV_GATE):>9s}')
for pass_name in PASSES:
    for cfg in SUBCLASSES:
        for age in AGES:
            sub = metrics[(metrics['subclass'] == cfg['subclass']) & (metrics['age'] == age)
                          & (metrics['pass'] == pass_name)]
            # ARV_rep can be NaN: with only 2-3 replicates per age, run_noc_sweep needs >=2
            # per-group fits to succeed and reports NaN otherwise. Guard rather than crash.
            a = int(sub.loc[sub['effEV_mean'].idxmax(), 'NOC'])
            b = (int(sub.loc[sub['effEV_rep'].idxmax(), 'NOC'])
                 if sub['effEV_rep'].notna().any() else None)
            gated = sub[sub['ARV_mean'] < ARV_GATE]
            c = int(gated['NOC'].max()) if len(gated) else None
            print(f'{cfg["subclass"]:8s} {age:5s} {pass_name:10s} '
                  f'{a:>10d} {str(b):>9s} {str(c):>9s}')

print('\nDone.')
