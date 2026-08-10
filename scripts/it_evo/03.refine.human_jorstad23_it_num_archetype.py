"""Archetype number selection — Jorstad23 human L2/3 / L4 / L5 IT / L6 IT (compute).

The l23_evo/08.refine procedure applied to the four IT subclasses. Two-stage structure,
copied exactly: `run_noc_sweep` supplies EV and the per-donor (`ARV_rep`) numbers via
`pcha_on_subset` refits — its second return value (plain-bootstrap ARV) is discarded —
while `ARV_mean`/`ARV_std` come from a separate N_OUTER loop calling
`bootstrap_proj_pcha(..., is_bootstrap=False, downsamp_p=DOWNSAMP_P)` directly, i.e.
20% downsampling without replacement.

G9: this is NOT the estimator the mouse sweeps in it/{20,22,24,26} used — those call
`bootstrap_proj_pcha` with the defaults (true bootstrap) and group by `Sample`, not
`donor_id`. Human and mouse ARV values are therefore not on the same scale; judge each
on its own elbow, never by comparing absolute values across the two.

NOC bound: `proj()` fits `PCA(NDIM+1)` and drops one component, so the fit lives in NDIM
dimensions and a non-degenerate simplex needs `NOC <= NDIM+1 = len(vx_cols)`. The sweep
is capped there per subclass; anything above would be structurally degenerate and its ARV
blow-up would carry no information.

Reads:  local_data/res/it_evo/02.human_<TOKEN>_varimax_coords.tsv
Outputs (per TOKEN):
  local_data/res/it_evo/03.refine.human_<TOKEN>_num_archetype_metrics.tsv
  local_data/res/it_evo/03.refine.human_<TOKEN>_num_archetype_plotdata.pkl
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from common import run_noc_sweep
from scomics.main import SCA
from scomics.utils import get_relative_variation

# --- file paths ---
OUT_RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')

# `vx_cols` chosen at Gate A from 02.human_<TOKEN>_vx_variance_partition.html: keep the VX
# components whose cell_type partial R² dominates donor + source + library_size, then drop
# any whose top-|loading| genes are mitochondrial or ribosomal (those are depth axes even
# when their cell_type R² looks respectable).
SUBCLASSES = [
    # published l23_evo choice; VX1 is libsize (0.702), VX3/4/5 are donor. NOC <= 6.
    {'token': 'L23',  'human_subclass': 'L2/3 IT',
     'vx_cols': ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10']},
    # dropped VX1 (libsize 0.657), VX2 (donor 0.195 and MT-ND* loaded), VX7/VX9 (donor
    # 0.468/0.735). Kept components all load on neuronal identity genes. NOC <= 6.
    {'token': 'L4',   'human_subclass': 'L4 IT',
     'vx_cols': ['VX3', 'VX4', 'VX5', 'VX6', 'VX8', 'VX10']},
    # dropped VX1 (ribo/glycolysis: RPL41, LDHA, LDHB), VX2 (libsize 0.614), VX7/VX9
    # (donor 0.515/0.573), and VX4 — its cell_type R² is 0.285 but its top loadings are
    # MT-ND1/MT-ATP6/MT-CYB, i.e. a depth axis. NOC <= 5.
    {'token': 'L5IT', 'human_subclass': 'L5 IT',
     'vx_cols': ['VX3', 'VX5', 'VX6', 'VX8', 'VX10']},
    # G4: L6 IT has only 2 WithinArea_cluster levels, a near-degenerate one-dummy design
    # that structurally suppresses cell_type R², so these are judged primarily on LOW
    # donor + source + library_size. Nuisance totals: VX7 0.021, VX8 0.059, VX6 0.067,
    # VX9 0.131, then a gap to VX10 0.188 (mostly libsize) and VX1 0.318 (donor + mito).
    # VX9 is the borderline inclusion — donor 0.100 vs cell_type 0.038 — kept because its
    # loadings are ordinary identity/ECM genes (SERPINE2, ADAMTS16, ITGB8) and dropping it
    # would leave NDIM=2. Confirm at Gate B that per-donor vertex clouds still overlap.
    # NOC <= 4.
    {'token': 'L6IT', 'human_subclass': 'L6 IT',
     'vx_cols': ['VX6', 'VX7', 'VX8', 'VX9']},
]

# --- parameters ---
CLUSTER_COL = 'WithinArea_cluster'
GROUP_COL   = 'donor_id'      # the human precedent — l23_evo/08.refine:38
NOC_MIN     = 2
NOC_MAX     = 6
NREPEATS    = 10    # downsample draws per ARV estimate
N_OUTER     = 20    # repeated ARV estimates → mean ± std
DOWNSAMP_P  = 0.2   # fraction of cells per downsample draw (without replacement)

os.makedirs(OUT_RES_DIR, exist_ok=True)

for cfg in SUBCLASSES:
    token    = cfg['token']
    subclass = cfg['human_subclass']
    vx_cols  = cfg['vx_cols']
    ndim     = len(vx_cols) - 1

    in_varimax      = os.path.join(OUT_RES_DIR, f'02.human_{token}_varimax_coords.tsv')
    out_metrics_tsv = os.path.join(
        OUT_RES_DIR, f'03.refine.human_{token}_num_archetype_metrics.tsv')
    out_plotdata    = os.path.join(
        OUT_RES_DIR, f'03.refine.human_{token}_num_archetype_plotdata.pkl')

    # NOC <= NDIM+1 = len(vx_cols); beyond that the simplex is degenerate.
    noc_max = min(NOC_MAX, len(vx_cols))
    print(f'\n{"=" * 70}\n{token} — human {subclass}  '
          f'(NDIM={ndim}, VX={vx_cols})\n{"=" * 70}')
    if noc_max < NOC_MAX:
        print(f'  NOC capped at {noc_max} (= len(vx_cols)); NOC {noc_max + 1}–{NOC_MAX} '
              f'would be structurally degenerate and are not swept.')

    # --- load varimax coords ---
    vx_df  = pd.read_csv(in_varimax, sep='\t', index_col=0)
    xn     = vx_df[vx_cols].values
    types  = vx_df[CLUSTER_COL].values
    groups = vx_df[GROUP_COL].values

    sca = SCA(xn, types)
    sca.setup_feature_matrix(method='data')

    noc_grid = np.arange(NOC_MIN, noc_max + 1)
    print(f'Running NOC sweep {NOC_MIN}–{noc_max}, NDIM={ndim}, NREPEATS={NREPEATS}...')

    ev_grid, _, av_rep_grid, xp_grid, aa_grid, aa_reps_grid = run_noc_sweep(
        sca, noc_grid, ndim, NREPEATS, groups)

    # --- repeated downsample ARV → mean ± std ---
    print(f'Repeating downsample ARV N_OUTER={N_OUTER} times per NOC...')
    arv_mean, arv_std, effev_mean, effev_std = [], [], [], []
    for i, noc in enumerate(noc_grid):
        arv_reps = np.array([
            get_relative_variation(sca.bootstrap_proj_pcha(
                ndim, noc, nrepeats=NREPEATS, is_bootstrap=False, downsamp_p=DOWNSAMP_P))
            for _ in range(N_OUTER)
        ])
        effev_reps = ev_grid[i] * (1 - arv_reps)
        arv_mean.append(arv_reps.mean())
        arv_std.append(arv_reps.std())
        effev_mean.append(effev_reps.mean())
        effev_std.append(effev_reps.std())
        print(f"  NOC={noc}  ARV={arv_reps.mean():.4f}±{arv_reps.std():.4f}"
              f"  effEV={effev_reps.mean():.4f}±{effev_reps.std():.4f}")

    arv_mean, arv_std = np.array(arv_mean), np.array(arv_std)
    effev_mean, effev_std = np.array(effev_mean), np.array(effev_std)
    effev_rep_grid = ev_grid * (1 - av_rep_grid)

    metrics_df = pd.DataFrame({
        'NOC':        noc_grid,
        'EV':         ev_grid,
        'ARV_mean':   arv_mean,
        'ARV_std':    arv_std,
        'ARV_rep':    av_rep_grid,
        'effEV_mean': effev_mean,
        'effEV_std':  effev_std,
        'effEV_rep':  effev_rep_grid,
    })
    metrics_df.to_csv(out_metrics_tsv, sep='\t', index=False)
    print(f'  Saved {out_metrics_tsv}')

    plotdata = {
        'noc_grid':     noc_grid,
        'ndim':         ndim,
        'groups':       groups,
        'xp':           xp_grid[0],
        'aa_reps_grid': aa_reps_grid,
    }
    with open(out_plotdata, 'wb') as f:
        pickle.dump(plotdata, f)
    print(f'  Saved {out_plotdata}')

print('\nDone.')
