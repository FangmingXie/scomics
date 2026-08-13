"""Top-5 CCA spectrum statistics per IT subclass — canonical r, permutation z, variance explained.

One row of four line panels summarizing the cross-species CCA (script 16) for every IT subclass
(L2/3, L4, L5IT, L6IT), so the strength and subspace coverage of the top canonical axes can be
compared across layers and read as they decay across components:

  * r        — canonical correlation of each component (16's spectrum).
  * z        — gene-label permutation z of each component (16's spectrum).
  * var(human), var(mouse) — the fraction of that species' Gate-A loading subspace that the
                 canonical axis captures (see below).

x-axis is the component index CCA1..min(5, k); one colored line per subclass. L4 and L6IT have
only 4 canonical components (min gate size), so their lines stop at CCA4.

VARIANCE EXPLAINED. A canonical axis is a unit direction w in a species' Gate-A VX loading space.
The canonical variate is the projection of the centered, shared-ortholog loading block Lc onto w;
the fraction of the subspace's total variance it captures is

    var_explained = Var(Lc @ w) / trace(Cov(Lc)) = wᵀ Σ w / trace(Σ),   Σ = Covₒₚₜ(Lc).

Computed separately for human and mouse from 16's unit-norm weights. It is in [0,1] but need NOT
sum to 1 across components — canonical directions are not covariance-orthogonal (unlike PCs).

Nothing is refit: r and z are read from 16's spectrum, the weights from 16's weight TSVs, and the
loading covariance from the same 2000-HVG (or 26-extended) loadings 16 used. A per-component
cross-check asserts |corr(Lc_h·wh, Lc_m·wm)| matches the spectrum r, tying weights + loadings back
to the record.

--universe selects the gene set (mirrors 16/32): hvg_intersect (all four layers; default) |
hvg_union (only L2/3 and L4 have 16's union weights + 26's extended loadings — the rest are
logged and omitted, as in 32).

Reads (per subclass TOKEN; paths switch with UNIVERSE):
  local_data/res/it_evo/16.<TOKEN>_axis_cca_spectrum<SUFFIX>.tsv
  local_data/res/it_evo/16.<TOKEN>_axis_cca_weights_{human,mouse}<SUFFIX>.tsv
  local_data/res/it_evo/02.human_<TOKEN>_varimax_loadings.tsv            (HVG membership)
  local_data/res/it/{19,21,23,25}.cheng22_<TOKEN>_varimax_loadings.tsv   (HVG membership)
  local_data/res/it_evo/26.{human,mouse}_<TOKEN>_varimax_loadings_full.tsv   (hvg_union only)
  data/human_mouse_orthologs.tsv
Outputs:
  local_data/fig/it_evo/33.it_cca_stats<SUFFIX>.pdf
  local_data/res/it_evo/33.it_cca_stats<SUFFIX>.tsv   (tidy: subclass, component, r, z, var_*)
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- config ---
UNIVERSE_SUFFIX = {'hvg_intersect': '', 'hvg_union': '_union'}
_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument('--universe', choices=list(UNIVERSE_SUFFIX), default='hvg_intersect')
UNIVERSE = _parser.parse_args().universe
SUFFIX   = UNIVERSE_SUFFIX[UNIVERSE]

# Gate-A VX sets mirror 16's SUBCLASSES; mouse_loadings is the it/{19,21,23,25} HVG file.
SUBCLASSES = [
    {'token': 'L23',  'label': 'L2/3', 'color': 'C0',
     'human_vx': ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10'],
     'mouse_vx': ['VX1', 'VX2', 'VX5', 'VX7', 'VX8', 'VX9'],
     'mouse_loadings': '19.cheng22_L23_varimax_loadings.tsv'},
    {'token': 'L4',   'label': 'L4', 'color': 'C1',
     'human_vx': ['VX3', 'VX4', 'VX5', 'VX6', 'VX8', 'VX10'],
     'mouse_vx': ['VX1', 'VX4', 'VX5', 'VX6'],
     'mouse_loadings': '21.cheng22_L4_varimax_loadings.tsv'},
    {'token': 'L5IT', 'label': 'L5IT', 'color': 'C2',
     'human_vx': ['VX3', 'VX5', 'VX6', 'VX8', 'VX10'],
     'mouse_vx': ['VX1', 'VX2', 'VX5', 'VX6', 'VX7'],
     'mouse_loadings': '23.cheng22_L5IT_varimax_loadings.tsv'},
    {'token': 'L6IT', 'label': 'L6IT', 'color': 'C3',
     'human_vx': ['VX6', 'VX7', 'VX8', 'VX9'],
     'mouse_vx': ['VX1', 'VX2', 'VX9', 'VX10'],
     'mouse_loadings': '25.cheng22_L6IT_varimax_loadings.tsv'},
]
TOP_N       = 5           # plot the first min(TOP_N, k) canonical components
R_MATCH_TOL = 1e-6        # |corr(projected loadings)| vs spectrum r (weights+loadings cross-check)
# panels: (tidy-column, y-axis label); one line per subclass, x = component index
METRICS = [('r', 'canonical correlation  r'),
           ('z', 'permutation  z'),
           ('var_explained_human', 'human subspace variance fraction'),
           ('var_explained_mouse', 'mouse subspace variance fraction')]

# --- file paths ---
RES_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
IT_RES_DIR   = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
FIG_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')
IN_ORTHOLOGS = os.path.join(PROJECT_ROOT, 'data', 'human_mouse_orthologs.tsv')
OUT_PDF      = os.path.join(FIG_DIR, f'33.it_cca_stats{SUFFIX}.pdf')
OUT_TSV      = os.path.join(RES_DIR, f'33.it_cca_stats{SUFFIX}.tsv')

os.makedirs(FIG_DIR, exist_ok=True)

ORTHO = (pd.read_csv(IN_ORTHOLOGS, sep='\t')
         .drop_duplicates('human_symbol').drop_duplicates('mouse_symbol'))


def paths_for(cfg):
    """Input paths this subclass needs for the current UNIVERSE (used for skip-check and load)."""
    token = cfg['token']
    p = {'spectrum': os.path.join(RES_DIR, f'16.{token}_axis_cca_spectrum{SUFFIX}.tsv'),
         'w_h':      os.path.join(RES_DIR, f'16.{token}_axis_cca_weights_human{SUFFIX}.tsv'),
         'w_m':      os.path.join(RES_DIR, f'16.{token}_axis_cca_weights_mouse{SUFFIX}.tsv'),
         'h_hvg':    os.path.join(RES_DIR, f'02.human_{token}_varimax_loadings.tsv'),
         'm_hvg':    os.path.join(IT_RES_DIR, cfg['mouse_loadings'])}
    if UNIVERSE == 'hvg_union':   # the expanded universe reads 26's out-of-sample loadings
        p['h_full'] = os.path.join(RES_DIR, f'26.human_{token}_varimax_loadings_full.tsv')
        p['m_full'] = os.path.join(RES_DIR, f'26.mouse_{token}_varimax_loadings_full.tsv')
    return p


def loading_blocks(cfg, P):
    """Centered, shared-ortholog Gate-A loading blocks (Xc human, Yc mouse). Mirrors 32."""
    hvg_h = pd.read_csv(P['h_hvg'], sep='\t', index_col=0)
    hvg_m = pd.read_csv(P['m_hvg'], sep='\t', index_col=0)
    if UNIVERSE == 'hvg_intersect':
        H, M = hvg_h, hvg_m
    else:
        H = pd.read_csv(P['h_full'], sep='\t', index_col=0)
        M = pd.read_csv(P['m_full'], sep='\t', index_col=0)
    shared = ORTHO[ORTHO['human_symbol'].isin(H.index) & ORTHO['mouse_symbol'].isin(M.index)]
    if UNIVERSE == 'hvg_union':
        shared = shared[shared['human_symbol'].isin(hvg_h.index)
                        | shared['mouse_symbol'].isin(hvg_m.index)]
    shared = shared.reset_index(drop=True)
    X = H.loc[shared['human_symbol'].values, cfg['human_vx']].values
    Y = M.loc[shared['mouse_symbol'].values, cfg['mouse_vx']].values
    return X - X.mean(0), Y - Y.mean(0), len(shared)


def subspace_fraction(Lc, w):
    """Fraction of Lc's total subspace variance captured along unit direction w: wᵀΣw / trace(Σ)."""
    Sigma = Lc.T @ Lc / (len(Lc) - 1)
    return float(w @ Sigma @ w / np.trace(Sigma))


def stats_for(cfg):
    """Tidy per-component stats for one subclass: r, z (from 16) + human/mouse variance explained."""
    P = paths_for(cfg)
    Xc, Yc, n = loading_blocks(cfg, P)

    spec = pd.read_csv(P['spectrum'], sep='\t')
    spec = spec[spec['component'].str.startswith('CCA')].set_index('component')
    wdf_h = pd.read_csv(P['w_h'], sep='\t', index_col=0)
    wdf_m = pd.read_csv(P['w_m'], sep='\t', index_col=0)

    k = min(TOP_N, len(spec))
    rows = []
    for i in range(1, k + 1):
        comp = f'CCA{i}'
        wh = wdf_h.loc[comp, cfg['human_vx']].to_numpy(dtype=float)
        wm = wdf_m.loc[comp, cfg['mouse_vx']].to_numpy(dtype=float)
        r_obs = float(np.corrcoef(Xc @ wh, Yc @ wm)[0, 1])
        r_spec = float(spec.loc[comp, 'r'])
        if not abs(abs(r_obs) - r_spec) < R_MATCH_TOL:
            raise ValueError(
                f'{cfg["label"]} {comp}: |corr(projected loadings)| = {abs(r_obs):.6f} does not '
                f'match spectrum r = {r_spec:.6f} — 16\'s weights/loadings disagree with its '
                f'spectrum for universe {UNIVERSE}')
        rows.append({'subclass': cfg['label'], 'component': comp, 'component_idx': i,
                     'r': r_spec, 'z': float(spec.loc[comp, 'z']),
                     'var_explained_human': subspace_fraction(Xc, wh),
                     'var_explained_mouse': subspace_fraction(Yc, wm)})
    print(f'  {cfg["label"]:5s}: {n} shared orthologs, top {k}/{len(spec)} components')
    return pd.DataFrame(rows)


print(f'--- IT CCA spectrum stats (r, z, variance explained), top {TOP_N} components ---')
print(f'  universe: {UNIVERSE}')

# A layer is drawn only if every input it needs for this universe exists (union lacks L5IT/L6IT).
active, tables = [], []
for cfg in SUBCLASSES:
    missing = [os.path.basename(p) for p in paths_for(cfg).values() if not os.path.exists(p)]
    if missing:
        print(f'  SKIP {cfg["label"]:5s} ({UNIVERSE}): missing {missing}')
        continue
    active.append(cfg)
    tables.append(stats_for(cfg))
if not active:
    raise FileNotFoundError(f'no subclass has all inputs for universe {UNIVERSE}')

tidy = pd.concat(tables, ignore_index=True)
tidy.to_csv(OUT_TSV, sep='\t', index=False)
print(f'  Saved {OUT_TSV}')

plt.rcParams['pdf.fonttype'] = 42
fig, axes = plt.subplots(1, len(METRICS), figsize=(5.0 * len(METRICS), 4.2))
xmax = int(tidy['component_idx'].max())
for ax, (col, ylabel) in zip(axes, METRICS):
    for cfg in active:
        d = tidy[tidy['subclass'] == cfg['label']]
        ax.plot(d['component_idx'], d[col], '-o', color=cfg['color'], lw=1.8, ms=5,
                label=cfg['label'])
    ax.set_xticks(range(1, xmax + 1))
    ax.set_xticklabels([f'CCA{i}' for i in range(1, xmax + 1)])
    ax.set_xlabel('canonical component')
    ax.set_ylabel(ylabel)
    ax.axhline(0, color='0.8', lw=0.6, zorder=0)
    if col in ('r', 'var_explained_human', 'var_explained_mouse'):
        ax.set_ylim(0, 1)
    sns.despine(ax=ax)
axes[0].legend(title='subclass', fontsize=8, frameon=False)
fig.suptitle(f'Cross-species CCA per IT subclass  (universe: {UNIVERSE})', fontsize=12)
fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches='tight', dpi=300)
plt.close(fig)
print(f'  Saved {OUT_PDF}')
print('\nDone.')
