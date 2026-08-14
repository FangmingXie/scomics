"""Top-4 CCA spectrum statistics per IT subclass — canonical r, permutation z, variance explained.

A five-panel summary of the cross-species CCA (script 16) for every IT subclass (L2/3, L4, L5IT,
L6IT), designed to make two points obvious:

  (1) SOMETHING IS SHARED IN EVERY SUBCLASS, and it is statistically significant — the rightmost
      panel plots each subclass's observed subspace overlap Σcos²θ against its gene-label
      permutation null (99th percentile band + z); every observed bar towers over its null.
  (2) L2/3 CCA1 & CCA2 STAND OUT — the leftmost panel is a grouped bar of canonical correlation r
      (subclass x CCA1..4); L2/3's first two bars dwarf every other bar.

Panels, left to right:
  * r (grouped bars)  — canonical correlation, subclass x component (16's spectrum).
  * z (line)          — gene-label permutation z per component (16's spectrum).
  * var(human), var(mouse) (lines) — the fraction of that species' Gate-A VX subspace variance the
                 canonical axis carries at the CELL level (see below).
  * subspace overlap (bars) — observed Σcos²θ vs the permutation null (see below).

SUBSPACE OVERLAP. The canonical correlations r_i are the cosines of the principal angles between
the two species' Gate-A gene-loading subspaces, so cos²θ_i = r_i². Their sum Σcos²θ = Σ r_i² is a
single scalar per subclass: 0 = orthogonal subspaces, 4 = identical over these components. Summed
over the SAME top 4 components shown. Its significance is a gene-label permutation null recomputed
here on the ortholog loading blocks (16 persists only an all-component subspace null): the null
shuffles mouse gene labels and recomputes Σcos²θ(top 4); the panel draws the null 99th percentile
as a grey ceiling and annotates z = (observed - null mean) / null sd.

VARIANCE EXPLAINED (identical definition to script 21's `ev_fraction`). A canonical axis is a
unit direction â in a species' Gate-A VX space. Its explained-variance fraction is the CELL-level
variance carried along â as a fraction of the whole Gate-A subspace:

    EV_i = âᵀ Σ â / trace(Σ),   Σ = cov of the cells' Gate-A VX coordinates (scores).

This is CELL variance (from `*_varimax_coords.tsv`), NOT gene-loading variance — varimax orders
components by cell-score variance, so an axis dominated by a high-variance VX (e.g. L2/3 CCA1 is
~VX2) carries a large fraction here. Computed separately for human and mouse.

Nothing is refit: r, z read from 16's spectrum, weights from 16's weight TSVs (a consistency check
asserts the spectrum r equals the weights' `canonical_r`), and Σ from the saved cell coordinates.
Gene universe is the historical hvg_intersect set (16's default), which all four layers have.

Reads (per subclass TOKEN):
  local_data/res/it_evo/16.<TOKEN>_axis_cca_spectrum.tsv
  local_data/res/it_evo/16.<TOKEN>_axis_cca_weights_{human,mouse}.tsv
  local_data/res/it_evo/02.human_<TOKEN>_varimax_{coords,loadings}.tsv
  local_data/res/it/{19,21,23,25}.cheng22_<TOKEN>_varimax_{coords,loadings}.tsv
  data/human_mouse_orthologs.tsv
Outputs:
  local_data/fig/it_evo/33.it_cca_stats.pdf
  local_data/res/it_evo/33.it_cca_stats.tsv            (tidy: subclass, component, r, z, var_*)
  local_data/res/it_evo/33.it_cca_subspace_overlap.tsv (subclass, k, sumcos2, null_p99, z)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- config ---
# Gate-A VX sets mirror 16's SUBCLASSES; coords files carry the cell VX scores (Σ for EV).
SUBCLASSES = [
    {'token': 'L23',  'label': 'L2/3', 'color': 'C0',
     'human_vx': ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10'],
     'mouse_vx': ['VX1', 'VX2', 'VX5', 'VX7', 'VX8', 'VX9'],
     'mouse_stem': '19.cheng22_L23'},
    {'token': 'L4',   'label': 'L4', 'color': 'C1',
     'human_vx': ['VX3', 'VX4', 'VX5', 'VX6', 'VX8', 'VX10'],
     'mouse_vx': ['VX1', 'VX4', 'VX5', 'VX6'],
     'mouse_stem': '21.cheng22_L4'},
    {'token': 'L5IT', 'label': 'L5IT', 'color': 'C2',
     'human_vx': ['VX3', 'VX5', 'VX6', 'VX8', 'VX10'],
     'mouse_vx': ['VX1', 'VX2', 'VX5', 'VX6', 'VX7'],
     'mouse_stem': '23.cheng22_L5IT'},
    {'token': 'L6IT', 'label': 'L6IT', 'color': 'C3',
     'human_vx': ['VX6', 'VX7', 'VX8', 'VX9'],
     'mouse_vx': ['VX1', 'VX2', 'VX9', 'VX10'],
     'mouse_stem': '25.cheng22_L6IT'},
]
TOP_N        = 4          # first TOP_N canonical components (every subclass has >= 4)
R_MATCH_TOL  = 1e-9       # spectrum r vs the weights' canonical_r (both from 16)
OBS_NULL_TOL = 1e-6       # recomputed Σcos²θ vs the spectrum-based sum (loadings match 16's set)
N_PERM       = 2000       # permutations for the overlap null (99th pct + z)
PERM_SEED    = 0
# line panels: (tidy-column, y-axis label); one line per subclass, x = component index
METRICS = [('z', 'permutation  z'),
           ('var_explained_human', 'human VX subspace variance fraction'),
           ('var_explained_mouse', 'mouse VX subspace variance fraction')]

# --- file paths ---
RES_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
IT_RES_DIR   = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
FIG_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')
IN_ORTHOLOGS = os.path.join(PROJECT_ROOT, 'data', 'human_mouse_orthologs.tsv')
OUT_PDF      = os.path.join(FIG_DIR, '33.it_cca_stats.pdf')
OUT_TSV      = os.path.join(RES_DIR, '33.it_cca_stats.tsv')
OUT_TSV_SUB  = os.path.join(RES_DIR, '33.it_cca_subspace_overlap.tsv')

os.makedirs(FIG_DIR, exist_ok=True)

ORTHO = (pd.read_csv(IN_ORTHOLOGS, sep='\t')
         .drop_duplicates('human_symbol').drop_duplicates('mouse_symbol'))


def loading_blocks(cfg):
    """Centered, shared-ortholog Gate-A gene-loading blocks (Xc human, Yc mouse) — the space CCA
    was fit on. hvg_intersect: the intersection of the two 2000-HVG loadings via 1-to-1 orthology."""
    token = cfg['token']
    H = pd.read_csv(os.path.join(RES_DIR, f'02.human_{token}_varimax_loadings.tsv'),
                    sep='\t', index_col=0)
    M = pd.read_csv(os.path.join(IT_RES_DIR, f"{cfg['mouse_stem']}_varimax_loadings.tsv"),
                    sep='\t', index_col=0)
    shared = ORTHO[ORTHO['human_symbol'].isin(H.index)
                   & ORTHO['mouse_symbol'].isin(M.index)].reset_index(drop=True)
    X = H.loc[shared['human_symbol'].values, cfg['human_vx']].values
    Y = M.loc[shared['mouse_symbol'].values, cfg['mouse_vx']].values
    return X - X.mean(0), Y - Y.mean(0)


def overlap_null(Xc, Yc, k):
    """Observed Σcos²θ(top k) and its gene-label permutation null (mirrors 16/21/24b's SVD null).

    The canonical correlations are the singular values of Qxᵀ Qy (orthonormalized loading blocks);
    permuting the mouse gene rows destroys the ortholog pairing while preserving each species'
    subspace. Returns (observed, null array of length N_PERM)."""
    Qx = np.linalg.qr(Xc)[0]
    Qy = np.linalg.qr(Yc)[0]
    obs = float(np.sum(np.clip(np.linalg.svd(Qx.T @ Qy, compute_uv=False), 0, 1)[:k] ** 2))
    rng = np.random.default_rng(PERM_SEED)
    n = Qy.shape[0]
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        sv = np.clip(np.linalg.svd(Qx.T @ Qy[rng.permutation(n)], compute_uv=False), 0, 1)
        null[i] = np.sum(sv[:k] ** 2)
    return obs, null


def ev_fraction(coords_path, vx_cols, wdf, k):
    """Cell-level explained-variance fraction per component: âᵀΣâ / trace(Σ) (mirrors script 21).

    Σ is the covariance of the cells' Gate-A VX coordinates; â is 16's canonical weight vector,
    renormalized to unit length.
    """
    C = pd.read_csv(coords_path, sep='\t', index_col=0)[vx_cols].values
    cov = np.cov(C, rowvar=False)
    total = np.trace(cov)
    out = []
    for i in range(1, k + 1):
        a = wdf.loc[f'CCA{i}', vx_cols].to_numpy(dtype=float)   # object row -> float
        a = a / np.linalg.norm(a)
        out.append(float(a @ cov @ a / total))
    return out


def stats_for(cfg):
    """Tidy per-component stats for one subclass: r, z (16's spectrum) + cell-level EV per species."""
    token = cfg['token']
    spec_full = pd.read_csv(os.path.join(RES_DIR, f'16.{token}_axis_cca_spectrum.tsv'),
                            sep='\t').set_index('component')
    spec = spec_full[spec_full.index.str.startswith('CCA')]
    wdf_h = pd.read_csv(os.path.join(RES_DIR, f'16.{token}_axis_cca_weights_human.tsv'),
                        sep='\t', index_col=0)
    wdf_m = pd.read_csv(os.path.join(RES_DIR, f'16.{token}_axis_cca_weights_mouse.tsv'),
                        sep='\t', index_col=0)

    k = min(TOP_N, len(spec))
    # consistency: 16's spectrum r must equal the weights' canonical_r (both persisted by 16)
    for i in range(1, k + 1):
        if not abs(float(spec.loc[f'CCA{i}', 'r']) - float(wdf_h.loc[f'CCA{i}', 'canonical_r'])) < R_MATCH_TOL:
            raise ValueError(f'{cfg["label"]} CCA{i}: spectrum r != weights canonical_r — '
                             f'16 spectrum and weight TSVs disagree')

    ev_h = ev_fraction(os.path.join(RES_DIR, f'02.human_{token}_varimax_coords.tsv'),
                       cfg['human_vx'], wdf_h, k)
    ev_m = ev_fraction(os.path.join(IT_RES_DIR, f"{cfg['mouse_stem']}_varimax_coords.tsv"),
                       cfg['mouse_vx'], wdf_m, k)

    rows = [{'subclass': cfg['label'], 'component': f'CCA{i}', 'component_idx': i,
             'r': float(spec.loc[f'CCA{i}', 'r']), 'z': float(spec.loc[f'CCA{i}', 'z']),
             'var_explained_human': ev_h[i - 1], 'var_explained_mouse': ev_m[i - 1]}
            for i in range(1, k + 1)]
    # subspace overlap over the SAME top-k components shown: Σcos²θ = Σ r_i² (i=1..k), with a
    # gene-label permutation null recomputed on the ortholog loading blocks (16's persisted
    # subspace null is over ALL components, not top-k). obs must match the spectrum-based sum.
    sumcos2 = float(sum(r['r'] ** 2 for r in rows))
    Xc, Yc = loading_blocks(cfg)
    obs, null = overlap_null(Xc, Yc, k)
    if not abs(obs - sumcos2) < OBS_NULL_TOL:
        raise ValueError(f'{cfg["label"]}: recomputed Σcos²θ {obs:.6f} != spectrum sum {sumcos2:.6f} '
                         f'— loading blocks disagree with 16 for this subclass')
    z = float((obs - null.mean()) / null.std())
    overlap = {'subclass': cfg['label'], 'k': k, 'sumcos2': sumcos2,
               'null_p99': float(np.percentile(null, 99)), 'z': z}
    print(f'  {cfg["label"]:5s}: top {k}/{len(spec)} components; CCA1 r={rows[0]["r"]:.3f}, '
          f'EV human {ev_h[0]:.0%} / mouse {ev_m[0]:.0%}; '
          f'Σcos²θ={sumcos2:.3f} (null 99th {overlap["null_p99"]:.3f}, z={z:.1f})')
    return pd.DataFrame(rows), overlap


print('--- IT CCA spectrum stats (r, z, cell-level variance explained), top 4 components ---')
results = [stats_for(cfg) for cfg in SUBCLASSES]
tidy = pd.concat([r[0] for r in results], ignore_index=True)
sub_df = pd.DataFrame([r[1] for r in results])
tidy.to_csv(OUT_TSV, sep='\t', index=False)
sub_df.to_csv(OUT_TSV_SUB, sep='\t', index=False)
print(f'  Saved {OUT_TSV}')
print(f'  Saved {OUT_TSV_SUB}')

plt.rcParams['pdf.fonttype'] = 42
n_panels = 1 + len(METRICS) + 1               # grouped-r bars | line metrics | overlap vs null
fig, axes = plt.subplots(1, n_panels, figsize=(4.4 * n_panels, 4.2))
ax_r, ax_lines, ax_ov = axes[0], axes[1:1 + len(METRICS)], axes[-1]
labels = [cfg['label'] for cfg in SUBCLASSES]
colors = [cfg['color'] for cfg in SUBCLASSES]
xmax = int(tidy['component_idx'].max())

# --- point 2: grouped canonical-correlation bars, subclass x CCA1..4 (L2/3 CCA1/2 dominate) ---
comp_colors = plt.cm.Blues(np.linspace(0.85, 0.35, TOP_N))
width = 0.8 / TOP_N
for gi, cfg in enumerate(SUBCLASSES):
    d = tidy[tidy['subclass'] == cfg['label']].sort_values('component_idx')
    for ci, (_, row) in enumerate(d.iterrows()):
        ax_r.bar(gi + (ci - (TOP_N - 1) / 2) * width, row['r'], width=width,
                 color=comp_colors[ci], label=f'CCA{ci + 1}' if gi == 0 else None)
ax_r.set_xticks(range(len(SUBCLASSES)))
ax_r.set_xticklabels(labels)
ax_r.set_xlabel('subclass')
ax_r.set_ylabel('canonical correlation  r')
ax_r.legend(title='component', fontsize=7, frameon=False)
sns.despine(ax=ax_r)

# --- line metrics (z, var human, var mouse): one line per subclass, x = component ---
for ax, (col, ylabel) in zip(ax_lines, METRICS):
    for cfg in SUBCLASSES:
        d = tidy[tidy['subclass'] == cfg['label']]
        ax.plot(d['component_idx'], d[col], '-o', color=cfg['color'], lw=1.8, ms=5,
                label=cfg['label'])
    ax.set_xticks(range(1, xmax + 1))
    ax.set_xticklabels([f'CCA{i}' for i in range(1, xmax + 1)])
    ax.set_xlabel('canonical component')
    ax.set_ylabel(ylabel)
    if col == 'z':
        ax.axhline(0, color='0.8', lw=0.6, zorder=0)
    ax.margins(y=0.08)
    sns.despine(ax=ax)
ax_lines[0].legend(title='subclass', fontsize=8, frameon=False)

# --- point 1: observed subspace overlap vs permutation null (grey 99th-pct ceiling) + z ---
xs = np.arange(len(sub_df))
ax_ov.bar(xs, sub_df['null_p99'], width=0.8, color='0.85', label='perm null (99th pct)')
ax_ov.bar(xs, sub_df['sumcos2'], width=0.55, color=colors, alpha=0.95)
for x, raw, z in zip(xs, sub_df['sumcos2'], sub_df['z']):
    ax_ov.text(x, raw, f'z={z:.0f}', ha='center', va='bottom', fontsize=8)
ax_ov.set_xticks(xs)
ax_ov.set_xticklabels(sub_df['subclass'])
ax_ov.set_xlabel('subclass')
ax_ov.set_ylabel('subspace overlap  Σcos²θ (top 4)')
ax_ov.legend(fontsize=7, frameon=False, loc='upper right')
ax_ov.margins(y=0.12)
sns.despine(ax=ax_ov)

fig.suptitle('Cross-species CCA per IT subclass  (universe: hvg_intersect)', fontsize=12)
fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches='tight', dpi=300)
plt.close(fig)
print(f'  Saved {OUT_PDF}')
print('\nDone.')
