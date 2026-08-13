"""Top-4 CCA spectrum statistics per IT subclass — canonical r, permutation z, variance explained.

One row of four line panels summarizing the cross-species CCA (script 16) for every IT subclass
(L2/3, L4, L5IT, L6IT), so the strength and subspace coverage of the top canonical axes can be
compared across layers and read as they decay across components:

  * r        — canonical correlation of each component (16's spectrum).
  * z        — gene-label permutation z of each component (16's spectrum).
  * var(human), var(mouse) — the fraction of that species' Gate-A VX subspace variance the
                 canonical axis carries at the CELL level (see below).

The first four panels are line plots over the component index CCA1..4, one colored line per
subclass. Every IT subclass has at least 4 canonical components (min gate size), so all are shown
to the same depth (4). A fifth panel is a per-subclass bar of the SUBSPACE OVERLAP.

SUBSPACE OVERLAP. The canonical correlations r_i are the cosines of the principal angles between
the two species' Gate-A gene-loading subspaces, so cos²θ_i = r_i². Their sum Σcos²θ = Σ r_i² is a
single scalar per subclass: 0 = orthogonal subspaces, k = identical. Here it is summed over the
SAME top 4 components shown (recomputed from the spectrum r, not 16's all-component `subspace`
row), so every subclass uses k=4 and the bars are directly comparable. The bar shows the
normalized Σcos²θ / 4 (mean squared canonical correlation), annotated with the raw Σcos²θ.

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
  local_data/res/it_evo/02.human_<TOKEN>_varimax_coords.tsv
  local_data/res/it/{19,21,23,25}.cheng22_<TOKEN>_varimax_coords.tsv
Outputs:
  local_data/fig/it_evo/33.it_cca_stats.pdf
  local_data/res/it_evo/33.it_cca_stats.tsv            (tidy: subclass, component, r, z, var_*)
  local_data/res/it_evo/33.it_cca_subspace_overlap.tsv (subclass, k, sumcos2, sumcos2_frac)
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
     'mouse_coords': '19.cheng22_L23_varimax_coords.tsv'},
    {'token': 'L4',   'label': 'L4', 'color': 'C1',
     'human_vx': ['VX3', 'VX4', 'VX5', 'VX6', 'VX8', 'VX10'],
     'mouse_vx': ['VX1', 'VX4', 'VX5', 'VX6'],
     'mouse_coords': '21.cheng22_L4_varimax_coords.tsv'},
    {'token': 'L5IT', 'label': 'L5IT', 'color': 'C2',
     'human_vx': ['VX3', 'VX5', 'VX6', 'VX8', 'VX10'],
     'mouse_vx': ['VX1', 'VX2', 'VX5', 'VX6', 'VX7'],
     'mouse_coords': '23.cheng22_L5IT_varimax_coords.tsv'},
    {'token': 'L6IT', 'label': 'L6IT', 'color': 'C3',
     'human_vx': ['VX6', 'VX7', 'VX8', 'VX9'],
     'mouse_vx': ['VX1', 'VX2', 'VX9', 'VX10'],
     'mouse_coords': '25.cheng22_L6IT_varimax_coords.tsv'},
]
TOP_N       = 4           # first TOP_N canonical components (every subclass has >= 4)
R_MATCH_TOL = 1e-9        # spectrum r vs the weights' canonical_r (both from 16)
# panels: (tidy-column, y-axis label); one line per subclass, x = component index
METRICS = [('r', 'canonical correlation  r'),
           ('z', 'permutation  z'),
           ('var_explained_human', 'human VX subspace variance fraction'),
           ('var_explained_mouse', 'mouse VX subspace variance fraction')]

# --- file paths ---
RES_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
IT_RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
FIG_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')
OUT_PDF     = os.path.join(FIG_DIR, '33.it_cca_stats.pdf')
OUT_TSV     = os.path.join(RES_DIR, '33.it_cca_stats.tsv')
OUT_TSV_SUB = os.path.join(RES_DIR, '33.it_cca_subspace_overlap.tsv')

os.makedirs(FIG_DIR, exist_ok=True)


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
    ev_m = ev_fraction(os.path.join(IT_RES_DIR, cfg['mouse_coords']),
                       cfg['mouse_vx'], wdf_m, k)

    rows = [{'subclass': cfg['label'], 'component': f'CCA{i}', 'component_idx': i,
             'r': float(spec.loc[f'CCA{i}', 'r']), 'z': float(spec.loc[f'CCA{i}', 'z']),
             'var_explained_human': ev_h[i - 1], 'var_explained_mouse': ev_m[i - 1]}
            for i in range(1, k + 1)]
    # subspace overlap over the SAME top-k components shown: Σcos²θ = Σ r_i² (i=1..k), frac = /k.
    # Recomputed over top k (not 16's all-component `subspace` row) so every subclass uses k=TOP_N
    # and the bars are directly comparable.
    sumcos2 = float(sum(r['r'] ** 2 for r in rows))
    overlap = {'subclass': cfg['label'], 'k': k, 'sumcos2': sumcos2, 'sumcos2_frac': sumcos2 / k}
    print(f'  {cfg["label"]:5s}: top {k}/{len(spec)} components; CCA1 r={rows[0]["r"]:.3f}, '
          f'EV human {ev_h[0]:.0%} / mouse {ev_m[0]:.0%}; '
          f'Σcos²θ(top{k})={sumcos2:.3f} (/{k}={overlap["sumcos2_frac"]:.3f})')
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
fig, axes = plt.subplots(1, len(METRICS) + 1, figsize=(4.6 * (len(METRICS) + 1), 4.2))
xmax = int(tidy['component_idx'].max())
for ax, (col, ylabel) in zip(axes, METRICS):
    for cfg in SUBCLASSES:
        d = tidy[tidy['subclass'] == cfg['label']]
        ax.plot(d['component_idx'], d[col], '-o', color=cfg['color'], lw=1.8, ms=5,
                label=cfg['label'])
    ax.set_xticks(range(1, xmax + 1))
    ax.set_xticklabels([f'CCA{i}' for i in range(1, xmax + 1)])
    ax.set_xlabel('canonical component')
    ax.set_ylabel(ylabel)
    # let each metric autoscale to its own dynamic range rather than pinning r/var to [0, 1];
    # only the z panel gets a 0 reference line (its values approach/cross zero)
    if col == 'z':
        ax.axhline(0, color='0.8', lw=0.6, zorder=0)
    ax.margins(y=0.08)
    sns.despine(ax=ax)

# fifth panel: per-subclass subspace overlap Σcos²θ / k (bar), annotated with raw Σcos²θ
axb = axes[-1]
colors = [cfg['color'] for cfg in SUBCLASSES]
axb.bar(range(len(sub_df)), sub_df['sumcos2_frac'], color=colors, alpha=0.85)
for x, frac, raw in zip(range(len(sub_df)), sub_df['sumcos2_frac'], sub_df['sumcos2']):
    axb.text(x, frac, f'Σ={raw:.2f}', ha='center', va='bottom', fontsize=7)
axb.set_xticks(range(len(sub_df)))
axb.set_xticklabels(sub_df['subclass'])
axb.set_xlabel('subclass')
axb.set_ylabel('subspace overlap  Σcos²θ / k')
axb.margins(y=0.12)
sns.despine(ax=axb)

axes[0].legend(title='subclass', fontsize=8, frameon=False)
fig.suptitle('Cross-species CCA per IT subclass  (universe: hvg_intersect)', fontsize=12)
fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches='tight', dpi=300)
plt.close(fig)
print(f'  Saved {OUT_PDF}')
print('\nDone.')
