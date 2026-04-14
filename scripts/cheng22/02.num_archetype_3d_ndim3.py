# Number of archetypes selection tutorial — 2D + 3D interactive version
# Runs PCHA with NDIM=10 and a range of NOC (2-6), computing explained variance
# and archetype relative variation (bootstrap stability) to identify optimal NOC.
# Saves an interactive HTML with 2D (PC1 vs PC2) panels on top and
# 3D (PC1 vs PC2 vs PC3) panels below.

import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import anndata as ad

from scomics.main import SCA
from scomics.utils import norm, get_relative_variation

# infer project root from this script's location
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'data_snrna_v1.h5ad')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig')
FIG_NOC_METRICS = os.path.join(FIG_DIR, '02.num_archetype_3d_metrics_ndim3.png')
FIG_NOC_ARCHETYPES = os.path.join(FIG_DIR, '02.num_archetype_3d_interactive_ndim3.html')

NDIM = 3 
NOC_MIN = 2
NOC_MAX = 6
NREPEATS = 10

# load data
adata = ad.read_h5ad(INPUT_FILE)
print(adata)

x = np.array(adata.X.todense())
depths = adata.obs['n_counts'].values
types = adata.obs['Type'].values

xn = norm(x, depths)
sca = SCA(xn, types)
sca.setup_feature_matrix(method='data')

noc_grid = np.arange(NOC_MIN, NOC_MAX + 1)
ev_grid = []
av_grid = []
xp_grid = []
aa_grid = []

for noc in noc_grid:
    xp, aa, ev = sca.proj_and_pcha(NDIM, noc)
    aa_boots = sca.bootstrap_proj_pcha(NDIM, noc, nrepeats=NREPEATS)
    av = get_relative_variation(aa_boots)
    print(f"NOC={noc}  EV={ev:.4f}  ARV={av:.4f}  effectiveEV={ev*(1-av):.4f}")
    ev_grid.append(ev)
    av_grid.append(av)
    xp_grid.append(xp)
    aa_grid.append(aa)

ev_grid = np.array(ev_grid)
av_grid = np.array(av_grid)

# --- plot: EV, ARV, effective EV vs NOC ---
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(noc_grid, ev_grid, '-o', label='explained variance (EV)')
ax.plot(noc_grid, av_grid, '-o', color='gray', label='archetype relative variation (ARV)')
ax.plot(noc_grid, ev_grid * (1 - av_grid), '-o', label='effective EV')
ax.set_xlabel('Number of archetypes (NOC)')
ax.set_ylabel('Score')
ax.set_ylim([0, 1])
ax.set_xticks(noc_grid)
ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=8)
ax.set_title(f'Archetype selection (NDIM={NDIM})')
sns.despine(ax=ax)
ax.grid(False)

os.makedirs(FIG_DIR, exist_ok=True)
fig.tight_layout()
fig.savefig(FIG_NOC_METRICS, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f"Saved {FIG_NOC_METRICS}")

# --- 3D interactive plot: one subplot per NOC ---
# map cell types to hex colors using matplotlib default color cycle
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
unique_types = np.unique(types)
type_to_color = {t: cycle[i % len(cycle)] for i, t in enumerate(unique_types)}
cell_colors = [type_to_color[t] for t in types]

nrows = len(noc_grid)
subplot_titles = []
for i, noc in enumerate(noc_grid):
    subplot_titles += [
        f"2D  NOC={noc}  EV={ev_grid[i]:.3f}  ARV={av_grid[i]:.3f}",
        f"3D  NOC={noc}  EV={ev_grid[i]:.3f}  ARV={av_grid[i]:.3f}",
    ]
specs = [[{'type': 'xy'}, {'type': 'scene'}]] * nrows

fig3d = make_subplots(rows=nrows, cols=2, specs=specs, subplot_titles=subplot_titles)

for i, noc in enumerate(noc_grid):
    row = i + 1
    xp = xp_grid[i]
    aa = aa_grid[i]

    # --- col 1: 2D scatter (PC1 vs PC2) with closed archetype polygon ---
    fig3d.add_trace(go.Scatter(
        x=xp[:, 0], y=xp[:, 1],
        mode='markers',
        marker=dict(size=2, color=cell_colors, opacity=0.6),
        name='cells' if i == 0 else None,
        showlegend=(i == 0),
        legendgroup='cells',
    ), row=row, col=1)

    fig3d.add_trace(go.Scatter(
        x=aa[0, :], y=aa[1, :],
        mode='markers',
        marker=dict(size=8, color='black', symbol='diamond'),
        name='archetypes' if i == 0 else None,
        showlegend=(i == 0),
        legendgroup='archetypes',
    ), row=row, col=1)

    # closed polygon: visit archetypes in order and close the loop
    px = list(aa[0, :]) + [aa[0, 0]]
    py = list(aa[1, :]) + [aa[1, 0]]
    fig3d.add_trace(go.Scatter(
        x=px, y=py,
        mode='lines',
        line=dict(color='black', width=1.5),
        name='polygon' if i == 0 else None,
        showlegend=(i == 0),
        legendgroup='polygon',
    ), row=row, col=1)

    fig3d.update_xaxes(title_text='PC1', row=row, col=1)
    fig3d.update_yaxes(title_text='PC2', row=row, col=1)

    # --- col 2: 3D scatter (PC1 vs PC2 vs PC3) with all-pairs edges ---
    fig3d.add_trace(go.Scatter3d(
        x=xp[:, 0], y=xp[:, 1], z=xp[:, 2],
        mode='markers',
        marker=dict(size=2, color=cell_colors, opacity=0.6),
        name='cells' if i == 0 else None,
        showlegend=False,
        legendgroup='cells',
    ), row=row, col=2)

    fig3d.add_trace(go.Scatter3d(
        x=aa[0, :], y=aa[1, :], z=aa[2, :],
        mode='markers',
        marker=dict(size=6, color='black', symbol='diamond'),
        name='archetypes' if i == 0 else None,
        showlegend=False,
        legendgroup='archetypes',
    ), row=row, col=2)

    ex, ey, ez = [], [], []
    for a, b in itertools.combinations(range(noc), 2):
        ex += [aa[0, a], aa[0, b], None]
        ey += [aa[1, a], aa[1, b], None]
        ez += [aa[2, a], aa[2, b], None]

    fig3d.add_trace(go.Scatter3d(
        x=ex, y=ey, z=ez,
        mode='lines',
        line=dict(color='black', width=2),
        name='edges' if i == 0 else None,
        showlegend=False,
        legendgroup='edges',
    ), row=row, col=2)

    scene_key = 'scene' if row == 1 else f'scene{row}'
    fig3d.update_layout(**{scene_key: dict(
        xaxis_title='PC1',
        yaxis_title='PC2',
        zaxis_title='PC3',
    )})

fig3d.update_layout(
    title=f'Archetype selection — 2D & 3D view (NDIM={NDIM})',
    width=900,
    height=450 * nrows,
)

fig3d.write_html(FIG_NOC_ARCHETYPES)
print(f"Saved {FIG_NOC_ARCHETYPES}")
