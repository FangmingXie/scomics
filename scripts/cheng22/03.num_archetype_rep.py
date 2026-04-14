# Number of archetypes selection — per-sample replicate variation version
# Extends 02.num_archetype_3d.py by adding av_rep: archetype variation across
# biological samples (replicates), alongside the bootstrap-based av metric.
# Outputs:
#   - metrics PNG with EV, ARV, ARV_rep, effective EV, effective EV_rep
#   - interactive HTML (2D + 3D, cells colored by type)
#   - interactive HTML (3D per-sample overlay, cells and edges colored by sample)

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
FIG_NOC_METRICS = os.path.join(FIG_DIR, '03.num_archetype_rep_metrics.png')
FIG_NOC_ARCHETYPES = os.path.join(FIG_DIR, '03.num_archetype_rep_interactive.html')
FIG_NOC_ARCHETYPES_REP = os.path.join(FIG_DIR, '03.num_archetype_rep_interactive_rep.html')

NDIM = 10 
NOC_MIN = 2
NOC_MAX = 6
NREPEATS = 10

# load data
adata = ad.read_h5ad(INPUT_FILE)
print(adata)

x = np.array(adata.X.todense())
depths = adata.obs['n_counts'].values
types = adata.obs['Type'].values
samples = adata.obs['sample'].values

xn = norm(x, depths)
sca = SCA(xn, types)
sca.setup_feature_matrix(method='data')

sample_ids = np.unique(samples)

noc_grid = np.arange(NOC_MIN, NOC_MAX + 1)
ev_grid = []
av_grid = []
av_rep_grid = []
xp_grid = []
aa_grid = []
aa_reps_grid = []  # list of lists: per-NOC, per-sample archetype matrices

for noc in noc_grid:
    xp, aa, ev = sca.proj_and_pcha(NDIM, noc)

    # bootstrap-based variation
    aa_boots = sca.bootstrap_proj_pcha(NDIM, noc, nrepeats=NREPEATS)
    av = get_relative_variation(aa_boots)

    # per-sample replicate variation
    aa_reps = []
    for s in sample_ids:
        mask = samples == s
        _, aa_s, _ = sca.pcha_on_subset(mask, noc)
        aa_reps.append(aa_s)
    av_rep = get_relative_variation(aa_reps)

    print(f"NOC={noc}  EV={ev:.4f}  ARV={av:.4f}  ARV_rep={av_rep:.4f}"
          f"  effEV={ev*(1-av):.4f}  effEV_rep={ev*(1-av_rep):.4f}")

    ev_grid.append(ev)
    av_grid.append(av)
    av_rep_grid.append(av_rep)
    xp_grid.append(xp)
    aa_grid.append(aa)
    aa_reps_grid.append(aa_reps)

ev_grid = np.array(ev_grid)
av_grid = np.array(av_grid)
av_rep_grid = np.array(av_rep_grid)

# --- plot: EV, ARV, ARV_rep, effective EV, effective EV_rep vs NOC ---
os.makedirs(FIG_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(noc_grid, ev_grid, '-o', color='black', label='explained variance (EV)')
ax.plot(noc_grid, av_grid, '-o', color='steelblue', label='ARV (bootstrap)')
ax.plot(noc_grid, av_rep_grid, '--o', color='steelblue', label='ARV_rep (per-sample)')
ax.plot(noc_grid, ev_grid * (1 - av_grid), '-o', color='tomato', label='effective EV (bootstrap)')
ax.plot(noc_grid, ev_grid * (1 - av_rep_grid), '--o', color='tomato', label='effective EV (rep)')
ax.set_xlabel('Number of archetypes (NOC)')
ax.set_ylabel('Score')
ax.set_ylim([0, 1])
ax.set_xticks(noc_grid)
ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=8)
ax.set_title(f'Archetype selection (NDIM={NDIM})')
sns.despine(ax=ax)
ax.grid(False)

fig.tight_layout()
fig.savefig(FIG_NOC_METRICS, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f"Saved {FIG_NOC_METRICS}")

# color helpers
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
type_to_color = {t: cycle[i % len(cycle)] for i, t in enumerate(np.unique(types))}
cell_colors_type = [type_to_color[t] for t in types]

sample_to_color = {s: cycle[i % len(cycle)] for i, s in enumerate(sample_ids)}
cell_colors_sample = [sample_to_color[s] for s in samples]

# --- HTML 1: 2D + 3D panels, colored by type ---
nrows = len(noc_grid)
subplot_titles = []
for i, noc in enumerate(noc_grid):
    subplot_titles += [
        f"2D  NOC={noc}  EV={ev_grid[i]:.3f}  ARV={av_grid[i]:.3f}",
        f"3D  NOC={noc}  EV={ev_grid[i]:.3f}  ARV={av_grid[i]:.3f}",
    ]
specs = [[{'type': 'xy'}, {'type': 'scene'}]] * nrows

fig_type = make_subplots(rows=nrows, cols=2, specs=specs, subplot_titles=subplot_titles)

for i, noc in enumerate(noc_grid):
    row = i + 1
    xp = xp_grid[i]
    aa = aa_grid[i]

    # 2D
    fig_type.add_trace(go.Scatter(
        x=xp[:, 0], y=xp[:, 1], mode='markers',
        marker=dict(size=2, color=cell_colors_type, opacity=0.6),
        name='cells' if i == 0 else None, showlegend=(i == 0), legendgroup='cells',
    ), row=row, col=1)
    fig_type.add_trace(go.Scatter(
        x=aa[0, :], y=aa[1, :], mode='markers',
        marker=dict(size=8, color='black', symbol='diamond'),
        name='archetypes' if i == 0 else None, showlegend=(i == 0), legendgroup='archetypes',
    ), row=row, col=1)
    px = list(aa[0, :]) + [aa[0, 0]]
    py = list(aa[1, :]) + [aa[1, 0]]
    fig_type.add_trace(go.Scatter(
        x=px, y=py, mode='lines',
        line=dict(color='black', width=1.5),
        name='polygon' if i == 0 else None, showlegend=(i == 0), legendgroup='polygon',
    ), row=row, col=1)
    fig_type.update_xaxes(title_text='PC1', row=row, col=1)
    fig_type.update_yaxes(title_text='PC2', row=row, col=1)

    # 3D
    fig_type.add_trace(go.Scatter3d(
        x=xp[:, 0], y=xp[:, 1], z=xp[:, 2], mode='markers',
        marker=dict(size=2, color=cell_colors_type, opacity=0.6),
        showlegend=False, legendgroup='cells',
    ), row=row, col=2)
    fig_type.add_trace(go.Scatter3d(
        x=aa[0, :], y=aa[1, :], z=aa[2, :], mode='markers',
        marker=dict(size=6, color='black', symbol='diamond'),
        showlegend=False, legendgroup='archetypes',
    ), row=row, col=2)
    ex, ey, ez = [], [], []
    for a, b in itertools.combinations(range(noc), 2):
        ex += [aa[0, a], aa[0, b], None]
        ey += [aa[1, a], aa[1, b], None]
        ez += [aa[2, a], aa[2, b], None]
    fig_type.add_trace(go.Scatter3d(
        x=ex, y=ey, z=ez, mode='lines',
        line=dict(color='black', width=2),
        showlegend=False, legendgroup='edges',
    ), row=row, col=2)

    scene_key = 'scene' if row == 1 else f'scene{row}'
    fig_type.update_layout(**{scene_key: dict(
        xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3',
    )})

fig_type.update_layout(
    title=f'Archetype selection — 2D & 3D view (NDIM={NDIM})',
    width=900, height=450 * nrows,
)
fig_type.write_html(FIG_NOC_ARCHETYPES)
print(f"Saved {FIG_NOC_ARCHETYPES}")

# --- HTML 2: 3D per-sample overlay, one panel per NOC, colored by sample ---
ncols = len(noc_grid)
subplot_titles_rep = [
    f"NOC={noc}  EV={ev_grid[i]:.3f}  ARV_rep={av_rep_grid[i]:.3f}"
    for i, noc in enumerate(noc_grid)
]
specs_rep = [[{'type': 'scene'}] * ncols]

fig_rep = make_subplots(rows=1, cols=ncols, specs=specs_rep,
                        subplot_titles=subplot_titles_rep)

for i, noc in enumerate(noc_grid):
    col = i + 1
    xp = xp_grid[i]
    aa_reps = aa_reps_grid[i]

    for j, s in enumerate(sample_ids):
        mask = samples == s
        color = sample_to_color[s]

        # cells for this sample
        fig_rep.add_trace(go.Scatter3d(
            x=xp[mask, 0], y=xp[mask, 1], z=xp[mask, 2], mode='markers',
            marker=dict(size=2, color=color, opacity=0.5),
            name=str(s) if i == 0 else None,
            showlegend=(i == 0),
            legendgroup=str(s),
        ), row=1, col=col)

        # archetype vertices for this sample
        aa_s = aa_reps[j]
        fig_rep.add_trace(go.Scatter3d(
            x=aa_s[0, :], y=aa_s[1, :], z=aa_s[2, :], mode='markers',
            marker=dict(size=6, color=color, symbol='diamond'),
            showlegend=False,
            legendgroup=str(s),
        ), row=1, col=col)

        # archetype edges for this sample
        ex, ey, ez = [], [], []
        for a, b in itertools.combinations(range(noc), 2):
            ex += [aa_s[0, a], aa_s[0, b], None]
            ey += [aa_s[1, a], aa_s[1, b], None]
            ez += [aa_s[2, a], aa_s[2, b], None]
        fig_rep.add_trace(go.Scatter3d(
            x=ex, y=ey, z=ez, mode='lines',
            line=dict(color=color, width=5),
            showlegend=False,
            legendgroup=str(s),
        ), row=1, col=col)

    scene_key = 'scene' if col == 1 else f'scene{col}'
    fig_rep.update_layout(**{scene_key: dict(
        xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3',
    )})

fig_rep.update_layout(
    title=f'Per-sample archetype overlay (NDIM={NDIM})',
    width=400 * ncols, height=500,
)
fig_rep.write_html(FIG_NOC_ARCHETYPES_REP)
print(f"Saved {FIG_NOC_ARCHETYPES_REP}")
