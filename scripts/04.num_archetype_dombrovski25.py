# Number of archetypes selection — Dombrowski 2025 fly dataset
# Analyzes each (cell type × age) combination separately.
# Cell types: LC4, LPLC1, LPLC2 (cell type X excluded)
# Ages: APF_48h, APF_72h, APF_96h (from orig.ident)
# Replicates: genotype
# Outputs per (celltype, age):
#   - metrics PNG with EV, ARV, ARV_rep, effective EV, effective EV_rep
#   - interactive HTML (2D + 3D, cells colored by genotype)
#   - interactive HTML (3D per-genotype overlay)

import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import anndata as ad

from SingleCellArchetype.main import SCA
from SingleCellArchetype.utils import norm, get_relative_variation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(PROJECT_ROOT, 'local_data', 'processed', 'dombrowski25_fly', 'dombrowski25_fly.h5ad')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'dombrowski25_fly')

CELLTYPES = ['LC4', 'LPLC1', 'LPLC2']  # exclude 'X'
N_TOP_GENES = 2000
NDIM = 10
NOC_MIN = 2
NOC_MAX = 6
NREPEATS = 10


def select_hvg(x_raw, depths_raw, n_top_genes):
    """Select top highly variable genes via normalize→log1p→variance ranking."""
    median_depth = np.median(depths_raw)
    x_norm = x_raw / depths_raw[:, None] * median_depth
    gene_var = np.log1p(x_norm).var(axis=0)
    top_idx = np.argsort(gene_var)[::-1][:n_top_genes]
    mask = np.zeros(x_raw.shape[1], dtype=bool)
    mask[top_idx] = True
    return mask


def run_noc_sweep(sca, noc_grid, ndim, nrepeats, genotypes):
    """Run PCHA + bootstrap ARV + per-genotype ARV for each NOC.

    Returns parallel lists: ev_grid, av_grid, av_rep_grid, xp_grid, aa_grid, aa_reps_grid.
    aa_reps_grid[i] is a list of (genotype, aa) pairs for NOC i.
    """
    genotype_ids = np.unique(genotypes)
    ev_grid, av_grid, av_rep_grid = [], [], []
    xp_grid, aa_grid, aa_reps_grid = [], [], []

    for noc in noc_grid:
        xp, aa, ev = sca.proj_and_pcha(ndim, noc)

        aa_boots = sca.bootstrap_proj_pcha(ndim, noc, nrepeats=nrepeats)
        av = get_relative_variation(aa_boots)

        aa_reps = []
        for g in genotype_ids:
            gmask = genotypes == g
            if gmask.sum() < max(noc * 5, 20):
                continue
            try:
                _, aa_g, _ = sca.pcha_on_subset(gmask, noc)
            except Exception:
                continue
            aa_reps.append((g, aa_g))
        av_rep = get_relative_variation([a for _, a in aa_reps]) if len(aa_reps) >= 2 else float('nan')

        print(f"  NOC={noc}  EV={ev:.4f}  ARV={av:.4f}  ARV_rep={av_rep:.4f}"
              f"  effEV={ev*(1-av):.4f}  effEV_rep={ev*(1-av_rep):.4f}")

        ev_grid.append(ev)
        av_grid.append(av)
        av_rep_grid.append(av_rep)
        xp_grid.append(xp)
        aa_grid.append(aa)
        aa_reps_grid.append(aa_reps)

    return (np.array(ev_grid), np.array(av_grid), np.array(av_rep_grid),
            xp_grid, aa_grid, aa_reps_grid)


def save_metrics_plot(noc_grid, ev_grid, av_grid, av_rep_grid, ndim, title, out_path):
    """Save EV / ARV / effective-EV metrics PNG."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(noc_grid, ev_grid, '-o', color='black', label='explained variance (EV)')
    ax.plot(noc_grid, av_grid, '-o', color='steelblue', label='ARV (bootstrap)')
    ax.plot(noc_grid, av_rep_grid, '--o', color='steelblue', label='ARV_rep (per-genotype)')
    ax.plot(noc_grid, ev_grid * (1 - av_grid), '-o', color='tomato', label='effective EV (bootstrap)')
    ax.plot(noc_grid, ev_grid * (1 - av_rep_grid), '--o', color='tomato', label='effective EV (rep)')
    ax.set_xlabel('Number of archetypes (NOC)')
    ax.set_ylabel('Score')
    ax.set_ylim([0, 1])
    ax.set_xticks(noc_grid)
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=8)
    ax.set_title(title)
    sns.despine(ax=ax)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def save_interactive_html(noc_grid, ev_grid, av_grid, xp_grid, aa_grid,
                          cell_colors, ndim, title, out_path):
    """Save 2D + 3D interactive HTML, cells colored by genotype."""
    nrows = len(noc_grid)
    subplot_titles = []
    for i, noc in enumerate(noc_grid):
        subplot_titles += [
            f"2D  NOC={noc}  EV={ev_grid[i]:.3f}  ARV={av_grid[i]:.3f}",
            f"3D  NOC={noc}  EV={ev_grid[i]:.3f}  ARV={av_grid[i]:.3f}",
        ]
    specs = [[{'type': 'xy'}, {'type': 'scene'}]] * nrows
    fig = make_subplots(rows=nrows, cols=2, specs=specs, subplot_titles=subplot_titles)

    for i, noc in enumerate(noc_grid):
        row = i + 1
        xp, aa = xp_grid[i], aa_grid[i]

        fig.add_trace(go.Scatter(
            x=xp[:, 0], y=xp[:, 1], mode='markers',
            marker=dict(size=2, color=cell_colors, opacity=0.6),
            name='cells' if i == 0 else None, showlegend=(i == 0), legendgroup='cells',
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=aa[0, :], y=aa[1, :], mode='markers',
            marker=dict(size=8, color='black', symbol='diamond'),
            name='archetypes' if i == 0 else None, showlegend=(i == 0), legendgroup='archetypes',
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=list(aa[0, :]) + [aa[0, 0]], y=list(aa[1, :]) + [aa[1, 0]], mode='lines',
            line=dict(color='black', width=1.5),
            name='polygon' if i == 0 else None, showlegend=(i == 0), legendgroup='polygon',
        ), row=row, col=1)
        fig.update_xaxes(title_text='PC1', row=row, col=1)
        fig.update_yaxes(title_text='PC2', row=row, col=1)

        fig.add_trace(go.Scatter3d(
            x=xp[:, 0], y=xp[:, 1], z=xp[:, 2], mode='markers',
            marker=dict(size=2, color=cell_colors, opacity=0.6),
            showlegend=False, legendgroup='cells',
        ), row=row, col=2)
        fig.add_trace(go.Scatter3d(
            x=aa[0, :], y=aa[1, :], z=aa[2, :], mode='markers',
            marker=dict(size=6, color='black', symbol='diamond'),
            showlegend=False, legendgroup='archetypes',
        ), row=row, col=2)
        ex, ey, ez = [], [], []
        for a, b in itertools.combinations(range(noc), 2):
            ex += [aa[0, a], aa[0, b], None]
            ey += [aa[1, a], aa[1, b], None]
            ez += [aa[2, a], aa[2, b], None]
        fig.add_trace(go.Scatter3d(
            x=ex, y=ey, z=ez, mode='lines',
            line=dict(color='black', width=2),
            showlegend=False, legendgroup='edges',
        ), row=row, col=2)

        scene_key = 'scene' if row == 1 else f'scene{row}'
        fig.update_layout(**{scene_key: dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3')})

    fig.update_layout(title=title, width=900, height=450 * nrows)
    fig.write_html(out_path)
    print(f"  Saved {out_path}")


def save_rep_overlay_html(noc_grid, ev_grid, av_rep_grid, xp_grid, aa_reps_grid,
                          genotypes, genotype_to_color, ndim, title, out_path):
    """Save 3D per-genotype overlay HTML (one panel per NOC)."""
    genotype_ids = np.unique(genotypes)
    ncols = len(noc_grid)
    subplot_titles = [
        f"NOC={noc}  EV={ev_grid[i]:.3f}  ARV_rep={av_rep_grid[i]:.3f}"
        for i, noc in enumerate(noc_grid)
    ]
    fig = make_subplots(rows=1, cols=ncols,
                        specs=[[{'type': 'scene'}] * ncols],
                        subplot_titles=subplot_titles)

    for i, noc in enumerate(noc_grid):
        col = i + 1
        xp = xp_grid[i]
        aa_reps_dict = dict(aa_reps_grid[i])

        for g in genotype_ids:
            gmask = genotypes == g
            color = genotype_to_color[g]

            fig.add_trace(go.Scatter3d(
                x=xp[gmask, 0], y=xp[gmask, 1], z=xp[gmask, 2], mode='markers',
                marker=dict(size=2, color=color, opacity=0.5),
                name=str(g) if i == 0 else None,
                showlegend=(i == 0), legendgroup=str(g),
            ), row=1, col=col)

            if g not in aa_reps_dict:
                continue
            aa_g = aa_reps_dict[g]
            fig.add_trace(go.Scatter3d(
                x=aa_g[0, :], y=aa_g[1, :], z=aa_g[2, :], mode='markers',
                marker=dict(size=6, color=color, symbol='diamond'),
                showlegend=False, legendgroup=str(g),
            ), row=1, col=col)
            ex, ey, ez = [], [], []
            for a, b in itertools.combinations(range(noc), 2):
                ex += [aa_g[0, a], aa_g[0, b], None]
                ey += [aa_g[1, a], aa_g[1, b], None]
                ez += [aa_g[2, a], aa_g[2, b], None]
            fig.add_trace(go.Scatter3d(
                x=ex, y=ey, z=ez, mode='lines',
                line=dict(color=color, width=5),
                showlegend=False, legendgroup=str(g),
            ), row=1, col=col)

        scene_key = 'scene' if col == 1 else f'scene{col}'
        fig.update_layout(**{scene_key: dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3')})

    fig.update_layout(title=title, width=400 * ncols, height=500)
    fig.write_html(out_path)
    print(f"  Saved {out_path}")


def run_one(celltype, age, x_all, depths_all, types_all, ages_all, genotypes_all,
            noc_grid, ndim, nrepeats, fig_dir):
    """Run full archetype-number sweep for one (celltype, age) and save outputs."""
    mask = (types_all == celltype) & (ages_all == age)
    n_cells = mask.sum()
    print(f"\n=== celltype={celltype}  age={age}  n_cells={n_cells} ===")
    if n_cells == 0:
        print("  Skipping: no cells.")
        return

    x = x_all[mask]
    depths = depths_all[mask]
    genotypes = genotypes_all[mask]
    genotype_ids = np.unique(genotypes)

    # drop constant genes to avoid NaN in zscore
    x = x[:, x.var(axis=0) > 0]

    xn = norm(x, depths)
    sca = SCA(xn, genotypes)
    sca.setup_feature_matrix(method='data')

    ev_grid, av_grid, av_rep_grid, xp_grid, aa_grid, aa_reps_grid = run_noc_sweep(
        sca, noc_grid, ndim, nrepeats, genotypes)

    tag = f"{celltype}_{age}"
    fig_metrics = os.path.join(fig_dir, f"04.{tag}_metrics.png")
    fig_interactive = os.path.join(fig_dir, f"04.{tag}_interactive.html")
    fig_rep = os.path.join(fig_dir, f"04.{tag}_interactive_rep.html")

    save_metrics_plot(noc_grid, ev_grid, av_grid, av_rep_grid,
                      ndim, f'{celltype}  {age}  (NDIM={ndim})', fig_metrics)

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    genotype_to_color = {g: cycle[i % len(cycle)] for i, g in enumerate(genotype_ids)}
    cell_colors = [genotype_to_color[g] for g in genotypes]

    save_interactive_html(noc_grid, ev_grid, av_grid, xp_grid, aa_grid,
                          cell_colors, ndim,
                          f'{celltype}  {age} — 2D & 3D view (NDIM={ndim})',
                          fig_interactive)

    save_rep_overlay_html(noc_grid, ev_grid, av_rep_grid, xp_grid, aa_reps_grid,
                          genotypes, genotype_to_color, ndim,
                          f'{celltype}  {age} — per-genotype overlay (NDIM={ndim})',
                          fig_rep)


# --- main ---
adata = ad.read_h5ad(INPUT_FILE)
print(adata)

x_raw = np.array(adata.X)
depths_raw = adata.obs['nCount_RNA'].values
hvg_mask = select_hvg(x_raw, depths_raw, N_TOP_GENES)

x_all = x_raw[:, hvg_mask]
depths_all = depths_raw
types_all = adata.obs['type1'].values
ages_all = adata.obs['orig.ident'].values
genotypes_all = adata.obs['genotype'].values

os.makedirs(FIG_DIR, exist_ok=True)
noc_grid = np.arange(NOC_MIN, NOC_MAX + 1)

for celltype in CELLTYPES:
    for age in np.unique(ages_all):
        run_one(celltype, age, x_all, depths_all, types_all, ages_all, genotypes_all,
                noc_grid, NDIM, NREPEATS, FIG_DIR)
