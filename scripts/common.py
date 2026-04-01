# Shared functions for archetype number selection scripts.

import itertools
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from SingleCellArchetype.utils import get_relative_variation


def select_hvg(x_raw, depths_raw, n_top_genes):
    """Select top highly variable genes via normalize→log1p→variance ranking."""
    median_depth = np.median(depths_raw)
    x_norm = x_raw / depths_raw[:, None] * median_depth
    gene_var = np.log1p(x_norm).var(axis=0)
    top_idx = np.argsort(gene_var)[::-1][:n_top_genes]
    mask = np.zeros(x_raw.shape[1], dtype=bool)
    mask[top_idx] = True
    return mask


def run_noc_sweep(sca, noc_grid, ndim, nrepeats, groups):
    """Run PCHA + bootstrap ARV + per-group ARV for each NOC.

    Returns parallel lists: ev_grid, av_grid, av_rep_grid, xp_grid, aa_grid, aa_reps_grid.
    aa_reps_grid[i] is a list of (group_id, aa) pairs for NOC i.
    """
    group_ids = np.unique(groups)
    ev_grid, av_grid, av_rep_grid = [], [], []
    xp_grid, aa_grid, aa_reps_grid = [], [], []

    for noc in noc_grid:
        xp, aa, ev = sca.proj_and_pcha(ndim, noc)

        aa_boots = sca.bootstrap_proj_pcha(ndim, noc, nrepeats=nrepeats)
        av = get_relative_variation(aa_boots)

        aa_reps = []
        for g in group_ids:
            gmask = groups == g
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
    ax.plot(noc_grid, av_rep_grid, '--o', color='steelblue', label='ARV_rep (per-group)')
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
    """Save 2D + 3D interactive HTML, cells colored by a categorical variable."""
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


def _metadata_to_colors(values):
    """Convert raw metadata values to per-cell hex color strings.

    Numeric arrays use viridis with vmin/vmax at 5th/95th percentiles.
    Non-numeric arrays use discrete matplotlib color cycle.
    """
    try:
        vals = np.array(values, dtype=float)
        vmin = np.nanpercentile(vals, 5)
        vmax = np.nanpercentile(vals, 95)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        return [mcolors.to_hex(cm.viridis(norm(v))) for v in vals]
    except (ValueError, TypeError):
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        unique_vals = sorted(set(str(v) for v in values))
        val_to_color = {v: cycle[i % len(cycle)] for i, v in enumerate(unique_vals)}
        return [val_to_color[str(v)] for v in values]


def save_interactive_html_pro(noc_grid, ev_grid, av_grid, xp_grid, aa_grid,
                              cell_metadata, ndim, title, out_path):
    """Save 2D + 3D interactive HTML with toggle buttons to color cells by different metadata labels.

    All NOC archetype polygons are overlaid on the same two panels (2D left, 3D right).
    Each NOC's archetypes can be toggled on/off via the Plotly legend.

    cell_metadata: dict[str, array] mapping label name to per-cell raw values.
    Categorical variables use discrete color cycle; continuous variables use viridis
    with vmin/vmax clipped at 5th/95th percentiles.
    """
    cell_metadata_colors = {k: _metadata_to_colors(v) for k, v in cell_metadata.items()}
    initial_colors = next(iter(cell_metadata_colors.values()))
    n = len(noc_grid)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'xy'}, {'type': 'scene'}]],
        subplot_titles=['2D  PC1 vs PC2', '3D  PC1–PC3'],
    )

    # Cells plotted once (PCA positions are NOC-independent; use xp_grid[0])
    xp = xp_grid[0]

    # Trace index 0: 2D cells
    fig.add_trace(go.Scatter(
        x=xp[:, 0], y=xp[:, 1], mode='markers',
        marker=dict(size=2, color=initial_colors, opacity=0.6),
        name='cells', showlegend=True, legendgroup='cells',
    ), row=1, col=1)
    fig.update_xaxes(title_text='PC1', row=1, col=1)
    fig.update_yaxes(title_text='PC2', row=1, col=1)

    # Traces 1..2n: 2D archetypes per NOC (2 traces each: markers + polygon)
    for i, noc in enumerate(noc_grid):
        aa = aa_grid[i]
        lg = f'NOC={noc}  EV={ev_grid[i]:.3f}  ARV={av_grid[i]:.3f}'
        fig.add_trace(go.Scatter(
            x=aa[0, :], y=aa[1, :], mode='markers',
            marker=dict(size=8, color='black', symbol='diamond'),
            name=lg, showlegend=True, legendgroup=lg,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=list(aa[0, :]) + [aa[0, 0]], y=list(aa[1, :]) + [aa[1, 0]], mode='lines',
            line=dict(color='black', width=1.5),
            showlegend=False, legendgroup=lg,
        ), row=1, col=1)

    # Trace index 2n+1: 3D cells
    fig.add_trace(go.Scatter3d(
        x=xp[:, 0], y=xp[:, 1], z=xp[:, 2], mode='markers',
        marker=dict(size=2, color=initial_colors, opacity=0.6),
        showlegend=False, legendgroup='cells',
    ), row=1, col=2)

    # Traces 2n+2..4n+1: 3D archetypes per NOC (2 traces each: markers + edges)
    for i, noc in enumerate(noc_grid):
        aa = aa_grid[i]
        lg = f'NOC={noc}  EV={ev_grid[i]:.3f}  ARV={av_grid[i]:.3f}'
        fig.add_trace(go.Scatter3d(
            x=aa[0, :], y=aa[1, :], z=aa[2, :], mode='markers',
            marker=dict(size=6, color='black', symbol='diamond'),
            showlegend=False, legendgroup=lg,
        ), row=1, col=2)
        ex, ey, ez = [], [], []
        for a, b in itertools.combinations(range(noc), 2):
            ex += [aa[0, a], aa[0, b], None]
            ey += [aa[1, a], aa[1, b], None]
            ez += [aa[2, a], aa[2, b], None]
        fig.add_trace(go.Scatter3d(
            x=ex, y=ey, z=ez, mode='lines',
            line=dict(color='black', width=2),
            showlegend=False, legendgroup=lg,
        ), row=1, col=2)

    fig.update_layout(scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'))

    # Metadata color buttons target cell traces only: index 0 (2D) and 2n+1 (3D)
    cell_trace_indices = [0, 2 * n + 1]
    buttons = [
        dict(label=label, method='restyle',
             args=[{'marker.color': [colors] * len(cell_trace_indices)}, cell_trace_indices])
        for label, colors in cell_metadata_colors.items()
    ]

    fig.update_layout(
        title=title, width=1100, height=600,
        updatemenus=[dict(
            type='buttons', direction='right',
            x=0.0, xanchor='left', y=1.05, yanchor='bottom',
            buttons=buttons,
        )],
    )
    fig.write_html(out_path)
    print(f"  Saved {out_path}")


def save_group_overlay_html(noc_grid, ev_grid, av_rep_grid, xp_grid, aa_reps_grid,
                            groups, group_to_color, ndim, title, out_path):
    """Save 3D per-group overlay HTML (one panel per NOC)."""
    group_ids = np.unique(groups)
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

        for g in group_ids:
            gmask = groups == g
            color = group_to_color[g]

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
