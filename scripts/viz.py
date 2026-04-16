# Shared visualization functions for archetype analysis scripts.

import itertools
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from natsort import natsorted


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


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

def _scene_key(col):
    return 'scene' if col == 1 else f'scene{col}'


def _archetype_edges_3d(aa, noc):
    """Return (ex, ey, ez) edge coordinate lists for all archetype pairs."""
    ex, ey, ez = [], [], []
    for a, b in itertools.combinations(range(noc), 2):
        ex += [aa[0, a], aa[0, b], None]
        ey += [aa[1, a], aa[1, b], None]
        ez += [aa[2, a], aa[2, b], None]
    return ex, ey, ez


def _add_archetype_2d(fig, aa, noc, lg, row, col, *, show_legend):
    """Add 2D archetype diamond markers and closing polygon (shared legendgroup lg)."""
    fig.add_trace(go.Scatter(
        x=aa[0, :], y=aa[1, :], mode='markers',
        marker=dict(size=8, color='black', symbol='diamond'),
        name=lg, showlegend=show_legend, legendgroup=lg,
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=list(aa[0, :]) + [aa[0, 0]], y=list(aa[1, :]) + [aa[1, 0]], mode='lines',
        line=dict(color='black', width=1.5),
        showlegend=False, legendgroup=lg,
    ), row=row, col=col)


def _add_archetype_3d(fig, aa, noc, lg, row, col, color='black', marker_size=6):
    """Add 3D archetype diamond markers and edge mesh (shared legendgroup lg)."""
    fig.add_trace(go.Scatter3d(
        x=aa[0, :], y=aa[1, :], z=aa[2, :], mode='markers',
        marker=dict(size=marker_size, color=color, symbol='diamond'),
        showlegend=False, legendgroup=lg,
    ), row=row, col=col)
    ex, ey, ez = _archetype_edges_3d(aa, noc)
    fig.add_trace(go.Scatter3d(
        x=ex, y=ey, z=ez, mode='lines',
        line=dict(color=color, width=2),
        showlegend=False, legendgroup=lg,
    ), row=row, col=col)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

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


def _add_2d_panel(fig, xp, noc_entries, initial_colors, row, col):
    """Add 2D cell scatter and per-NOC archetype overlays to fig. Returns cell trace index.

    noc_entries: list of (noc, ev, av, aa) tuples.
    """
    cell_trace_index = len(fig.data)
    fig.add_trace(go.Scatter(
        x=xp[:, 0], y=xp[:, 1], mode='markers',
        marker=dict(size=2, color=initial_colors, opacity=0.6),
        name='cells', showlegend=True, legendgroup='cells',
    ), row=row, col=col)
    fig.update_xaxes(title_text='PC1', row=row, col=col)
    fig.update_yaxes(title_text='PC2', row=row, col=col)
    for noc, ev, av, aa in noc_entries:
        lg = f'NOC={noc}  EV={ev:.3f}  ARV={av:.3f}'
        _add_archetype_2d(fig, aa, noc, lg, row=row, col=col, show_legend=True)
    return cell_trace_index


def _add_3d_panel(fig, xp, noc_entries, initial_colors, row, col):
    """Add 3D cell scatter and per-NOC archetype overlays to fig. Returns cell trace index.

    noc_entries: list of (noc, ev, av, aa) tuples.
    """
    cell_trace_index = len(fig.data)
    fig.add_trace(go.Scatter3d(
        x=xp[:, 0], y=xp[:, 1], z=xp[:, 2], mode='markers',
        marker=dict(size=2, color=initial_colors, opacity=0.6),
        showlegend=True, legendgroup='cells',
    ), row=row, col=col)
    for noc, ev, av, aa in noc_entries:
        lg = f'NOC={noc}  EV={ev:.3f}  ARV={av:.3f}'
        _add_archetype_3d(fig, aa, noc, lg, row=row, col=col)
    fig.update_layout(**{_scene_key(col): dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3')})
    return cell_trace_index


def scatter_html(xp_grid, cell_metadata, title, out_path,
                 noc_grid=(), ev_grid=(), av_grid=(), aa_grid=()):
    """Save 2D + 3D interactive HTML with toggle buttons to color cells by different metadata labels.

    All NOC archetype polygons are overlaid on the same two panels (2D left, 3D right).
    Each NOC's archetypes can be toggled on/off via the Plotly legend.

    cell_metadata: dict[str, array] mapping label name to per-cell raw values.
    Categorical variables use discrete color cycle; continuous variables use viridis
    with vmin/vmax clipped at 5th/95th percentiles.
    """
    cell_metadata_colors = {k: _metadata_to_colors(v) for k, v in cell_metadata.items()}
    initial_colors = next(iter(cell_metadata_colors.values()))
    noc_entries = list(zip(noc_grid, ev_grid, av_grid, aa_grid))

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'xy'}, {'type': 'scene'}]],
        subplot_titles=['2D  PC1 vs PC2', '3D  PC1–PC3'],
    )

    # PCA positions are NOC-independent; use xp_grid[0]
    xp = xp_grid[0]
    idx_2d = _add_2d_panel(fig, xp, noc_entries, initial_colors, row=1, col=1)
    idx_3d = _add_3d_panel(fig, xp, noc_entries, initial_colors, row=1, col=2)

    cell_trace_indices = [idx_2d, idx_3d]
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


def scatter_categorical_html(xp_grid, cell_metadata, title, out_path,
                             noc_grid=(), ev_grid=(), av_grid=(), aa_grid=(),
                             ordered_labels=(),
                             xlabel='PC1', ylabel='PC2', zlabel='PC3'):
    """Like scatter_html but uses per-category traces so the Plotly legend shows one entry per category.

    For categorical metadata: one 2D + one 3D trace per unique value; clicking a legend entry
    hides/shows that category across both panels.
    For continuous metadata: single 2D + 3D trace with viridis colorscale.

    Buttons switch which metadata label is active (all other label traces are hidden).
    Archetype traces are always visible.
    ordered_labels: collection of metadata keys that should use evenly spaced turbo colors
                    (e.g. time-ordered categories like Age).
    """
    xp = xp_grid[0]
    noc_entries = list(zip(noc_grid, ev_grid, av_grid, aa_grid))
    cat_palette = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'xy'}, {'type': 'scene'}]],
        subplot_titles=['2D  PC1 vs PC2', '3D  PC1–PC3'],
    )

    labels = list(cell_metadata.keys())
    label_trace_ranges = {}  # label -> (start_idx, end_idx)

    for li, (label, values) in enumerate(cell_metadata.items()):
        visible = (li == 0)
        start_idx = len(fig.data)

        try:
            vals = np.array(values, dtype=float)
            vmin = np.nanpercentile(vals, 5)
            vmax = np.nanpercentile(vals, 95)
            fig.add_trace(go.Scatter(
                x=xp[:, 0], y=xp[:, 1], mode='markers',
                marker=dict(size=2, color=vals, colorscale='Viridis', cmin=vmin, cmax=vmax,
                            opacity=0.6, showscale=True),
                name=label, showlegend=False, visible=visible,
            ), row=1, col=1)
            fig.add_trace(go.Scatter3d(
                x=xp[:, 0], y=xp[:, 1], z=xp[:, 2], mode='markers',
                marker=dict(size=2, color=vals, colorscale='Viridis', cmin=vmin, cmax=vmax,
                            opacity=0.6, showscale=False),
                showlegend=False, visible=visible,
            ), row=1, col=2)
        except (ValueError, TypeError):
            str_vals = np.array([str(v) for v in values])
            unique_vals = natsorted(set(str_vals))
            n = len(unique_vals)
            if label in ordered_labels:
                cmap = cm.get_cmap('turbo', n)
                val_to_color = {v: mcolors.to_hex(cmap(i / max(n - 1, 1))) for i, v in enumerate(unique_vals)}
            else:
                val_to_color = {v: cat_palette[i % len(cat_palette)] for i, v in enumerate(unique_vals)}
            for uv in unique_vals:
                mask = str_vals == uv
                color = val_to_color[uv]
                lg = f'{label}__{uv}'
                fig.add_trace(go.Scatter(
                    x=xp[mask, 0], y=xp[mask, 1], mode='markers',
                    marker=dict(size=2, color=color, opacity=0.6),
                    name=uv, legendgroup=lg, showlegend=True, visible=visible,
                ), row=1, col=1)
                fig.add_trace(go.Scatter3d(
                    x=xp[mask, 0], y=xp[mask, 1], z=xp[mask, 2], mode='markers',
                    marker=dict(size=2, color=color, opacity=0.6),
                    showlegend=False, legendgroup=lg, visible=visible,
                ), row=1, col=2)

        label_trace_ranges[label] = (start_idx, len(fig.data))

    fig.update_xaxes(title_text=xlabel, row=1, col=1)
    fig.update_yaxes(title_text=ylabel, row=1, col=1)
    fig.update_layout(**{_scene_key(2): dict(xaxis_title=xlabel, yaxis_title=ylabel, zaxis_title=zlabel)})

    arch_start = len(fig.data)
    for noc, ev, av, aa in noc_entries:
        lg = f'NOC={noc}  EV={ev:.3f}  ARV={av:.3f}'
        _add_archetype_2d(fig, aa, noc, lg, row=1, col=1, show_legend=True)
        _add_archetype_3d(fig, aa, noc, lg, row=1, col=2)
    n_total = len(fig.data)

    buttons = []
    for label in labels:
        start, end = label_trace_ranges[label]
        vis = [
            True if (i >= arch_start or start <= i < end) else False
            for i in range(n_total)
        ]
        buttons.append(dict(label=label, method='update', args=[{'visible': vis}]))

    fig.update_layout(
        title=title, width=1100, height=600,
        legend=dict(itemsizing='constant'),
        updatemenus=[dict(
            type='buttons', direction='right',
            x=0.0, xanchor='left', y=1.05, yanchor='bottom',
            buttons=buttons,
        )],
    )
    fig.write_html(out_path)
    print(f"  Saved {out_path}")


def scatter_2d_categorical_html(xp_grid, cell_metadata, title, out_path,
                                xlabel='Dim1', ylabel='Dim2', ordered_labels=()):
    """Like scatter_categorical_html but a single 2D panel only.

    Useful for UMAP or any 2D embedding where a 3D view is not meaningful.
    xlabel/ylabel label the axes (e.g. 'UMAP1'/'UMAP2' or 'PC1'/'PC2').
    ordered_labels: collection of metadata keys that should use evenly spaced
                    viridis colors (e.g. time-ordered categories like Age).
                    All other categorical labels use the default color cycle.
    """
    xp = xp_grid[0]
    cat_palette = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig = go.Figure()

    labels = list(cell_metadata.keys())
    label_trace_ranges = {}

    for li, (label, values) in enumerate(cell_metadata.items()):
        visible = (li == 0)
        start_idx = len(fig.data)

        try:
            vals = np.array(values, dtype=float)
            vmin = np.nanpercentile(vals, 5)
            vmax = np.nanpercentile(vals, 95)
            fig.add_trace(go.Scatter(
                x=xp[:, 0], y=xp[:, 1], mode='markers',
                marker=dict(size=2, color=vals, colorscale='Viridis', cmin=vmin, cmax=vmax,
                            opacity=0.6, showscale=True),
                name=label, showlegend=False, visible=visible,
            ))
        except (ValueError, TypeError):
            str_vals = np.array([str(v) for v in values])
            unique_vals = natsorted(set(str_vals))
            n = len(unique_vals)
            if label in ordered_labels:
                cmap = cm.get_cmap('turbo', n)
                val_to_color = {v: mcolors.to_hex(cmap(i / max(n - 1, 1))) for i, v in enumerate(unique_vals)}
            else:
                val_to_color = {v: cat_palette[i % len(cat_palette)] for i, v in enumerate(unique_vals)}
            for uv in unique_vals:
                mask = str_vals == uv
                fig.add_trace(go.Scatter(
                    x=xp[mask, 0], y=xp[mask, 1], mode='markers',
                    marker=dict(size=2, color=val_to_color[uv], opacity=0.6),
                    name=uv, legendgroup=f'{label}__{uv}',
                    showlegend=True, visible=visible,
                ))

        label_trace_ranges[label] = (start_idx, len(fig.data))

    n_total = len(fig.data)
    buttons = []
    for label in labels:
        start, end = label_trace_ranges[label]
        vis = [start <= i < end for i in range(n_total)]
        buttons.append(dict(label=label, method='update', args=[{'visible': vis}]))

    fig.update_layout(
        title=title,
        xaxis_title=xlabel, yaxis_title=ylabel,
        width=700, height=600,
        legend=dict(itemsizing='constant'),
        updatemenus=[dict(
            type='buttons', direction='right',
            x=0.0, xanchor='left', y=1.05, yanchor='bottom',
            buttons=buttons,
        )],
    )
    fig.write_html(out_path)
    print(f"  Saved {out_path}")


def scatter_per_group_html(noc_grid, ev_grid, av_rep_grid, xp_grid, aa_reps_grid,
                            groups, group_to_color, title, out_path):
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
            lg = str(g)

            fig.add_trace(go.Scatter3d(
                x=xp[gmask, 0], y=xp[gmask, 1], z=xp[gmask, 2], mode='markers',
                marker=dict(size=2, color=color, opacity=0.5),
                name=lg if i == 0 else None,
                showlegend=(i == 0), legendgroup=lg,
            ), row=1, col=col)

            if g not in aa_reps_dict:
                continue
            _add_archetype_3d(fig, aa_reps_dict[g], noc, lg, row=1, col=col,
                               color=color, marker_size=6)

        fig.update_layout(**{_scene_key(col): dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3')})

    fig.update_layout(title=title, width=400 * ncols, height=500)
    fig.write_html(out_path)
    print(f"  Saved {out_path}")


def stacked_bar_html(panel_data, celltypes, title, out_path, ct_colors=None, panel_width=500):
    """Save an interactive stacked bar chart HTML with one panel per group.

    panel_data: list of (panel_title, group_order, frac_df) where frac_df is a
                DataFrame indexed by group with celltypes as columns (values in [0,1]).
    celltypes:  ordered list of cell type names (stacking order).
    ct_colors:  dict mapping celltype -> hex color string. Defaults to tab10.
    """
    if ct_colors is None:
        # cmap = cm.get_cmap('tab10', len(celltypes))
        cmap = cm.get_cmap('tab10', 10) 
        ct_colors = {ct: mcolors.to_hex(cmap(i)) for i, ct in enumerate(celltypes)}
    fig = make_subplots(
        rows=1, cols=len(panel_data),
        subplot_titles=[pd_[0] for pd_ in panel_data],
        shared_yaxes=True,
    )

    for col_idx, (panel_title, group_order, frac_df) in enumerate(panel_data, start=1):
        for i, ct in enumerate(celltypes):
            fig.add_trace(go.Bar(
                name=ct,
                x=group_order,
                y=frac_df.reindex(group_order)[ct].values,
                marker_color=ct_colors[ct],
                legendgroup=ct,
                showlegend=(col_idx == 1),
            ), row=1, col=col_idx)

    fig.update_layout(
        barmode='stack',
        title=title,
        yaxis_title='Fraction of cells',
        yaxis=dict(range=[0, 1]),
        legend=dict(itemsizing='constant', traceorder='normal'),
        width=panel_width * len(panel_data),
        height=600,
    )
    fig.update_xaxes(tickangle=45)
    fig.write_html(out_path)
    print(f"  Saved {out_path}")


def gene_expr_scatter_html(x, y, gene_vals, title, out_path,
                           xlabel='Dim1', ylabel='Dim2',
                           z=None, zlabel='Dim3',
                           aa=None,
                           colorscale='RdBu_r',
                           pctile_low=9, pctile_high=95,
                           marker_size=3, marker_opacity=0.6,
                           colorbar_title='z-score',
                           width=850, height=700):
    """Save interactive 2D scatter HTML colored by gene expression with a gene dropdown.

    gene_vals: dict[str, np.ndarray] mapping gene name to per-cell float values (e.g. z-scores).
    z:  optional array for a third dimension; when provided, adds a 3D panel alongside the 2D one.
    aa: optional archetype coordinate array (ndim × noc); overlaid on both panels when provided.
    """
    genes = list(gene_vals.keys())
    has_3d = z is not None

    if has_3d:
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'xy'}, {'type': 'scene'}]],
            subplot_titles=[f'2D  {xlabel} vs {ylabel}', f'3D  {xlabel}–{zlabel}'],
        )
    else:
        fig = go.Figure()

    traces_per_gene = 2 if has_3d else 1
    for i, gene in enumerate(genes):
        vals = gene_vals[gene]
        cmin = np.nanpercentile(vals, pctile_low)
        cmax = np.nanpercentile(vals, pctile_high)
        visible = (i == 0)
        marker_2d = dict(size=marker_size, color=vals, colorscale=colorscale,
                         cmin=cmin, cmax=cmax, opacity=marker_opacity,
                         showscale=True, colorbar=dict(title=colorbar_title))
        if has_3d:
            fig.add_trace(go.Scatter(
                x=x, y=y, mode='markers', name=gene,
                marker=marker_2d, visible=visible, showlegend=False,
            ), row=1, col=1)
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z, mode='markers',
                marker=dict(size=marker_size, color=vals, colorscale=colorscale,
                            cmin=cmin, cmax=cmax, opacity=marker_opacity,
                            showscale=False),
                visible=visible, showlegend=False,
            ), row=1, col=2)
        else:
            fig.add_trace(go.Scatter(
                x=x, y=y, mode='markers', name=gene,
                marker=marker_2d, visible=visible, showlegend=False,
            ))

    # archetype overlay traces (always visible — added after gene traces)
    n_gene_traces = len(genes) * traces_per_gene
    if aa is not None:
        noc = aa.shape[1]
        lg = 'archetypes'
        if has_3d:
            _add_archetype_2d(fig, aa, noc, lg, row=1, col=1, show_legend=True)
            _add_archetype_3d(fig, aa, noc, lg, row=1, col=2)
        else:
            _add_archetype_2d(fig, aa, noc, lg, row=None, col=None, show_legend=True)

    n_total = len(fig.data)
    buttons = []
    for i, gene in enumerate(genes):
        vis = [False] * n_total
        for k in range(traces_per_gene):
            vis[i * traces_per_gene + k] = True
        # archetype traces are always visible
        for j in range(n_gene_traces, n_total):
            vis[j] = True
        buttons.append(dict(
            label=gene, method='update',
            args=[{'visible': vis}, {'title': f'{gene} — {title}'}],
        ))

    layout_kwargs = dict(
        title=f'{genes[0]} — {title}',
        width=width if not has_3d else max(width, 1100), height=height,
        updatemenus=[dict(
            type='dropdown', buttons=buttons,
            x=0.0, xanchor='left', y=1.07, yanchor='top',
            bgcolor='white', bordercolor='grey', font=dict(size=12),
        )],
    )
    if has_3d:
        fig.update_xaxes(title_text=xlabel, row=1, col=1)
        fig.update_yaxes(title_text=ylabel, row=1, col=1)
        fig.update_layout(**{_scene_key(2): dict(
            xaxis_title=xlabel, yaxis_title=ylabel, zaxis_title=zlabel)})
    else:
        layout_kwargs['xaxis_title'] = xlabel
        layout_kwargs['yaxis_title'] = ylabel
    fig.update_layout(**layout_kwargs)
    fig.write_html(out_path)
    print(f"  Saved {out_path}")


def gene_expr_boxplot_html(df, genes, sample_col, condition_col, sample_order,
                           condition_colors, title, out_path,
                           yaxis_title='log2(CP10k + 1)', width=600, height=600):
    """Save interactive boxplot HTML with a dropdown to switch between genes.

    df: DataFrame with sample_col, condition_col, and one column per gene.
    genes: list of gene names (must be columns in df).
    sample_order: ordered list of sample names for the x-axis.
    condition_colors: dict mapping condition -> hex color string.
    """
    conditions = list(condition_colors.keys())
    samples_by_condition = {
        c: [s for s in sample_order if df.loc[df[condition_col] == c, sample_col].isin([s]).any()]
        for c in conditions
    }

    all_traces = []
    gene_trace_ranges = {}

    for gene in genes:
        start = len(all_traces)
        for condition in conditions:
            for i, sample in enumerate(samples_by_condition[condition]):
                mask = df[sample_col] == sample
                all_traces.append(go.Box(
                    x=df.loc[mask, sample_col],
                    y=df.loc[mask, gene],
                    name=condition,
                    legendgroup=condition,
                    showlegend=(i == 0),
                    marker_color=condition_colors[condition],
                    boxpoints='outliers',
                    width=0.75,
                    visible=False,
                ))
        gene_trace_ranges[gene] = (start, len(all_traces))

    # make first gene visible
    first_start, first_end = gene_trace_ranges[genes[0]]
    for i in range(first_start, first_end):
        all_traces[i].visible = True

    fig = go.Figure(data=all_traces)

    buttons = []
    n_total = len(all_traces)
    for gene in genes:
        start, end = gene_trace_ranges[gene]
        vis = [start <= i < end for i in range(n_total)]
        buttons.append(dict(label=gene, method='update',
                            args=[{'visible': vis}, {'title': f'{gene} — {title}'}]))

    fig.update_layout(
        title=f'{genes[0]} — {title}',
        xaxis=dict(title='Sample', categoryorder='array', categoryarray=sample_order, tickangle=45),
        yaxis_title=yaxis_title,
        boxmode='group',
        width=width,
        height=height,
        legend_title='Condition',
        updatemenus=[dict(
            type='dropdown',
            buttons=buttons,
            x=0.0, xanchor='left', y=1.07, yanchor='top',
            bgcolor='white', bordercolor='grey', font=dict(size=12),
        )],
    )
    fig.write_html(out_path)
    print(f"  Saved {out_path}")
