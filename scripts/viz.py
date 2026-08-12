# Shared visualization functions for archetype analysis scripts.

import os
import itertools
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from natsort import natsorted


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _write_fig(fig, out_path, screenshot_format='svg'):
    """Write a plotly figure to HTML.

    The saved HTML's modebar download ("screenshot") button exports a static image
    in `screenshot_format` — SVG vector by default instead of the plotly default
    PNG. The image is rendered client-side by plotly.js in the browser, so no
    kaleido / server-side image export is needed.
    """
    config = {'toImageButtonOptions': {'format': screenshot_format}}
    fig.write_html(out_path, config=config)
    print(f"  Saved {out_path}")


def _rgba(color, alpha):
    """Return an 'rgba(r,g,b,alpha)' string for any matplotlib-recognized color."""
    r, g, b, _ = mcolors.to_rgba(color)
    return f'rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{alpha})'


def _add_band(fig, x, mean, std, color, alpha=0.3):
    """Add a translucent mean±std band (fill_between equivalent) as two helper traces."""
    mean = np.asarray(mean)
    std = np.asarray(std)
    fig.add_trace(go.Scatter(x=x, y=mean + std, mode='lines', line=dict(width=0),
                             showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x, y=mean - std, mode='lines', line=dict(width=0),
                             fill='tonexty', fillcolor=_rgba(color, alpha),
                             showlegend=False, hoverinfo='skip'))


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


def _add_archetype_2d(fig, aa, noc, lg, row, col, *, show_legend, cx=0, cy=1):
    """Add 2D archetype diamond markers and closing polygon (shared legendgroup lg)."""
    fig.add_trace(go.Scatter(
        x=aa[cx, :], y=aa[cy, :], mode='markers',
        marker=dict(size=8, color='black', symbol='diamond'),
        name=lg, showlegend=show_legend, legendgroup=lg,
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=list(aa[cx, :]) + [aa[cx, 0]], y=list(aa[cy, :]) + [aa[cy, 0]], mode='lines',
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


def _add_archetype_3d_scene(fig, aa, noc, lg, scene, color='black', marker_size=6):
    """Like _add_archetype_3d but targets a named scene directly (bypasses row/col routing).

    Use this for 3D subplots with colspan > 1 where row/col routing is unreliable.
    """
    fig.add_trace(go.Scatter3d(
        x=aa[0, :], y=aa[1, :], z=aa[2, :], mode='markers',
        marker=dict(size=marker_size, color=color, symbol='diamond'),
        showlegend=False, legendgroup=lg, scene=scene,
    ))
    ex, ey, ez = _archetype_edges_3d(aa, noc)
    fig.add_trace(go.Scatter3d(
        x=ex, y=ey, z=ez, mode='lines',
        line=dict(color=color, width=2),
        showlegend=False, legendgroup=lg, scene=scene,
    ))


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def save_score_scatter_pdf(xp, scores, names, aa, title, out_path,
                           cmap='RdBu_r', pctile=(5, 95), s=3, dpi=300,
                           aa_labels=None, colorbar_title='archetype score [0–1]',
                           vlims=None, axis_labels=('PC1', 'PC2')):
    """Save a per-score PC1-vs-PC2 scatter as a vectorized PDF (rasterized points).

    One panel per score (column of `scores`). Points are drawn with rasterized=True so
    the dense cloud is a single embedded bitmap, while axes/text/archetype overlay stay
    vector. Per-panel color scale is clipped at the `pctile` (low, high) of each score.

    xp:             (n_cells, >=2) coordinate array; columns 0,1 used as PC1, PC2.
    scores:         (n_cells, n_scores) array; each column colors one panel.
    names:          list of score names (one per scores column), used in panel titles.
    aa:             (n_archetypes, >=2) archetype coords (PC space); diamonds + polygon.
    aa_labels:      optional list of labels (one per aa row) annotated next to each diamond.
    cmap:           colormap shared by all panels, or one per panel (name or Colormap).
    colorbar_title: colorbar label; a str shared by all panels, or one str per panel.
    vlims:          optional list, one entry per panel: (vmin, vmax) to override the
                    percentile clipping for that panel, or None to keep it (e.g. use
                    (-x, x) to center a difference panel's diverging colormap at zero).
    axis_labels:    (x, y) axis labels; defaults to ('PC1', 'PC2') for the PCHA embedding.
    """
    plt.rcParams['pdf.fonttype'] = 42   # editable vector text
    scores = np.asarray(scores)
    n = len(names)
    if aa_labels is not None and len(aa_labels) != len(aa):
        raise ValueError(f'aa_labels has {len(aa_labels)} entries but aa has {len(aa)} rows')
    cbar_titles = [colorbar_title] * n if isinstance(colorbar_title, str) else list(colorbar_title)
    if len(cbar_titles) != n:
        raise ValueError(f'colorbar_title has {len(cbar_titles)} entries but there are {n} panels')
    cmaps = [cmap] * n if isinstance(cmap, (str, mcolors.Colormap)) else list(cmap)
    if len(cmaps) != n:
        raise ValueError(f'cmap has {len(cmaps)} entries but there are {n} panels')
    if vlims is not None and len(vlims) != n:
        raise ValueError(f'vlims has {len(vlims)} entries but there are {n} panels')
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4), squeeze=False)
    for k, name in enumerate(names):
        ax = axes[0, k]
        vals = scores[:, k]
        vmin, vmax = (vlims[k] if vlims is not None and vlims[k] is not None
                      else np.percentile(vals, pctile))
        sc = ax.scatter(xp[:, 0], xp[:, 1], c=vals, cmap=cmaps[k], vmin=vmin, vmax=vmax,
                        s=s, linewidths=0, rasterized=True)
        # archetype overlay (vector): diamonds + closing polygon
        ax.plot(list(aa[:, 0]) + [aa[0, 0]], list(aa[:, 1]) + [aa[0, 1]],
                '-', color='black', linewidth=1.0)
        ax.scatter(aa[:, 0], aa[:, 1], marker='D', color='black', s=30, zorder=3)
        if aa_labels is not None:
            for (ax_, ay_), label in zip(aa[:, :2], aa_labels):
                ax.annotate(label, (ax_, ay_), textcoords='offset points', xytext=(5, 5),
                            fontsize=8, fontweight='bold', color='black', zorder=4)
        ax.set_aspect('equal', adjustable='box')   # equal x/y scaling (true geometry)
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_title(f'Score {name}')
        fig.colorbar(sc, ax=ax, label=cbar_titles[k], shrink=0.8)
        sns.despine(ax=ax)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    print(f"  Saved {out_path}")


def save_score_grid_pdf(rows, col_names, scores, title, out_path,
                        cmap='RdBu_r', pctile=(5, 95), s=1.5, dpi=300,
                        panel_size=2.0, colorbar_title='archetype score [0–1]',
                        highlight=None):
    """Save a rows x columns grid of score scatters with one shared colorbar per column.

    `save_score_scatter_pdf` draws a single row and gives every panel its own colorbar and
    its own percentile limits, which makes panels comparable only within a score. Here each
    column's (vmin, vmax) is taken from the pooled values of that column across *all* rows,
    so a column reads as "where does this score live" across the different embeddings —
    which requires the caller to have computed the scores on one common scale.

    rows:      list of dicts, one per row: 'label' (row title), 'xp' (n_cells, >=2)
               coordinates, 'aa' (n_archetypes, >=2) archetype coords, 'aa_labels'
               (optional list, one per aa row). Rows may differ in cell count.
    col_names: list of column titles.
    scores:    scores[i][j] is the length-n_cells_i score vector for row i, column j.
    highlight: optional set of (row, col) index pairs drawn with contrasting spines.
    """
    plt.rcParams['pdf.fonttype'] = 42   # editable vector text
    n_rows, n_cols = len(rows), len(col_names)
    if len(scores) != n_rows or any(len(r) != n_cols for r in scores):
        raise ValueError(f'scores must be {n_rows} x {n_cols}, got '
                         f'{len(scores)} x {[len(r) for r in scores]}')
    for i, row in enumerate(rows):
        for j in range(n_cols):
            if len(scores[i][j]) != len(row['xp']):
                raise ValueError(f'scores[{i}][{j}] has {len(scores[i][j])} values but row '
                                 f'{i} ({row["label"]}) has {len(row["xp"])} cells')
    highlight = set() if highlight is None else set(highlight)

    # per-column limits over the pooled rows — the point of the figure
    vlims = [np.percentile(np.concatenate([scores[i][j] for i in range(n_rows)]), pctile)
             for j in range(n_cols)]

    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, layout='constrained',
                             figsize=(panel_size * n_cols, panel_size * n_rows))
    for j, col_name in enumerate(col_names):
        vmin, vmax = vlims[j]
        for i, row in enumerate(rows):
            ax = axes[i, j]
            xp, aa = row['xp'], row['aa']
            sc = ax.scatter(xp[:, 0], xp[:, 1], c=scores[i][j], cmap=cmap,
                            vmin=vmin, vmax=vmax, s=s, linewidths=0, rasterized=True)
            # archetype overlay (vector): diamonds + closing polygon
            ax.plot(list(aa[:, 0]) + [aa[0, 0]], list(aa[:, 1]) + [aa[0, 1]],
                    '-', color='black', linewidth=0.8)
            ax.scatter(aa[:, 0], aa[:, 1], marker='D', color='black', s=12, zorder=3)
            for (ax_, ay_), label in zip(aa[:, :2], row.get('aa_labels') or []):
                ax.annotate(label, (ax_, ay_), textcoords='offset points', xytext=(3, 3),
                            fontsize=6, fontweight='bold', color='black', zorder=4)
            ax.set_aspect('equal', adjustable='box')   # one embedding per row
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(col_name, fontsize=9)
            if j == 0:
                ax.set_ylabel(row['label'], fontsize=9)
            for spine in ax.spines.values():
                spine.set_visible((i, j) in highlight)
                spine.set_color('#e41a1c')
                spine.set_linewidth(1.5)
        cb = fig.colorbar(sc, ax=axes[:, j].tolist(), location='bottom',
                          shrink=0.9, aspect=12, pad=0.01)
        cb.ax.tick_params(labelsize=6)
        cb.set_label(colorbar_title, fontsize=6)
    fig.suptitle(f'{title}\ncolor scale shared down each column', fontsize=11)
    fig.savefig(out_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    print(f"  Saved {out_path}")


def save_gene_scatter_pdf(xp, gene_vals, panels, aa, title, out_path,
                          cmap='RdBu_r', pctile=(9, 95), s=5, dpi=300,
                          colorbar_title='z-score'):
    """Save a multi-page gene-expression scatter PDF (one page per gene).

    Static, vectorized counterpart of `gene_expr_scatter_html`. Each page renders the
    same gene across all `panels` (e.g. PC1-PC3, PC1-PC4, PC3-PC4); points are drawn
    rasterized so each dense cloud is a single embedded bitmap, while axes/text/archetype
    overlay stay vector. Per-gene color scale is clipped at the `pctile` (low, high).

    xp:        (n_cells, n_dims) coordinate array.
    gene_vals: dict[str, np.ndarray] mapping gene name -> per-cell value (e.g. z-score).
    panels:    list of (col_x, col_y, xlabel, ylabel) into xp columns.
    aa:        (n_dims, n_archetypes) archetype coords; diamonds + closing polygon overlay.
    """
    plt.rcParams['pdf.fonttype'] = 42   # editable vector text
    n_panels = len(panels)
    with PdfPages(out_path) as pdf:
        for gene, vals in gene_vals.items():
            vals = np.asarray(vals)
            vmin, vmax = np.nanpercentile(vals, pctile)
            fig, axes = plt.subplots(1, n_panels, figsize=(4.6 * n_panels, 4.2),
                                     squeeze=False)
            for pi, (cx, cy, xl, yl) in enumerate(panels):
                ax = axes[0, pi]
                sc = ax.scatter(xp[:, cx], xp[:, cy], c=vals, cmap=cmap,
                                vmin=vmin, vmax=vmax, s=s, linewidths=0, rasterized=True)
                if aa is not None:
                    ax.plot(list(aa[cx, :]) + [aa[cx, 0]], list(aa[cy, :]) + [aa[cy, 0]],
                            '-', color='black', linewidth=1.0)
                    ax.scatter(aa[cx, :], aa[cy, :], marker='D', color='black',
                               s=30, zorder=3)
                ax.set_xlabel(xl)
                ax.set_ylabel(yl)
                ax.set_title(f'{xl} vs {yl}')
                sns.despine(ax=ax)
            fig.colorbar(sc, ax=axes[0, :].tolist(), label=colorbar_title, shrink=0.8)
            fig.suptitle(f'{gene} — {title}')
            pdf.savefig(fig, bbox_inches='tight', dpi=dpi)
            plt.close(fig)
    print(f"  Saved {out_path}")


def save_archetype_scatter_pdf(xp, labels, panels, aa, title, out_path,
                               label_order=None, cmap='tab10', s=5, dpi=300,
                               legend_title='archetype'):
    """Save a single-page categorical scatter PDF colored by archetype label.

    Static, vectorized counterpart of `scatter_categorical_html` (archetype coloring).
    One row of panels (e.g. PC1-PC3, PC1-PC4, PC3-PC4); each cell is colored by its
    discrete `labels` value via the `cmap` color cycle, with a shared legend. Points are
    drawn rasterized (single embedded bitmap per panel); axes/text/archetype overlay and
    legend stay vector.

    xp:          (n_cells, n_dims) coordinate array.
    labels:      (n_cells,) array of categorical labels (e.g. 'Arch1'..'Arch4').
    panels:      list of (col_x, col_y, xlabel, ylabel) into xp columns.
    aa:          (n_dims, n_archetypes) archetype coords; diamonds + closing polygon overlay.
    label_order: optional ordered list of label values (defaults to sorted unique).
    legend_title: heading over the legend. Defaults to 'archetype' — this helper is used for
                  other categorical colourings too (subclass, cluster), where that heading
                  names the wrong variable.
    """
    plt.rcParams['pdf.fonttype'] = 42   # editable vector text
    labels = np.asarray(labels)
    if label_order is None:
        label_order = sorted(np.unique(labels))
    cycle = plt.get_cmap(cmap).colors
    label_to_color = {lab: cycle[i % len(cycle)] for i, lab in enumerate(label_order)}

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.6 * n_panels, 4.2), squeeze=False)
    for pi, (cx, cy, xl, yl) in enumerate(panels):
        ax = axes[0, pi]
        for lab in label_order:
            m = labels == lab
            ax.scatter(xp[m, cx], xp[m, cy], color=label_to_color[lab], s=s,
                       linewidths=0, rasterized=True, label=lab)
        if aa is not None:
            ax.plot(list(aa[cx, :]) + [aa[cx, 0]], list(aa[cy, :]) + [aa[cy, 0]],
                    '-', color='black', linewidth=1.0)
            ax.scatter(aa[cx, :], aa[cy, :], marker='D', color='black', s=30, zorder=3)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f'{xl} vs {yl}')
        sns.despine(ax=ax)
    axes[0, -1].legend(title=legend_title, loc='center left', bbox_to_anchor=(1.02, 0.5),
                       frameon=False, markerscale=2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    print(f"  Saved {out_path}")


def save_volcano_pdf(deg_df, title, out_path,
                     fdr_thresh=0.05, log2fc_thresh=np.log2(1.5), n_label=5,
                     highlight_genes=(),
                     color_up='#d62728', color_dn='#1f77b4', color_ns='#aaaaaa',
                     s=6, dpi=300):
    """Save a single-panel NR-vs-DR volcano as a vectorized PDF (rasterized points).

    Static, matplotlib counterpart of the plotly volcano in 45.v4.viz_volcano_vx3_bins.
    x-axis = shrunk log2FC (DR / NR), y-axis = -log10(FDR). Points are drawn rasterized
    so the dense cloud is a single embedded bitmap; axes/text/threshold lines/gene labels
    stay vector. Genes are split into up-in-DR (positive log2FC, red), up-in-NR (negative,
    blue), and non-significant (gray).

    deg_df: DataFrame with columns 'gene', 'log2FC' (shrunk), 'fdr'. Rows with NaN fdr
            should already be dropped by the caller (DESeq2 independent filtering).
    n_label: number of top significant genes (smallest fdr) to annotate per direction
             (n_label up-in-DR + n_label up-in-NR).
    highlight_genes: extra gene names to always label and ring (drawn with a black edge),
             in addition to the per-direction top-n_label genes.
    """
    plt.rcParams['pdf.fonttype'] = 42   # editable vector text
    df = deg_df.copy()
    df['neg_log10_fdr'] = -np.log10(df['fdr'].clip(lower=1e-300))

    sig_up = (df['fdr'] < fdr_thresh) & (df['log2FC'] >  log2fc_thresh)
    sig_dn = (df['fdr'] < fdr_thresh) & (df['log2FC'] < -log2fc_thresh)
    ns     = ~(sig_up | sig_dn)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df.loc[ns, 'log2FC'], df.loc[ns, 'neg_log10_fdr'],
               c=color_ns, s=s * 0.6, linewidths=0, alpha=0.4, rasterized=True,
               label='n.s.')
    ax.scatter(df.loc[sig_dn, 'log2FC'], df.loc[sig_dn, 'neg_log10_fdr'],
               c=color_dn, s=s, linewidths=0, alpha=0.8, rasterized=True,
               label=f'up in NR ({int(sig_dn.sum())})')
    ax.scatter(df.loc[sig_up, 'log2FC'], df.loc[sig_up, 'neg_log10_fdr'],
               c=color_up, s=s, linewidths=0, alpha=0.8, rasterized=True,
               label=f'up in DR ({int(sig_up.sum())})')

    # threshold lines (vector)
    ax.axhline(-np.log10(fdr_thresh), color='black', lw=1, ls='--')
    ax.axvline(-log2fc_thresh, color='black', lw=1, ls='--')
    ax.axvline(log2fc_thresh, color='black', lw=1, ls='--')

    # label top-n_label per direction + any explicit highlight genes (vector text)
    label_genes = set(df[sig_up].nsmallest(n_label, 'fdr')['gene'])
    label_genes |= set(df[sig_dn].nsmallest(n_label, 'fdr')['gene'])
    label_genes |= set(highlight_genes)
    highlight = set(highlight_genes)
    for _, r in df[df['gene'].isin(label_genes)].iterrows():
        is_hl = r['gene'] in highlight
        if is_hl:   # ring the highlighted points for emphasis
            ax.scatter([r['log2FC']], [r['neg_log10_fdr']], s=s * 4,
                       facecolors='none', edgecolors='black', linewidths=1.2, zorder=4)
        ax.annotate(r['gene'], (r['log2FC'], r['neg_log10_fdr']),
                    fontsize=8, fontweight='bold' if is_hl else 'normal',
                    xytext=(3, 3), textcoords='offset points', zorder=5)

    ax.set_xlabel('log2FC (DR / NR), shrunk')
    ax.set_ylabel('-log10(FDR)')
    ax.set_title(title)
    ax.legend(frameon=False, markerscale=1.5, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    print(f"  Saved {out_path}")


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


def save_metrics_plot_html(noc_grid, ev_grid, av_grid, av_rep_grid, ndim, title, out_path):
    """Save EV / ARV / effective-EV metrics as an interactive HTML.

    Plotly equivalent of save_metrics_plot. The saved HTML's modebar download
    button exports a static SVG vector (see _write_fig).
    """
    ev_grid = np.asarray(ev_grid)
    av_grid = np.asarray(av_grid)
    av_rep_grid = np.asarray(av_rep_grid)

    marker = dict(size=10)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=noc_grid, y=ev_grid, mode='lines+markers',
                             line=dict(color='black'), marker=marker, name='explained variance (EV)'))
    fig.add_trace(go.Scatter(x=noc_grid, y=av_grid, mode='lines+markers',
                             line=dict(color='steelblue'), marker=marker, name='ARV (bootstrap)'))
    fig.add_trace(go.Scatter(x=noc_grid, y=av_rep_grid, mode='lines+markers',
                             line=dict(color='steelblue', dash='dash'), marker=marker, name='ARV_rep (per-group)'))
    fig.add_trace(go.Scatter(x=noc_grid, y=ev_grid * (1 - av_grid), mode='lines+markers',
                             line=dict(color='tomato'), marker=marker, name='effective EV (bootstrap)'))
    fig.add_trace(go.Scatter(x=noc_grid, y=ev_grid * (1 - av_rep_grid), mode='lines+markers',
                             line=dict(color='tomato', dash='dash'), marker=marker, name='effective EV (rep)'))
    axis_style = dict(showline=True, linecolor='black', ticks='outside', tickcolor='black',
                      linewidth=1.5, color='black')
    fig.update_layout(
        title=title,
        xaxis=dict(title='Number of archetypes (NOC)', tickmode='array', tickvals=list(noc_grid), **axis_style),
        yaxis=dict(title='Score', range=[0, 1], **axis_style),
        paper_bgcolor='white', plot_bgcolor='white',
        width=700, height=500,
    )
    _write_fig(fig, out_path)


def save_metrics_err_plot_html(noc_grid, ev_grid, arv_mean, arv_std, av_rep_grid,
                               effev_mean, effev_std, effev_rep_grid, ndim, title, out_path):
    """Save EV / ARV / effective-EV metrics with bootstrap std bands as interactive HTML.

    Like save_metrics_plot_html, but the bootstrap ARV and effective-EV(bootstrap)
    curves are mean ± std over repeated bootstrap runs, drawn as translucent
    fill_between-style bands (alpha=0.3). EV and the per-group (rep) curves are
    single-valued. The saved HTML's modebar download button exports a static SVG
    vector (see _write_fig).
    """
    ev_grid       = np.asarray(ev_grid)
    arv_mean      = np.asarray(arv_mean)
    arv_std       = np.asarray(arv_std)
    av_rep_grid   = np.asarray(av_rep_grid)
    effev_mean    = np.asarray(effev_mean)
    effev_std     = np.asarray(effev_std)
    effev_rep_grid = np.asarray(effev_rep_grid)

    arv_color   = mcolors.to_hex('gray')   # ARV / ARV_rep (was steelblue)
    effev_color = mcolors.to_hex('C1')     # effective EV (was tomato)
    marker = dict(size=10)
    fig = go.Figure()
    # std bands (drawn first so the mean lines sit on top)
    _add_band(fig, noc_grid, arv_mean, arv_std, arv_color)
    _add_band(fig, noc_grid, effev_mean, effev_std, effev_color)
    fig.add_trace(go.Scatter(x=noc_grid, y=ev_grid, mode='lines+markers',
                             line=dict(color='black'), marker=marker, name='explained variance (EV)'))
    fig.add_trace(go.Scatter(x=noc_grid, y=arv_mean, mode='lines+markers',
                             line=dict(color=arv_color), marker=marker, name='ARV (bootstrap)'))
    fig.add_trace(go.Scatter(x=noc_grid, y=av_rep_grid, mode='lines+markers',
                             line=dict(color=arv_color, dash='dash'), marker=marker, name='ARV_rep (per-group)'))
    fig.add_trace(go.Scatter(x=noc_grid, y=effev_mean, mode='lines+markers',
                             line=dict(color=effev_color), marker=marker, name='effective EV (bootstrap)'))
    fig.add_trace(go.Scatter(x=noc_grid, y=effev_rep_grid, mode='lines+markers',
                             line=dict(color=effev_color, dash='dash'), marker=marker, name='effective EV (rep)'))
    axis_style = dict(showline=True, linecolor='black', ticks='outside', tickcolor='black',
                      linewidth=1.5, color='black', gridcolor='lightgray')
    fig.update_layout(
        title=title,
        xaxis=dict(title='Number of archetypes (NOC)', tickmode='array', tickvals=list(noc_grid), **axis_style),
        yaxis=dict(title='Score', range=[0, 1], **axis_style),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='black', family='Arial'),
        width=700, height=500,
    )
    _write_fig(fig, out_path)


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
    fig.update_scenes(dragmode='orbit')
    _write_fig(fig, out_path)


def scatter_categorical_html(xp_grid, cell_metadata, title, out_path,
                             noc_grid=(), ev_grid=(), av_grid=(), aa_grid=(),
                             ordered_labels=(),
                             xlabel='PC1', ylabel='PC2', zlabel='PC3',
                             panels=None, panel_3d=None,
                             arch_vis=None,
                             height=None,
                             equal_aspect=False):
    """Like scatter_html but uses per-category traces so the Plotly legend shows one entry per category.

    For categorical metadata: one trace per unique value per panel; clicking a legend entry
    hides/shows that category across all panels.
    For continuous metadata: single trace per panel with viridis colorscale.

    Buttons switch which metadata label is active (all other label traces are hidden).
    Archetype traces are always visible.
    ordered_labels: collection of metadata keys that should use evenly spaced turbo colors
                    (e.g. time-ordered categories like Age).
    panels: list of (col_x, col_y, xlabel, ylabel) tuples. When provided, renders one 2D
            scatter per panel instead of the default 2D+3D layout.
            Example: [(0,1,'PC1','PC3'), (0,2,'PC1','PC4'), (1,2,'PC3','PC4')]
    panel_3d: (col_x, col_y, col_z, xlabel, ylabel, zlabel) tuple. When provided alongside
              panels, appends a 3D scatter as the last subplot.
              Example: (0,1,2,'PC1','PC3','PC4')
    arch_vis: optional (n_vis_dims, noc) array of archetype positions in the visualization
              coordinate system (e.g. centroids per archetype). When provided, overlays
              diamond markers and connecting polygon on each panel.
    """
    xp = xp_grid[0]
    noc_entries = list(zip(noc_grid, ev_grid, av_grid, aa_grid))
    cat_palette = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if panels is not None:
        n2d = len(panels)
        has_3d = panel_3d is not None
        titles_2d = [f'{xl} vs {yl}' for _, _, xl, yl in panels]
        if has_3d:
            # 3D panel on row 2, spanning all columns
            nrows, ncols = 2, n2d
            specs = [
                [{'type': 'xy'}] * n2d,
                [{'type': 'scene', 'colspan': n2d}] + [None] * (n2d - 1),
            ]
            titles_3d = [f'3D  {panel_3d[3]}–{panel_3d[5]}']
        else:
            nrows, ncols = 1, n2d
            specs = [[{'type': 'xy'}] * n2d]
            titles_3d = []
        fig = make_subplots(
            rows=nrows, cols=ncols,
            specs=specs,
            subplot_titles=titles_2d + titles_3d,
        )
    else:
        has_3d = True
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'xy'}, {'type': 'scene'}]],
            subplot_titles=[f'2D  {xlabel} vs {ylabel}', f'3D  {xlabel}–{zlabel}'],
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
            if panels is not None:
                for pi, (cx, cy, xl, yl) in enumerate(panels):
                    fig.add_trace(go.Scatter(
                        x=xp[:, cx], y=xp[:, cy], mode='markers',
                        marker=dict(size=2, color=vals, colorscale='Viridis', cmin=vmin, cmax=vmax,
                                    opacity=0.6, showscale=(pi == 0)),
                        name=label, showlegend=False, visible=visible,
                    ), row=1, col=pi + 1)
                if panel_3d is not None:
                    cx, cy, cz = panel_3d[:3]
                    fig.add_trace(go.Scatter3d(
                        x=xp[:, cx], y=xp[:, cy], z=xp[:, cz], mode='markers',
                        marker=dict(size=2, color=vals, colorscale='Viridis', cmin=vmin, cmax=vmax,
                                    opacity=0.6, showscale=False),
                        showlegend=False, visible=visible,
                    ), row=2, col=1)
            else:
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
                if panels is not None:
                    for pi, (cx, cy, xl, yl) in enumerate(panels):
                        fig.add_trace(go.Scatter(
                            x=xp[mask, cx], y=xp[mask, cy], mode='markers',
                            marker=dict(size=2, color=color, opacity=0.6),
                            name=uv, legendgroup=lg, showlegend=(pi == 0), visible=visible,
                        ), row=1, col=pi + 1)
                    if panel_3d is not None:
                        cx, cy, cz = panel_3d[:3]
                        fig.add_trace(go.Scatter3d(
                            x=xp[mask, cx], y=xp[mask, cy], z=xp[mask, cz], mode='markers',
                            marker=dict(size=2, color=color, opacity=0.6),
                            showlegend=False, legendgroup=lg, visible=visible,
                        ), row=2, col=1)
                else:
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

    if panels is not None:
        for pi, (cx, cy, xl, yl) in enumerate(panels):
            fig.update_xaxes(title_text=xl, row=1, col=pi + 1)
            fig.update_yaxes(title_text=yl, row=1, col=pi + 1)
        if panel_3d is not None:
            _, _, _, xl3, yl3, zl3 = panel_3d
            fig.update_layout(scene=dict(xaxis_title=xl3, yaxis_title=yl3, zaxis_title=zl3))
    else:
        fig.update_xaxes(title_text=xlabel, row=1, col=1)
        fig.update_yaxes(title_text=ylabel, row=1, col=1)
        fig.update_layout(**{_scene_key(2): dict(xaxis_title=xlabel, yaxis_title=ylabel, zaxis_title=zlabel)})

    arch_start = len(fig.data)
    for noc, ev, av, aa in noc_entries:
        lg = f'NOC={noc}  EV={ev:.3f}  ARV={av:.3f}'
        if panels is not None:
            for pi, (cx, cy, xl, yl) in enumerate(panels):
                _add_archetype_2d(fig, aa, noc, lg, row=1, col=pi + 1, show_legend=(pi == 0), cx=cx, cy=cy)
            if panel_3d is not None:
                _add_archetype_3d_scene(fig, aa, noc, lg, scene='scene')
        else:
            _add_archetype_2d(fig, aa, noc, lg, row=1, col=1, show_legend=True)
            _add_archetype_3d(fig, aa, noc, lg, row=1, col=2)

    if arch_vis is not None:
        noc_vis = arch_vis.shape[1]
        lg = 'Archetypes'
        if panels is not None:
            for pi, (cx, cy, xl, yl) in enumerate(panels):
                _add_archetype_2d(fig, arch_vis, noc_vis, lg, row=1, col=pi + 1, show_legend=(pi == 0), cx=cx, cy=cy)
            if panel_3d is not None:
                _add_archetype_3d_scene(fig, arch_vis, noc_vis, lg, scene='scene')
        else:
            _add_archetype_2d(fig, arch_vis, noc_vis, lg, row=1, col=1, show_legend=True)
            _add_archetype_3d(fig, arch_vis, noc_vis, lg, row=1, col=2)

    n_total = len(fig.data)

    buttons = []
    for label in labels:
        start, end = label_trace_ranges[label]
        vis = [
            True if (i >= arch_start or start <= i < end) else False
            for i in range(n_total)
        ]
        buttons.append(dict(label=label, method='update', args=[{'visible': vis}]))

    if panels is not None:
        width         = 550 * n2d
        default_height = 1100 if panel_3d is not None else 600
    else:
        width, default_height = 1100, 600
    fig.update_layout(
        title=title, width=width, height=height if height is not None else default_height,
        legend=dict(itemsizing='constant'),
        updatemenus=[dict(
            type='buttons', direction='right',
            x=0.0, xanchor='left', y=1.05, yanchor='bottom',
            buttons=buttons,
        )],
    )
    if equal_aspect and panels is not None:
        for pi in range(n2d):
            xref = 'x' if pi == 0 else f'x{pi + 1}'
            fig.update_yaxes(scaleanchor=xref, scaleratio=1, row=1, col=pi + 1)
    fig.update_scenes(dragmode='orbit')
    _write_fig(fig, out_path)


def scatter_2d_categorical_html(xp_grid, cell_metadata, title, out_path,
                                xlabel='Dim1', ylabel='Dim2', ordered_labels=(),
                                return_html=False):
    """Like scatter_categorical_html but a single 2D panel only.

    Useful for UMAP or any 2D embedding where a 3D view is not meaningful.
    xlabel/ylabel label the axes (e.g. 'UMAP1'/'UMAP2' or 'PC1'/'PC2').
    ordered_labels: collection of metadata keys that should use evenly spaced
                    viridis colors (e.g. time-ordered categories like Age).
                    All other categorical labels use the default color cycle.
    return_html: if True, return an HTML string instead of writing to out_path.
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
            type='dropdown',
            x=0.0, xanchor='left', y=1.05, yanchor='bottom',
            buttons=buttons,
        )],
    )
    if return_html:
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    _write_fig(fig, out_path)


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
    fig.update_scenes(dragmode='orbit')
    _write_fig(fig, out_path)


def stacked_bar_html(panel_data, celltypes, title, out_path, ct_colors=None, panel_width=500,
                     vertical=False):
    """Save an interactive stacked bar chart HTML with one panel per group.

    panel_data: list of (panel_title, group_order, frac_df) where frac_df is a
                DataFrame indexed by group with celltypes as columns (values in [0,1]).
    celltypes:  ordered list of cell type names (stacking order).
    ct_colors:  dict mapping celltype -> hex color string. Defaults to tab10.
    vertical:   if True, stack panels vertically (rows) instead of horizontally (cols).
    """
    if ct_colors is None:
        # cmap = cm.get_cmap('tab10', len(celltypes))
        cmap = cm.get_cmap('tab10', 10)
        ct_colors = {ct: mcolors.to_hex(cmap(i)) for i, ct in enumerate(celltypes)}

    n = len(panel_data)
    if vertical:
        fig = make_subplots(
            rows=n, cols=1,
            subplot_titles=[pd_[0] for pd_ in panel_data],
            shared_xaxes=False,
        )
    else:
        fig = make_subplots(
            rows=1, cols=n,
            subplot_titles=[pd_[0] for pd_ in panel_data],
            shared_yaxes=True,
        )

    for idx, (panel_title, group_order, frac_df) in enumerate(panel_data, start=1):
        row = idx if vertical else 1
        col = 1 if vertical else idx
        for i, ct in enumerate(celltypes):
            fig.add_trace(go.Bar(
                name=ct,
                x=group_order,
                y=frac_df.reindex(group_order)[ct].values,
                marker_color=ct_colors[ct],
                legendgroup=ct,
                showlegend=(idx == 1),
            ), row=row, col=col)

    fig.update_layout(
        barmode='stack',
        title=title,
        legend=dict(itemsizing='constant', traceorder='normal'),
        width=panel_width if vertical else panel_width * n,
        height=600 * n if vertical else 600,
    )
    fig.update_xaxes(tickangle=45)
    for i in range(1, n + 1):
        yaxis_key = 'yaxis' if i == 1 else f'yaxis{i}'
        fig.update_layout(**{yaxis_key: dict(range=[0, 1], title='Fraction of cells')})
    _write_fig(fig, out_path)


def gene_expr_scatter_html(x, y, gene_vals, title, out_path,
                           xlabel='Dim1', ylabel='Dim2',
                           z=None, zlabel='Dim3',
                           aa=None,
                           colorscale='RdBu_r',
                           pctile_low=9, pctile_high=95,
                           marker_size=3, marker_opacity=0.6,
                           colorbar_title='z-score',
                           width=850, height=700,
                           bg_color=None,
                           xp=None, panels=None, panel_3d=None,
                           return_html=False):
    """Save interactive 2D scatter HTML colored by gene expression with a gene dropdown.

    gene_vals: dict[str, np.ndarray] mapping gene name to per-cell float values (e.g. z-scores).
    z:  optional array for a third dimension; when provided, adds a 3D panel alongside the 2D one.
    aa: optional archetype coordinate array (ndim × noc); overlaid on both panels when provided.
    xp: coordinate array (n_cells, n_dims); required when panels or panel_3d is provided.
    panels: list of (col_x, col_y, xlabel, ylabel) — same semantics as scatter_categorical_html.
    panel_3d: (col_x, col_y, col_z, xlabel, ylabel, zlabel) — placed on row 2, spanning all columns.
    """
    genes = list(gene_vals.keys())

    if panels is not None:
        n2d = len(panels)
        has_3d = panel_3d is not None
        if has_3d:
            nrows, ncols = 2, n2d
            specs = [
                [{'type': 'xy'}] * n2d,
                [{'type': 'scene', 'colspan': n2d}] + [None] * (n2d - 1),
            ]
            titles_3d = [f'3D  {panel_3d[3]}–{panel_3d[5]}']
        else:
            nrows, ncols = 1, n2d
            specs = [[{'type': 'xy'}] * n2d]
            titles_3d = []
        fig = make_subplots(
            rows=nrows, cols=ncols,
            specs=specs,
            subplot_titles=[f'{xl} vs {yl}' for _, _, xl, yl in panels] + titles_3d,
        )
        traces_per_gene = n2d + (1 if has_3d else 0)
    else:
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

        if panels is not None:
            for pi, (cx, cy, xl, yl) in enumerate(panels):
                fig.add_trace(go.Scatter(
                    x=xp[:, cx], y=xp[:, cy], mode='markers', name=gene,
                    marker=dict(size=marker_size, color=vals, colorscale=colorscale,
                                cmin=cmin, cmax=cmax, opacity=marker_opacity,
                                showscale=(pi == 0), colorbar=dict(title=colorbar_title)),
                    visible=visible, showlegend=False,
                ), row=1, col=pi + 1)
            if panel_3d is not None:
                cx, cy, cz = panel_3d[:3]
                fig.add_trace(go.Scatter3d(
                    x=xp[:, cx], y=xp[:, cy], z=xp[:, cz], mode='markers',
                    marker=dict(size=marker_size, color=vals, colorscale=colorscale,
                                cmin=cmin, cmax=cmax, opacity=marker_opacity,
                                showscale=False),
                    visible=visible, showlegend=False,
                ), row=2, col=1)
        else:
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
        if panels is not None:
            for pi, (cx, cy, xl, yl) in enumerate(panels):
                _add_archetype_2d(fig, aa, noc, lg, row=1, col=pi + 1, show_legend=(pi == 0), cx=cx, cy=cy)
            if panel_3d is not None:
                _add_archetype_3d_scene(fig, aa, noc, lg, scene='scene')
        elif has_3d:
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
        for j in range(n_gene_traces, n_total):
            vis[j] = True
        buttons.append(dict(
            label=gene, method='update',
            args=[{'visible': vis}, {'title': f'{gene} — {title}'}],
        ))

    if panels is not None:
        for pi, (cx, cy, xl, yl) in enumerate(panels):
            fig.update_xaxes(title_text=xl, row=1, col=pi + 1)
            fig.update_yaxes(title_text=yl, row=1, col=pi + 1)
        if panel_3d is not None:
            _, _, _, xl3, yl3, zl3 = panel_3d
            fig.update_layout(scene=dict(xaxis_title=xl3, yaxis_title=yl3, zaxis_title=zl3))
        fig_width  = 550 * n2d
        fig_height = 1100 if panel_3d is not None else 600
    else:
        if has_3d:
            fig.update_xaxes(title_text=xlabel, row=1, col=1)
            fig.update_yaxes(title_text=ylabel, row=1, col=1)
            fig.update_layout(**{_scene_key(2): dict(
                xaxis_title=xlabel, yaxis_title=ylabel, zaxis_title=zlabel)})
        else:
            fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel)
        fig_width  = width if not has_3d else max(width, 1100)
        fig_height = height

    layout_kwargs = dict(
        title=f'{genes[0]} — {title}',
        width=fig_width, height=fig_height,
        updatemenus=[dict(
            type='dropdown', buttons=buttons,
            x=0.0, xanchor='left', y=1.07, yanchor='top',
            bgcolor='white', bordercolor='grey', font=dict(size=12),
        )],
    )
    if bg_color is not None:
        layout_kwargs['paper_bgcolor'] = bg_color
        layout_kwargs['plot_bgcolor'] = bg_color
    fig.update_layout(**layout_kwargs)
    fig.update_scenes(dragmode='orbit')
    if return_html:
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    _write_fig(fig, out_path)


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
    _write_fig(fig, out_path)
