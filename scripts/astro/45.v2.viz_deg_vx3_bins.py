"""Visualize NR-vs-DR DEGs across 4 equal-size VX3 bins — HARDENED (v2) tables.

Same figures as 45.viz_deg_vx3_bins.py but driven by the hardened DEG tables from
45.v2.deg_vx3_bins_nr_vs_dr_lmm_hardened.py. The v2 tables carry a `status` column
and only assign an FDR to trustworthy ('ok') fits, so the volcano plots show only
genes with stable, identifiable LMM fits — the unstable_se / singular_re cloud that
polluted the v1 volcanoes is excluded.

Reproduces the binning pipeline of 45.deg_vx3_bins_nr_vs_dr.py to recover per-cell
bin labels, condition, and log1p(CP10k) expression, then makes:
  1. Volcano plots (1x4 grid, one per bin) from the v2 'all' tables ('ok' genes only).
  2. Boxplots of selected top-significant genes (top-N per bin, union),
     NR vs DR across all 4 VX3 bins, with a gene dropdown.

Reads:
  links/astro/cheng22_astro.h5ad
  local_data/res/astro/26.combined_labels.parquet
  local_data/res/astro/41.cheng22_nr_harmony_arch12.h5ad
  local_data/res/astro/45.v2.deg_vx3_bin{1..4}_NR_vs_DR_{all,sig}.tsv
Outputs:
  local_data/fig/astro/45.v2.volcano_vx3_bins_NR_vs_DR.html
  local_data/fig/astro/45.v2.boxplot_vx3_bins_NR_vs_DR.html
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from scomics.utils import norm

INPUT_H5AD         = os.path.join(PROJECT_ROOT, 'links', 'astro', 'cheng22_astro.h5ad')
IN_COMBINED_LABELS = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '26.combined_labels.parquet')
IN_NR_H5AD         = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '41.cheng22_nr_harmony_arch12.h5ad')
DEG_DIR            = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
FIG_DIR            = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
OUT_VOLCANO        = os.path.join(FIG_DIR, '45.v2.volcano_vx3_bins_NR_vs_DR.html')
OUT_BOXPLOT        = os.path.join(FIG_DIR, '45.v2.boxplot_vx3_bins_NR_vs_DR.html')

DEG_ALL_TMPL = os.path.join(DEG_DIR, '45.v2.deg_vx3_bin{b}_NR_vs_DR_all.tsv')
DEG_SIG_TMPL = os.path.join(DEG_DIR, '45.v2.deg_vx3_bin{b}_NR_vs_DR_sig.tsv')

LABELED_AGES  = ['P28', 'P28_dl', 'P28_dr', 'P38', 'P38_dr']
NR_AGES       = ['P28', 'P38']
DR_AGES       = ['P28_dr', 'P38_dr']
MIN_CELLS     = 50
N_BINS        = 4
FDR_THRESH    = 0.05
LOG2FC_THRESH = np.log2(1.5)
TOP_N_PER_BIN = 5
N_LABEL       = 10  # top genes labeled per volcano panel
MANUAL_GENES  = ['Rfx4', 'Mertk']  # always included in the boxplot, listed first

COLOR_UP = '#d62728'  # up in DR (positive log2FC)
COLOR_DN = '#1f77b4'  # up in NR (negative log2FC)
COLOR_NS = '#aaaaaa'
CONDITION_COLORS = {'NR': '#4C72B0', 'DR': '#DD8452'}

os.makedirs(FIG_DIR, exist_ok=True)

# =============================================================================
# Step 1 — recompute bins + per-cell expression (mirrors script 45 lines 48-143)
# =============================================================================
print(f'Loading NR embedding from {IN_NR_H5AD}')
adata_nr = ad.read_h5ad(IN_NR_H5AD)
nr_barcodes  = adata_nr.obs_names.values
vx_load      = np.array(adata_nr.varm['VX_loadings'], dtype=np.float64)
hvg_names    = adata_nr.var_names.values
vx_scores_nr = np.array(adata_nr.obsm['X_vx'], dtype=np.float64)
pca_mean     = np.array(adata_nr.X.toarray() if hasattr(adata_nr.X, 'toarray')
                        else adata_nr.X, dtype=np.float64).mean(axis=0)
samples_nr   = adata_nr.obs['Sample'].values
print(f'  {len(nr_barcodes)} NR cells, {len(hvg_names)} HVGs, {vx_load.shape[1]} VX dims')

print(f'Loading arch labels from {IN_COMBINED_LABELS}')
df_combined = pd.read_parquet(IN_COMBINED_LABELS)
labels_c22  = df_combined[df_combined['dataset'] == 'cheng22'].reset_index(drop=True)

print(f'Loading {INPUT_H5AD}')
adata = ad.read_h5ad(INPUT_H5AD)
print(f'  {adata.shape[0]} cells × {adata.shape[1]} genes')

mt_mask = np.array([g.lower().startswith('mt-') for g in adata.var_names])
adata = adata[:, ~mt_mask].copy()

ages = adata.obs['Age'].values
age_counts = pd.Series(ages).value_counts()
valid_ages = age_counts[age_counts > MIN_CELLS].index
adata = adata[np.isin(ages, valid_ages)].copy()
ages = adata.obs['Age'].values
print(f'  {adata.shape[0]} cells after MT + age filter')

x = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X, dtype=np.float64)
depths = x.sum(axis=1)
gene_names = adata.var_names.tolist()

hvg_col_idx = np.array([np.where(adata.var_names == g)[0][0] for g in hvg_names])

print('Normalizing all cells...')
xn_all = norm(x[:, hvg_col_idx], depths)

labeled_mask = np.isin(ages, LABELED_AGES)
assert len(labels_c22) == labeled_mask.sum()
labeled_idx = np.where(labeled_mask)[0]

all_barcodes = adata.obs_names.values
barcode_to_idx = {b: i for i, b in enumerate(all_barcodes)}
nr_idx_in_adata = np.array([barcode_to_idx[b] for b in nr_barcodes])

dr_mask_in_labeled = np.isin(ages[labeled_mask], DR_AGES)
dr_idx_in_adata    = labeled_idx[dr_mask_in_labeled]
samples_dr         = adata.obs['Sample'].values[dr_idx_in_adata]
print(f'  NR cells: {len(nr_idx_in_adata)}, DR cells: {len(dr_idx_in_adata)}')

print('Regressing library size out of DR cells...')
xn_dr = xn_all[dr_idx_in_adata].copy()
log_depth_dr = np.log(depths[dr_idx_in_adata]).reshape(-1, 1)
reg_dr = LinearRegression().fit(log_depth_dr, xn_dr)
xn_dr = xn_dr - reg_dr.predict(log_depth_dr)

print('Projecting DR cells onto NR VX space...')
vx_scores_dr = (xn_dr - pca_mean) @ vx_load

n_nr = len(vx_scores_nr)
n_dr = len(vx_scores_dr)
vx3_combined    = np.concatenate([vx_scores_nr[:, 2], vx_scores_dr[:, 2]])
condition_code  = np.array([0] * n_nr + [1] * n_dr)
sample_combined = np.concatenate([samples_nr, samples_dr])
combined_idx    = np.concatenate([nr_idx_in_adata, dr_idx_in_adata])

print(f'Binning into {N_BINS} equal-size bins within each sample...')
bin_labels = np.full(len(vx3_combined), -1, dtype=int)
for sample in np.unique(sample_combined):
    mask = sample_combined == sample
    bin_labels[mask] = pd.qcut(vx3_combined[mask], q=N_BINS, labels=False, duplicates='drop')

x_comb       = x[combined_idx]
depths_comb  = depths[combined_idx]
logcpm_all   = np.log1p(x_comb / depths_comb.reshape(-1, 1) * 1e4).astype(np.float32)
gene_to_col  = {g: i for i, g in enumerate(gene_names)}

# =============================================================================
# Step 2 — volcano plots (1x4 grid, one per bin) — trustworthy ('ok') genes only
# =============================================================================
print('Building volcano plots...')
fig_v = make_subplots(
    rows=1, cols=N_BINS,
    subplot_titles=[f'Bin {b + 1}' for b in range(N_BINS)],
    horizontal_spacing=0.04,
)

for b in range(N_BINS):
    col = b + 1
    deg_df = pd.read_csv(DEG_ALL_TMPL.format(b=b + 1), sep='\t')
    n_total_genes = len(deg_df)
    # keep only stable, identifiable fits (FDR is assigned only to 'ok' genes)
    deg_df = deg_df[(deg_df['status'] == 'ok') & deg_df['fdr'].notna()].copy()
    n_excluded = n_total_genes - len(deg_df)
    deg_df['neg_log10_fdr'] = -np.log10(deg_df['fdr'].clip(lower=1e-300))

    sig_up = (deg_df['fdr'] < FDR_THRESH) & (deg_df['log2FC'] >  LOG2FC_THRESH)
    sig_dn = (deg_df['fdr'] < FDR_THRESH) & (deg_df['log2FC'] < -LOG2FC_THRESH)
    ns     = ~(sig_up | sig_dn)

    for sub, color, name in [(ns, COLOR_NS, 'n.s.'),
                             (sig_dn, COLOR_DN, 'up in NR'),
                             (sig_up, COLOR_UP, 'up in DR')]:
        fig_v.add_trace(go.Scatter(
            x=deg_df.loc[sub, 'log2FC'], y=deg_df.loc[sub, 'neg_log10_fdr'],
            mode='markers', name=name, legendgroup=name,
            showlegend=(b == 0),
            marker=dict(color=color, size=5 if name == 'n.s.' else 6,
                        opacity=0.5 if name == 'n.s.' else 0.8),
            text=deg_df.loc[sub, 'gene'],
            hovertemplate='%{text}<br>log2FC=%{x:.2f}<br>-log10(FDR)=%{y:.1f}<extra></extra>',
        ), row=1, col=col)

    # threshold lines (per-subplot axes)
    xref = 'x' if col == 1 else f'x{col}'
    yref = 'y' if col == 1 else f'y{col}'
    x_lo = deg_df['log2FC'].min() - 0.3
    x_hi = deg_df['log2FC'].max() + 0.3
    y_hi = deg_df['neg_log10_fdr'].max() * 1.05
    fig_v.add_shape(type='line', x0=x_lo, x1=x_hi,
                    y0=-np.log10(FDR_THRESH), y1=-np.log10(FDR_THRESH),
                    line=dict(color='black', width=1, dash='dash'), xref=xref, yref=yref)
    for xthr in (-LOG2FC_THRESH, LOG2FC_THRESH):
        fig_v.add_shape(type='line', x0=xthr, x1=xthr, y0=0, y1=y_hi,
                        line=dict(color='black', width=1, dash='dash'), xref=xref, yref=yref)

    # label top genes by FDR
    sig_df = deg_df[sig_up | sig_dn]
    for _, r in sig_df.nsmallest(N_LABEL, 'fdr').iterrows():
        fig_v.add_annotation(x=r['log2FC'], y=r['neg_log10_fdr'], text=r['gene'],
                             showarrow=False, font=dict(size=9), xshift=6, yshift=4,
                             xref=xref, yref=yref)

    fig_v.layout.annotations[b].text = (
        f'Bin {b + 1} (up DR: {int(sig_up.sum())}, up NR: {int(sig_dn.sum())}; '
        f'{len(deg_df)} ok / {n_excluded} excl)')
    fig_v.update_xaxes(title_text='log2FC (DR / NR)', row=1, col=col)
    if col == 1:
        fig_v.update_yaxes(title_text='-log10(FDR)', row=1, col=col)

fig_v.update_layout(
    title=(f'NR vs DR DEGs across VX3 bins — hardened LMM (status==ok only; '
           f'FDR<{FDR_THRESH}, |log2FC|>log2(1.5))'),
    width=1600, height=450, legend=dict(itemsizing='constant'),
)
fig_v.write_html(OUT_VOLCANO)
print(f'  Saved {OUT_VOLCANO}')

# =============================================================================
# Step 3 — boxplots across bins (gene dropdown)
# =============================================================================
print('Selecting top-N-per-bin genes...')
genes, seen = list(MANUAL_GENES), set(MANUAL_GENES)
for b in range(N_BINS):
    df_sig = pd.read_csv(DEG_SIG_TMPL.format(b=b + 1), sep='\t')
    for gene in df_sig.head(TOP_N_PER_BIN)['gene'].tolist():
        if gene not in seen:
            genes.append(gene)
            seen.add(gene)

missing = [g for g in genes if g not in gene_to_col]
if missing:
    print(f'  Warning: genes not found, skipping: {missing}')
    genes = [g for g in genes if g in gene_to_col]
print(f'  Genes to plot ({len(genes)}): {genes}')

conditions = ['NR', 'DR']
all_traces = []
gene_trace_ranges = {}

# x-axis order: Bin1 NR, Bin1 DR, (spacer), Bin2 NR, Bin2 DR, ...
x_order = []
for b in range(N_BINS):
    for cond in conditions:
        x_order.append(f'Bin {b + 1} {cond}')
    x_order.append(f'_Bin{b + 1}')  # invisible spacer
x_order = x_order[:-1]

for gene in genes:
    col_idx = gene_to_col[gene]
    expr = logcpm_all[:, col_idx]
    start = len(all_traces)
    for b in range(N_BINS):
        for cond in conditions:
            cc = conditions.index(cond)
            mask = (bin_labels == b) & (condition_code == cc)
            all_traces.append(go.Box(
                x=[f'Bin {b + 1} {cond}'] * int(mask.sum()),
                y=expr[mask],
                name=cond, legendgroup=cond,
                showlegend=(b == 0),
                marker_color=CONDITION_COLORS[cond],
                boxpoints='outliers', visible=False,
            ))
    gene_trace_ranges[gene] = (start, len(all_traces))

first_start, first_end = gene_trace_ranges[genes[0]]
for i in range(first_start, first_end):
    all_traces[i].visible = True

fig_b = go.Figure(data=all_traces)
n_total = len(all_traces)
buttons = []
for gene in genes:
    start, end = gene_trace_ranges[gene]
    vis = [start <= i < end for i in range(n_total)]
    buttons.append(dict(label=gene, method='update',
                        args=[{'visible': vis},
                              {'title': f'{gene} — NR vs DR across VX3 bins | cheng22 astrocytes'}]))

fig_b.update_layout(
    title=f'{genes[0]} — NR vs DR across VX3 bins | cheng22 astrocytes',
    xaxis=dict(
        title='', categoryorder='array', categoryarray=x_order,
        tickmode='array', tickvals=x_order,
        ticktext=['' if v.startswith('_') else v for v in x_order],
        tickangle=45,
    ),
    yaxis_title='log(CP10k + 1)',
    boxmode='overlay', width=950, height=550,
    legend_title='Condition',
    updatemenus=[dict(
        type='dropdown', buttons=buttons,
        x=0.0, xanchor='left', y=1.07, yanchor='top',
        bgcolor='white', bordercolor='grey', font=dict(size=12),
    )],
)
fig_b.write_html(OUT_BOXPLOT)
print(f'  Saved {OUT_BOXPLOT}')
print('Done.')
