# Boxplot of gene expression (NR vs DR) per archetype with gene dropdown.
# Uses archetype labels from 26.combined_labels.parquet and raw counts from cheng22 h5ad.
# Genes: manual list + top 3 NR-vs-DR DEGs per archetype (from script 29 outputs).

import os
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import plotly.graph_objects as go
from natsort import natsorted

# --- file paths ---
SCRIPTS_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT      = os.path.dirname(SCRIPTS_DIR)
PARQUET_COMBINED  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '26.combined_labels.parquet')
INPUT_CHENG22     = os.path.join(PROJECT_ROOT, 'links', 'astro', 'cheng22_astro.h5ad')
DEG_DIR           = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
FIG_DIR           = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
OUT_HTML          = os.path.join(FIG_DIR, '30.gene_boxplot_NR_vs_DR.html')
OUT_HTML_BAR      = os.path.join(FIG_DIR, '30.deg_count_barplot_NR_vs_DR.html')

# --- config ---
CHENG22_AGES  = ['P28', 'P28_dl', 'P28_dr', 'P38', 'P38_dr']  # same subset as script 26
NR_AGES       = ['P28', 'P38']
DR_AGES       = ['P28_dr', 'P38_dr']
ARCHETYPES    = natsorted(['Arch1', 'Arch2', 'Arch3', 'Arch4'])
MANUAL_GENES  = ['Rfx4', 'Mertk', 'Trpm3', 'Cdh13', 'Cst3', 'Gfap']
TOP_N_DEGS    = 3
CONDITION_COLORS = {'NR': '#4C72B0', 'DR': '#DD8452'}
LOG2FC_THRESH    = 0.5

os.makedirs(FIG_DIR, exist_ok=True)

# --- build gene list: manual + top N DEGs per archetype ---
genes = list(MANUAL_GENES)
seen = set(genes)
for arch in ARCHETYPES:
    sig_path = os.path.join(DEG_DIR, f'29.deg_{arch}_NR_vs_DR_sig.tsv')
    df_sig = pd.read_csv(sig_path, sep='\t')
    # top N upregulated (highest log2FC, sorted by fdr already)
    top_up = df_sig[df_sig['log2FC'] > 0].head(TOP_N_DEGS)['gene'].tolist()
    # top N downregulated (most negative log2FC)
    top_dn = df_sig[df_sig['log2FC'] < 0].nsmallest(TOP_N_DEGS, 'log2FC')['gene'].tolist()
    for gene in top_up + top_dn:
        if gene not in seen:
            genes.append(gene)
            seen.add(gene)
print(f'Genes to plot ({len(genes)}): {genes}')

# --- load & align metadata ---
print('Loading combined parquet...')
df_combined = pd.read_parquet(PARQUET_COMBINED)
df_cheng22_meta = df_combined[df_combined['dataset'] == 'cheng22'].copy()

print(f'Loading {INPUT_CHENG22}...')
adata_full = ad.read_h5ad(INPUT_CHENG22)
adata_cheng22 = adata_full[adata_full.obs['Age'].isin(CHENG22_AGES)].copy()

if len(df_cheng22_meta) != adata_cheng22.shape[0]:
    raise ValueError(
        f'Cell count mismatch: parquet {len(df_cheng22_meta)} vs h5ad {adata_cheng22.shape[0]}'
    )
df_cheng22_meta = df_cheng22_meta.copy()
df_cheng22_meta['cell_barcode'] = adata_cheng22.obs_names.values

# subset to NR/DR
keep = df_cheng22_meta['age'].isin(NR_AGES + DR_AGES)
df_meta = df_cheng22_meta[keep].copy()
df_meta['condition'] = df_meta['age'].apply(lambda x: 'DR' if x in DR_AGES else 'NR')
df_meta = df_meta.reset_index(drop=True)

# --- normalize and extract gene expression ---
adata_sub = adata_cheng22[df_meta['cell_barcode'].values].copy()
X = adata_sub.X
if sp.issparse(X):
    X = X.toarray()
X = X.astype(np.float32)
logcpm = np.log1p(X / X.sum(axis=1, keepdims=True) * 1e4)

missing = [g for g in genes if g not in adata_sub.var_names]
if missing:
    print(f'  Warning: genes not found in h5ad, skipping: {missing}')
    genes = [g for g in genes if g in adata_sub.var_names]

for gene in genes:
    idx = adata_sub.var_names.get_loc(gene)
    df_meta[gene] = logcpm[:, idx]

# --- plot ---
print('Building boxplot...')
conditions = list(CONDITION_COLORS.keys())
all_traces = []
gene_trace_ranges = {}

# x-axis order: Arch1 NR, Arch1 DR, (gap), Arch2 NR, Arch2 DR, ...
x_order = []
for arch in ARCHETYPES:
    for cond in conditions:
        x_order.append(f'{arch} {cond}')
    x_order.append(f'_{arch}')  # invisible spacer between archetype groups
x_order = x_order[:-1]  # drop trailing spacer

for gene in genes:
    start = len(all_traces)
    for i, arch in enumerate(ARCHETYPES):
        for cond in conditions:
            mask = (df_meta['archetype'] == arch) & (df_meta['condition'] == cond)
            all_traces.append(go.Box(
                x=[f'{arch} {cond}'] * mask.sum(),
                y=df_meta.loc[mask, gene].values,
                name=cond,
                legendgroup=cond,
                showlegend=(i == 0),
                marker_color=CONDITION_COLORS[cond],
                boxpoints='outliers',
                visible=False,
            ))
    gene_trace_ranges[gene] = (start, len(all_traces))

# make first gene visible
first_start, first_end = gene_trace_ranges[genes[0]]
for i in range(first_start, first_end):
    all_traces[i].visible = True

fig = go.Figure(data=all_traces)

n_total = len(all_traces)
buttons = []
for gene in genes:
    start, end = gene_trace_ranges[gene]
    vis = [start <= i < end for i in range(n_total)]
    buttons.append(dict(label=gene, method='update',
                        args=[{'visible': vis}, {'title': f'{gene} — NR vs DR | cheng22 astrocytes'}]))

fig.update_layout(
    title=f'{genes[0]} — NR vs DR | cheng22 astrocytes',
    xaxis=dict(
        title='',
        categoryorder='array', categoryarray=x_order,
        tickmode='array',
        tickvals=x_order,
        ticktext=['' if v.startswith('_') else v for v in x_order],
        tickangle=45,
    ),
    yaxis_title='log(CP10k + 1)',
    boxmode='overlay',
    width=850,
    height=550,
    legend_title='Condition',
    updatemenus=[dict(
        type='dropdown',
        buttons=buttons,
        x=0.0, xanchor='left', y=1.07, yanchor='top',
        bgcolor='white', bordercolor='grey', font=dict(size=12),
    )],
)
fig.write_html(OUT_HTML)
print(f'  Saved {OUT_HTML}')

# --- barplot: count of significant DEGs per archetype ---
print('Building DEG count barplot...')
up_counts, dn_counts = [], []
for arch in ARCHETYPES:
    sig_path = os.path.join(DEG_DIR, f'29.deg_{arch}_NR_vs_DR_sig.tsv')
    df_sig = pd.read_csv(sig_path, sep='\t')
    up_counts.append((df_sig['log2FC'] > LOG2FC_THRESH).sum())
    dn_counts.append((df_sig['log2FC'] < -LOG2FC_THRESH).sum())

fig_bar = go.Figure([
    go.Bar(name='Up (DR>NR)', x=ARCHETYPES, y=up_counts,
           marker_color=CONDITION_COLORS['DR']),
    go.Bar(name='Down (NR>DR)', x=ARCHETYPES, y=[-n for n in dn_counts],
           marker_color=CONDITION_COLORS['NR']),
])
fig_bar.update_layout(
    title='Significant DEGs per archetype (FDR<0.05, |log2FC|>1) — NR vs DR',
    xaxis_title='Archetype',
    yaxis_title='Gene count (up positive, down negative)',
    barmode='overlay',
    width=650, height=450,
)
fig_bar.write_html(OUT_HTML_BAR)
print(f'  Saved {OUT_HTML_BAR}')
print('Done.')
