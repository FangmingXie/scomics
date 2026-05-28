# Cln3, Slc17a7, Gad1 expression in P56 neuronal cells from DevVIS dataset.
# Barplot: mean log(CP10k+1) per subclass, colored by class, dropdown to switch gene.

import os
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import plotly.graph_objects as go

# --- file paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H5AD_FILE    = os.path.join(PROJECT_ROOT, 'links', 'broad', 'DevVIS_scRNA_processed.h5ad')
OUT_FIG_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'cln3')
OUT_HTML     = os.path.join(OUT_FIG_DIR, '01.cln3_expr_p56_subclass.html')

os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- config ---
GENES            = ['Cln3', 'Slc17a7', 'Gad1']
AGE              = 'P56'
NEURONAL_CLASSES = ['CNU-MGE GABA', 'CTX-CGE GABA', 'CTX-MGE GABA', 'IMN', 'IT Glut', 'nonIT Glut']
COLORS           = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# --- load and filter ---
print('Loading h5ad...')
adata = ad.read_h5ad(H5AD_FILE)

mask = (adata.obs['Age'] == AGE) & (adata.obs['class_label'].isin(NEURONAL_CLASSES))
adata = adata[mask].copy()
print(f'P56 neuronal cells: {adata.shape[0]}')

# Fixed subclass order — consistent x-axis across all genes
meta = (adata.obs[['class_label', 'subclass_label']]
        .drop_duplicates()
        .sort_values(['class_label', 'subclass_label'])
        .reset_index(drop=True))
subclass_order = meta['subclass_label'].tolist()
subclass_to_class = dict(zip(meta['subclass_label'], meta['class_label']))

# Precompute cell depths once
depths = np.asarray(adata.X.sum(axis=1)).flatten()

sorted_classes = sorted(NEURONAL_CLASSES)
class_color    = dict(zip(sorted_classes, COLORS))

# --- aggregate mean ± SEM per gene ---
def get_agg(gene):
    if gene not in adata.var_names:
        raise ValueError(f'Gene {gene} not found in var_names')
    gene_idx = list(adata.var_names).index(gene)
    X_gene = adata.X[:, gene_idx]
    if sp.issparse(X_gene):
        X_gene = np.asarray(X_gene.todense()).flatten()
    else:
        X_gene = np.array(X_gene).flatten()
    expr = np.log1p(X_gene / depths * 1e4)

    df = pd.DataFrame({
        'expr':           expr,
        'subclass_label': adata.obs['subclass_label'].values,
        'class_label':    adata.obs['class_label'].values,
    })
    agg = (df.groupby(['class_label', 'subclass_label'], observed=True)['expr']
             .agg(['mean', 'sem'])
             .reset_index()
             .set_index('subclass_label')
             .reindex(subclass_order)
             .reset_index())
    agg['class_label'] = agg['subclass_label'].map(subclass_to_class)
    agg['mean'] = agg['mean'].fillna(0)
    agg['sem']  = agg['sem'].fillna(0)
    return agg

gene_aggs = {}
for gene in GENES:
    print(f'Computing {gene}...')
    gene_aggs[gene] = get_agg(gene)

# --- build figure: N_genes × N_classes traces ---
n_classes = len(sorted_classes)
n_genes   = len(GENES)

fig = go.Figure()
for g_idx, gene in enumerate(GENES):
    agg     = gene_aggs[gene]
    active  = (g_idx == 0)
    for cls in sorted_classes:
        sub = agg[agg['class_label'] == cls]
        fig.add_trace(go.Bar(
            x=sub['subclass_label'],
            y=sub['mean'],
            name=cls,
            marker_color=class_color[cls],
            error_y=dict(type='data', array=sub['sem'].values, visible=True),
            hovertemplate='%{x}<br>Mean: %{y:.3f}<extra>' + cls + '</extra>',
            visible=active,
            showlegend=active,
        ))

# --- dropdown ---
def make_trace_updates(gene_idx):
    n_traces = n_genes * n_classes
    visible     = [False] * n_traces
    showlegend  = [False] * n_traces
    for c in range(n_classes):
        i = gene_idx * n_classes + c
        visible[i]    = True
        showlegend[i] = True
    return visible, showlegend

buttons = []
for g_idx, gene in enumerate(GENES):
    vis, show_leg = make_trace_updates(g_idx)
    buttons.append(dict(
        label=gene,
        method='update',
        args=[
            {'visible': vis, 'showlegend': show_leg},
            {'title': f'{gene} expression in {AGE} neuronal cells by subclass'},
        ],
    ))

fig.update_layout(
    title=f'{GENES[0]} expression in {AGE} neuronal cells by subclass',
    xaxis_title='Subclass',
    yaxis_title='Mean log(CP10k+1)',
    xaxis_tickangle=-45,
    barmode='group',
    legend_title='Class',
    height=550,
    width=1100,
    template='plotly_white',
    margin=dict(b=160, t=100),
    updatemenus=[dict(
        type='dropdown',
        direction='down',
        x=0.0,
        xanchor='left',
        y=1.18,
        yanchor='top',
        buttons=buttons,
        showactive=True,
    )],
)

fig.write_html(OUT_HTML)
print(f'Saved: {OUT_HTML}')
