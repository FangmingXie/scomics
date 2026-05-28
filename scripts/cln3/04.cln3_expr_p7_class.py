# Cln3, Slc17a7, Gad1 expression in P7 neurons from DevVIS dataset.
# Barplot: mean log(CP10k+1) per class label (Glut vs GABA), dropdown to switch gene.

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
OUT_HTML     = os.path.join(OUT_FIG_DIR, '04.cln3_expr_p7_class.html')

os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- config ---
GENES       = ['Cln3', 'Slc17a7', 'Gad1']
AGE         = 'P7'
GLUT_CLASSES = ['IT Glut', 'nonIT Glut']
GABA_CLASSES = ['CNU-MGE GABA', 'CTX-CGE GABA', 'CTX-MGE GABA']
CLASS_ORDER  = GLUT_CLASSES + GABA_CLASSES
CLASS_TYPE   = {c: 'Glut' for c in GLUT_CLASSES} | {c: 'GABA' for c in GABA_CLASSES}
TYPE_COLOR   = {'Glut': '#d62728', 'GABA': '#1f77b4'}

# --- load and filter ---
print('Loading h5ad...')
adata = ad.read_h5ad(H5AD_FILE)

mask = (adata.obs['Age'] == AGE) & (adata.obs['class_label'].isin(CLASS_ORDER))
adata = adata[mask].copy()
print(f'P7 Glut+GABA cells: {adata.shape[0]}')

depths = np.asarray(adata.X.sum(axis=1)).flatten()

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
        'expr':        expr,
        'class_label': adata.obs['class_label'].values,
    })
    agg = (df.groupby('class_label', observed=True)['expr']
             .agg(['mean', 'sem', 'count'])
             .reindex(CLASS_ORDER)
             .reset_index())
    agg['mean'] = agg['mean'].fillna(0)
    agg['sem']  = agg['sem'].fillna(0)
    return agg

gene_aggs = {}
for gene in GENES:
    print(f'Computing {gene}...')
    gene_aggs[gene] = get_agg(gene)

# --- build figure: N_genes × N_types traces (one Bar trace per broad type per gene) ---
# Each Bar trace holds all class bars of that type so the legend groups cleanly.
broad_types = ['Glut', 'GABA']
n_types  = len(broad_types)
n_genes  = len(GENES)

fig = go.Figure()
for g_idx, gene in enumerate(GENES):
    agg    = gene_aggs[gene]
    active = (g_idx == 0)
    for btype in broad_types:
        sub = agg[agg['class_label'].map(CLASS_TYPE) == btype]
        fig.add_trace(go.Bar(
            x=sub['class_label'],
            y=sub['mean'],
            name=btype,
            marker_color=TYPE_COLOR[btype],
            error_y=dict(type='data', array=sub['sem'].values, visible=True),
            hovertemplate='%{x}<br>Mean: %{y:.3f}<extra>' + btype + '</extra>',
            visible=active,
            showlegend=active,
        ))

# --- dropdown ---
n_traces = n_genes * n_types

def make_trace_updates(gene_idx):
    visible    = [False] * n_traces
    showlegend = [False] * n_traces
    for t in range(n_types):
        i = gene_idx * n_types + t
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
            {'title': f'{gene} expression in {AGE} neurons by class'},
        ],
    ))

fig.update_layout(
    title=f'{GENES[0]} expression in {AGE} neurons by class',
    xaxis_title='Class',
    yaxis_title='Mean log(CP10k+1)',
    xaxis_tickangle=-30,
    barmode='group',
    legend_title='Type',
    height=500,
    width=700,
    template='plotly_white',
    margin=dict(b=120, t=100),
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
