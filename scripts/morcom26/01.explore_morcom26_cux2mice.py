import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import anndata as ad

# --- file paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(PROJECT_ROOT, 'local_data', 'raw', 'morcom26_cux2mice', 'P26_ENs.h5ad')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'morcom26_cux2mice')

# --- config ---
SAMPLE_COL = 'samples'
CONDITION_COL = 'Condition'
CELLTYPE_COL = 'celltype'
CONDITION_COLORS = {'WT': '#4878CF', 'Null': '#D65F5F'}

# --- load ---
print(f'Loading {INPUT_FILE}')
adata = ad.read_h5ad(INPUT_FILE)
obs = adata.obs.copy()
print(f'  {adata.shape[0]} cells x {adata.shape[1]} genes')
print(f'  Conditions: {obs[CONDITION_COL].value_counts().to_dict()}')
print(f'  Samples: {obs[SAMPLE_COL].value_counts().to_dict()}')
print(f'  Cell types: {obs[CELLTYPE_COL].nunique()}')

os.makedirs(FIG_DIR, exist_ok=True)

# order samples consistently
SAMPLE_ORDER = sorted(obs[SAMPLE_COL].unique())

# --- Fig 1: cell type abundance ---
print('Plotting Fig 1: cell type abundance...')
ct_counts = obs[CELLTYPE_COL].value_counts()

fig, ax = plt.subplots(figsize=(5, 7))
ax.barh(ct_counts.index[::-1], ct_counts.values[::-1], color='steelblue')
ax.set_xlabel('Number of cells')
ax.set_title('Cell type abundance')
sns.despine(ax=ax)
fig.tight_layout()
OUT1 = os.path.join(FIG_DIR, '01.celltype_abundance.png')
fig.savefig(OUT1, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f'  Saved {OUT1}')

# --- Fig 2: cell type composition by sample and condition ---
print('Plotting Fig 2: cell type composition...')
ct_order = ct_counts.index.tolist()  # same order as Fig 1 (most to least abundant)
cmap = plt.get_cmap('tab20', len(ct_order))
ct_colors = {ct: mcolors.to_hex(cmap(i)) for i, ct in enumerate(ct_order)}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, group_col, group_order in [
    (axes[0], SAMPLE_COL, SAMPLE_ORDER),
    (axes[1], CONDITION_COL, ['WT', 'Null']),
]:
    fracs = (
        obs.groupby([group_col, CELLTYPE_COL], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=ct_order, fill_value=0)
    )
    fracs = fracs.div(fracs.sum(axis=1), axis=0)
    fracs = fracs.reindex(group_order)

    bottom = np.zeros(len(group_order))
    for ct in ct_order:
        vals = fracs[ct].values
        ax.bar(group_order, vals, bottom=bottom, color=ct_colors[ct], label=ct, width=0.6)
        bottom += vals

    ax.set_ylabel('Fraction of cells')
    ax.set_title(f'Cell type composition by {group_col}')
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=30)
    sns.despine(ax=ax)

# single legend outside the right panel
handles = [plt.Rectangle((0, 0), 1, 1, color=ct_colors[ct]) for ct in ct_order]
axes[1].legend(handles, ct_order, bbox_to_anchor=(1.01, 1), loc='upper left',
               fontsize=7, frameon=False)
fig.tight_layout()
OUT2 = os.path.join(FIG_DIR, '01.celltype_composition.png')
fig.savefig(OUT2, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f'  Saved {OUT2}')

# --- Fig 3: QC metrics per sample ---
print('Plotting Fig 3: QC metrics...')
QC_COLS = ['total_counts', 'n_genes_by_counts', 'pct_counts_mt', 'doublet_scores']
QC_LABELS = ['Total counts', 'Genes detected', 'Mt %', 'Doublet score']

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
sample_conditions = obs.drop_duplicates(SAMPLE_COL).set_index(SAMPLE_COL)[CONDITION_COL].to_dict()
palette = {s: CONDITION_COLORS[sample_conditions[s]] for s in SAMPLE_ORDER}

for ax, col, label in zip(axes, QC_COLS, QC_LABELS):
    sample_cond = obs[SAMPLE_COL].map(sample_conditions)
    sns.violinplot(
        data=obs.assign(_cond=sample_cond), x=SAMPLE_COL, y=col, order=SAMPLE_ORDER,
        hue='_cond', hue_order=['WT', 'Null'], palette=CONDITION_COLORS,
        ax=ax, inner='box', linewidth=0.8, legend=False,
    )
    ax.set_xlabel('')
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.tick_params(axis='x', rotation=40)
    sns.despine(ax=ax)

# condition legend
for cond, color in CONDITION_COLORS.items():
    axes[0].plot([], [], color=color, linewidth=4, label=cond)
axes[0].legend(title='Condition', fontsize=8, frameon=False)

fig.tight_layout()
OUT3 = os.path.join(FIG_DIR, '01.qc_metrics.png')
fig.savefig(OUT3, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f'  Saved {OUT3}')

# --- Fig 4: QC scatter total_counts vs n_genes ---
print('Plotting Fig 4: QC scatter...')
fig, ax = plt.subplots(figsize=(5, 4))

for cond in ['WT', 'Null']:
    mask = obs[CONDITION_COL] == cond
    # singlets
    sing = mask & ~obs['predicted_doublets'].astype(bool)
    ax.scatter(obs.loc[sing, 'total_counts'], obs.loc[sing, 'n_genes_by_counts'],
               s=1, alpha=0.3, color=CONDITION_COLORS[cond], label=f'{cond} singlet', rasterized=True)
    # doublets
    doub = mask & obs['predicted_doublets'].astype(bool)
    ax.scatter(obs.loc[doub, 'total_counts'], obs.loc[doub, 'n_genes_by_counts'],
               s=3, alpha=0.6, color=CONDITION_COLORS[cond], marker='x',
               label=f'{cond} doublet', rasterized=True)

ax.set_xlabel('Total counts')
ax.set_ylabel('Genes detected')
ax.set_title('QC: counts vs genes')
ax.legend(fontsize=7, markerscale=3, frameon=False)
sns.despine(ax=ax)
fig.tight_layout()
OUT4 = os.path.join(FIG_DIR, '01.qc_scatter.png')
fig.savefig(OUT4, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f'  Saved {OUT4}')

print('Done.')
