# Visualization of cheng22-only cross-dataset astrocyte integration with P28_dl (from script 26).
# Loads combined parquet; renders per-dataset PCA scatter, all-datasets PCA scatter,
# all-datasets UMAP scatter, and archetype abundance barplot.

import os
import sys
import numpy as np
import pandas as pd
from natsort import natsorted

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz import scatter_categorical_html, stacked_bar_html

SCRIPTS_DIR           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT          = os.path.dirname(SCRIPTS_DIR)
PARQUET_COMBINED_IN   = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '26.combined_labels.parquet')
ARCHETYPE_VERTICES_IN = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '26.archetype_vertices.parquet')
FIG_DIR               = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
HTML_PER_DATASET      = os.path.join(FIG_DIR, '27.scatter_{dataset}.html')
HTML_ALL_DATASETS     = os.path.join(FIG_DIR, '27.scatter_all_datasets.html')
HTML_UMAP_ALL         = os.path.join(FIG_DIR, '27.umap_all_datasets.html')
HTML_BARPLOT          = os.path.join(FIG_DIR, '27.archetype_abundance.html')

DATASETS    = ['gao25', 'cheng22']
PANELS      = [(0, 1, 'PC1', 'PC2'), (0, 2, 'PC1', 'PC3'), (1, 2, 'PC2', 'PC3')]
PANEL_3D    = (0, 1, 2, 'PC1', 'PC2', 'PC3')
UMAP_PANELS = [(0, 1, 'UMAP1', 'UMAP2')]

os.makedirs(FIG_DIR, exist_ok=True)

df_all = pd.read_parquet(PARQUET_COMBINED_IN)
print(f'Loaded {len(df_all)} cells from {PARQUET_COMBINED_IN}')
print(f'  Datasets: {df_all["dataset"].value_counts().to_dict()}')

pc_cols   = [c for c in df_all.columns if c.startswith('PC')]
umap_cols = ['UMAP1', 'UMAP2']

# Load archetype centroid vertices (PCA space)
df_vertices = pd.read_parquet(ARCHETYPE_VERTICES_IN)
aa_vis = df_vertices.values.T[:3, :]   # (3, NOC) — first 3 dims used by PANELS
print(f'Loaded archetype vertices:\n{df_vertices}')

# --- Per-dataset PCA scatter ---
for dataset in DATASETS:
    df_ds = df_all[df_all['dataset'] == dataset].reset_index(drop=True)
    xp_ds = df_ds[pc_cols].values

    meta = {
        'archetype':  df_ds['archetype'].values,
        'age':        df_ds['age'].values,
        'donor_name': df_ds['donor_name'].values,
    }
    if dataset != 'gao25':
        meta['cell_type'] = df_ds['cell_type'].values

    scatter_categorical_html(
        xp_grid=[xp_ds],
        cell_metadata=meta,
        title=f'{dataset} — archetype labels (cross-dataset kNN transfer from gao25 P56)',
        out_path=HTML_PER_DATASET.format(dataset=dataset),
        panels=PANELS,
        panel_3d=PANEL_3D,
        arch_vis=aa_vis,
    )
    print(f'  Saved {HTML_PER_DATASET.format(dataset=dataset)}')

# --- All-datasets PCA scatter ---
xp_all = df_all[pc_cols].values

scatter_categorical_html(
    xp_grid=[xp_all],
    cell_metadata={
        'archetype':  df_all['archetype'].values,
        'dataset':    df_all['dataset'].values,
        'age':        df_all['age'].values,
        'donor_name': df_all['donor_name'].values,
    },
    title='All datasets — archetype labels in shared PCA space (gao25 P56 reference)',
    out_path=HTML_ALL_DATASETS,
    panels=PANELS,
    panel_3d=PANEL_3D,
    arch_vis=aa_vis,
)
print(f'Saved {HTML_ALL_DATASETS}')

# --- All-datasets UMAP scatter ---
umap_all = df_all[umap_cols].values

scatter_categorical_html(
    xp_grid=[umap_all],
    cell_metadata={
        'archetype':  df_all['archetype'].values,
        'dataset':    df_all['dataset'].values,
        'age':        df_all['age'].values,
        'donor_name': df_all['donor_name'].values,
    },
    title='All datasets — UMAP (joint embedding, gao25 P56 reference)',
    out_path=HTML_UMAP_ALL,
    panels=UMAP_PANELS,
)
print(f'Saved {HTML_UMAP_ALL}')

# --- Archetype abundance barplot (two panels: by age and by biological replicate) ---
archetype_order = sorted(df_all['archetype'].unique())

# Explicit group order: cheng22 P28 → P38 → P28DR → P38DR → P28DL, then gao25
AGE_ORDER = ['cheng22_P28', 'cheng22_P38', 'cheng22_P28_dr', 'cheng22_P38_dr', 'cheng22_P28_dl', 'gao25_P56']

df_all['_age_group'] = df_all['dataset'] + '_' + df_all['age']
age_groups  = [g for g in AGE_ORDER if g in df_all['_age_group'].values]
counts_age  = df_all.groupby(['_age_group', 'archetype']).size().unstack(fill_value=0)
counts_age  = counts_age.reindex(index=age_groups, columns=archetype_order, fill_value=0)
frac_age    = counts_age.div(counts_age.sum(axis=1), axis=0)

# Sort replicates by AGE_ORDER position, then natsort within each group
donor_age_idx = df_all.drop_duplicates('donor_name').set_index('donor_name')['_age_group'] \
                      .map({g: i for i, g in enumerate(AGE_ORDER)})
rep_groups  = sorted(df_all['donor_name'].unique(),
                     key=lambda d: (donor_age_idx.get(d, len(AGE_ORDER)), d))
counts_rep  = df_all.groupby(['donor_name', 'archetype']).size().unstack(fill_value=0)
counts_rep  = counts_rep.reindex(index=rep_groups, columns=archetype_order, fill_value=0)
frac_rep    = counts_rep.div(counts_rep.sum(axis=1), axis=0)

stacked_bar_html(
    panel_data=[
        ('By age', age_groups, frac_age),
        ('By biological replicate', rep_groups, frac_rep),
    ],
    celltypes=archetype_order,
    title='Archetype abundance (cross-dataset kNN transfer)',
    out_path=HTML_BARPLOT,
    panel_width=1200,
    vertical=True,
)
print(f'Saved {HTML_BARPLOT}')
