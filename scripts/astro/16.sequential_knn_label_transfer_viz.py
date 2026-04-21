# Visualization of sequential kNN archetype label transfer (from script 15).
# Loads combined parquet; renders per-age archetype scatter + all-ages scatter + abundance barplot.

import os
import sys
import numpy as np
import pandas as pd
from natsort import natsorted

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz import scatter_categorical_html, stacked_bar_html

SCRIPTS_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
RES_DIR        = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
FIG_DIR        = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
PARQUET_ALL_IN = os.path.join(RES_DIR, '15.labels_all_ages.parquet')
HTML_PER_AGE   = os.path.join(FIG_DIR, '16.scatter_{age}.html')
HTML_ALL_AGES  = os.path.join(FIG_DIR, '16.scatter_all_ages.html')
HTML_BARPLOT   = os.path.join(FIG_DIR, '16.archetype_abundance.html')

SCATTER_AGES = ['P0', 'P7', 'P14', 'P21', 'P28', 'P56']

os.makedirs(FIG_DIR, exist_ok=True)

df_all = pd.read_parquet(PARQUET_ALL_IN)
print(f'Loaded {len(df_all)} cells from {PARQUET_ALL_IN}')

pc_cols = [c for c in df_all.columns if c.startswith('PC')]

# Per-age scatter colored by archetype (PC1 vs PC3 → drop PC2 at index 1)
for age_val in SCATTER_AGES:
    df_age = df_all[df_all['age'] == age_val].reset_index(drop=True)
    xp_age = np.delete(df_age[pc_cols].values, 1, axis=1)  # drop PC2

    scatter_categorical_html(
        xp_grid=[xp_age],
        cell_metadata={
            'archetype':  df_age['archetype'].values,
            'donor_name': df_age['donor_name'].values,
        },
        title=f'{age_val} — kNN-transferred archetype labels (P56 joint PCA, no PC2)',
        out_path=HTML_PER_AGE.format(age=age_val),
        xlabel='PC1', ylabel='PC3', zlabel='PC4',
    )
    print(f'  Saved {HTML_PER_AGE.format(age=age_val)}')

# All-ages scatter colored by archetype or age
xp_all = np.delete(df_all[pc_cols].values, 1, axis=1)  # drop PC2

scatter_categorical_html(
    xp_grid=[xp_all],
    cell_metadata={
        'archetype':  df_all['archetype'].values,
        'age':        df_all['age'].values,
        'donor_name': df_all['donor_name'].values,
    },
    title='All postnatal ages — kNN-transferred archetype labels (P56 joint PCA, no PC2)',
    out_path=HTML_ALL_AGES,
    ordered_labels=('age',),
    xlabel='PC1', ylabel='PC3', zlabel='PC4',
)
print(f'Saved {HTML_ALL_AGES}')

# Archetype abundance barplot across all ages
age_order = natsorted(df_all['age'].unique())
archetype_order = sorted(df_all['archetype'].unique())

counts = df_all.groupby(['age', 'archetype']).size().unstack(fill_value=0)
counts = counts.reindex(columns=archetype_order, fill_value=0)
frac = counts.div(counts.sum(axis=1), axis=0)

stacked_bar_html(
    panel_data=[('Archetype fraction by age', age_order, frac)],
    celltypes=archetype_order,
    title='Archetype abundance across postnatal ages (sequential kNN transfer)',
    out_path=HTML_BARPLOT,
    panel_width=1000,
)
print(f'Saved {HTML_BARPLOT}')
