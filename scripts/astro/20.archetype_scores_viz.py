# Visualization of archetype scores (from script 19).
# Renders per-age archetype score scatter for selected time points.

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz import gene_expr_scatter_html

SCRIPTS_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
RES_DIR               = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
FIG_DIR               = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
PARQUET_SCORES_IN     = os.path.join(RES_DIR, '19.archetype_scores.parquet')
ARCHETYPE_VERTICES_IN = os.path.join(RES_DIR, '17.archetype_vertices_knn.parquet')
HTML_PER_AGE          = os.path.join(FIG_DIR, '20.archetype_scores_{age}.html')

SCATTER_AGES = ['P0', 'P7', 'P14', 'P21', 'P28', 'P56']
PANELS       = [(0, 1, 'PC1', 'PC3'), (0, 2, 'PC1', 'PC4'), (1, 2, 'PC3', 'PC4')]
PANEL_3D     = (0, 1, 2, 'PC1', 'PC3', 'PC4')

os.makedirs(FIG_DIR, exist_ok=True)

df_all      = pd.read_parquet(PARQUET_SCORES_IN)
df_vertices = pd.read_parquet(ARCHETYPE_VERTICES_IN)
aa_vis      = df_vertices.values.T[:3, :]   # (3, NOC)

pc_cols    = [c for c in df_all.columns if c.startswith('PC')]
score_cols = [c for c in df_all.columns if c.endswith('_score')]

for age_val in SCATTER_AGES:
    df_age    = df_all[df_all['age'] == age_val].reset_index(drop=True)
    xp_age    = df_age[pc_cols].values
    gene_vals = {col: df_age[col].values for col in score_cols}

    out_path = HTML_PER_AGE.format(age=age_val)
    gene_expr_scatter_html(
        x=None, y=None,
        xp=xp_age,
        gene_vals=gene_vals,
        aa=aa_vis,
        title=f'{age_val} — archetype scores (top 10 P56 marker genes, z-scored log1p)',
        out_path=out_path,
        panels=PANELS,
        panel_3d=PANEL_3D,
        colorbar_title='archetype score (z)',
    )
    print(f'  Saved {out_path}')
