# P56 astrocyte archetype-label scatter as a vectorized PDF (plots only).
#
# Companion to 18.4 with the same setup (P56-only, cached coords + vertices, three PC
# panels), but cells are colored by their kNN-transferred archetype label rather than by
# gene expression. Plotting-only: no recomputation of labels, PCA, or archetypes.
# Scatter points are rasterized; axes/text/archetype overlay/legend stay vector.

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz import save_archetype_scatter_pdf

SCRIPTS_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
RES_DIR               = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
FIG_DIR               = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
PARQUET_ALL_IN        = os.path.join(RES_DIR, '17.labels_all_ages.parquet')
ARCHETYPE_VERTICES_IN = os.path.join(RES_DIR, '17.archetype_vertices_knn.parquet')
PDF_ARCH_SCATTER      = os.path.join(FIG_DIR, '18.5.archetype_scatter.pdf')

# (col_x, col_y, xlabel, ylabel) — retained cols: 0=PC1, 1=PC3, 2=PC4 (PC2 dropped)
PANELS = [(0, 1, 'PC1', 'PC3'), (0, 2, 'PC1', 'PC4'), (1, 2, 'PC3', 'PC4')]

os.makedirs(FIG_DIR, exist_ok=True)

# --- load cached coords + archetype vertices (P56 joint PCA, PC2 dropped) ---
df_all  = pd.read_parquet(PARQUET_ALL_IN)
pc_cols = [c for c in df_all.columns if c.startswith('PC')]

df_vertices = pd.read_parquet(ARCHETYPE_VERTICES_IN)
aa_vis = df_vertices.values.T[:3, :]  # (3, NOC) — rows PC1, PC3, PC4

df_p56 = df_all[df_all['age'] == 'P56'].reset_index(drop=True)
xp_p56 = df_p56[pc_cols].values
arch_p56 = df_p56['archetype'].values
arch_order = sorted(pd.unique(arch_p56))
print(f'Plotting {len(arch_order)} archetypes {arch_order} for {xp_p56.shape[0]} P56 cells')

save_archetype_scatter_pdf(
    xp=xp_p56,
    labels=arch_p56,
    panels=PANELS,
    aa=aa_vis,
    label_order=arch_order,
    title='P56 astrocytes (joint PCA, no PC2) NOC=4 — archetype labels',
    out_path=PDF_ARCH_SCATTER,
)
print('Done.')
