"""Volcano plot of whole-population NR-vs-DR pseudobulk DESeq2 DEGs as a PDF (v5, plots only).

Standalone: reads only the v5 DESeq2 table (no per-cell pipeline). x-axis = shrunk
log2FC (DESeq2 apeglm), y-axis = -log10(FDR / padj). Genes with FDR = NaN (DESeq2
independent filtering / Cook's outliers) are dropped. Scatter points are rasterized;
axes/text/threshold lines/gene labels stay vector.

Reads:
  local_data/res/astro/45.v5.deg_nr_vs_dr_all.tsv
Outputs:
  local_data/fig/astro/45.v5.volcano_nr_vs_dr.pdf
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz import save_volcano_pdf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
FIG_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
IN_DEG_ALL   = os.path.join(RES_DIR, '45.v5.deg_nr_vs_dr_all.tsv')
OUT_PDF      = os.path.join(FIG_DIR, '45.v5.volcano_nr_vs_dr.pdf')

FDR_THRESH    = 0.05
LOG2FC_THRESH = 1.0
N_LABEL       = 5   # top genes labeled per direction (up in DR + up in NR)
HIGHLIGHT     = ['Rfx4']  # extra genes to always label + ring

os.makedirs(FIG_DIR, exist_ok=True)

deg_df = pd.read_csv(IN_DEG_ALL, sep='\t')
deg_df = deg_df[deg_df['fdr'].notna()].copy()   # drop independently-filtered genes
print(f'Loaded {len(deg_df)} genes (FDR not NaN) from {IN_DEG_ALL}')

save_volcano_pdf(
    deg_df,
    title=('NR vs DR pseudobulk DESeq2 DEGs — all astrocytes (Arch1-4)\n'
           f'(FDR<{FDR_THRESH}, |log2FC_shrink|>{LOG2FC_THRESH:g})'),
    out_path=OUT_PDF,
    fdr_thresh=FDR_THRESH,
    log2fc_thresh=LOG2FC_THRESH,
    n_label=N_LABEL,
    highlight_genes=HIGHLIGHT,
)
print('Done.')
