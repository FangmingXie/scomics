"""Heatmap of top archetype markers (rows) × per (sample × archetype) pseudobulks (cols).

Variant of the 45.v6 marker heatmap that shows expression across ARCHETYPES rather than
across VX1 bins. Rows are the top archetype-specific markers; columns are per
(sample × archetype) pseudobulks, so archetype-identity genes should light up on the
diagonal (Arch-k markers high in the Arch-k columns).

Rows: for each archetype, the TOP_N_ARCH one-vs-rest markers with the largest effect size
(log2FC) from the cheng22 marker table (script 45.v6.cheng22_archetype_markers, computed on
the same cells/labels), keeping significant enriched genes (FDR<0.05, log2FC>0), excluding
technical gene classes (ribosomal, predicted Gm, mito, metallothionein), and de-duplicated
across archetypes (a gene keeps the lowest archetype it was selected for). Rows are grouped
into four archetype
blocks (Arch1..Arch4); within each block they are ordered by hierarchical clustering (ward,
optimal leaf ordering) of the z-scored per (sample × archetype) profiles.

Columns: one pseudobulk per (sample, archetype) over the same Arch1-4 NR + all-DR cells as
the 45.v6 pipeline (NR cells + archetype labels from the 41.v3 embedding; DR cells + labels
from the combined-labels parquet). Expression = log1p(CP10k) of the summed raw counts, then
z-scored PER GENE across all columns. Columns are grouped by condition (NR then DR), and
within each condition ordered by archetype (then age). No clustering.

Two panels share the rows and z-score scale: (A) the full per (sample × archetype) matrix,
and (B) a simpler summary = the mean z across samples for each condition × archetype (8 cols).

Reads:
  links/astro/cheng22_astro.h5ad
  local_data/res/astro/41.v3.cheng22_nr_harmony_arch1234.h5ad
  local_data/res/astro/26.combined_labels.parquet
  local_data/res/astro/45.v6.cheng22_archetype_markers.tsv
Outputs:
  local_data/fig/astro/45.v6.heatmap_archetype_markers_by_archetype_pseudobulk.pdf
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy.cluster.hierarchy import linkage, leaves_list
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

INPUT_H5AD         = os.path.join(PROJECT_ROOT, 'links', 'astro', 'cheng22_astro.h5ad')
IN_NR_H5AD         = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '41.v3.cheng22_nr_harmony_arch1234.h5ad')
IN_COMBINED_LABELS = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '26.combined_labels.parquet')
MARKERS_TSV        = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '45.v6.cheng22_archetype_markers.tsv')

# technical gene classes excluded from marker selection (ribosomal, predicted Gm, mito, metallothionein)
TECH_GENE_RE = re.compile(r'^(Rpl|Rps|Rplp|Mrpl|Mrps|Gm\d|mt-|Mt\d)')
FIG_DIR            = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
OUT_PDF            = os.path.join(FIG_DIR, '45.v6.heatmap_archetype_markers_by_archetype_pseudobulk.pdf')

LABELED_AGES  = ['P28', 'P28_dl', 'P28_dr', 'P38', 'P38_dr']
NR_AGES       = ['P28', 'P38']
DR_AGES       = ['P28_dr', 'P38_dr']
MIN_CELLS     = 50
MIN_PB_CELLS  = 10             # drop a (sample x archetype) pseudobulk with fewer cells

# --- row (gene) selection: top archetype markers ---
N_ARCH        = 4
TOP_N_ARCH    = 10             # top markers per archetype (by specificity, enriched log2FC > 0)

# --- column ordering: NR ages first, then DR ages ---
AGE_ORDER = {'P28': 0, 'P38': 1, 'P28_dr': 2, 'P38_dr': 3}

COLOR_NR = '#1f77b4'
COLOR_DR = '#d62728'
VMIN, VMAX = -2.5, 2.5         # z-score color clip

os.makedirs(FIG_DIR, exist_ok=True)

# =============================================================================
# Step 1 — assemble the combined NR + DR cell set with sample / condition /
#          archetype labels (same cells as the 45.v6 pipeline; no VX binning)
# =============================================================================
print(f'Loading NR cell set from {IN_NR_H5AD}')
adata_nr = ad.read_h5ad(IN_NR_H5AD)
nr_barcodes = adata_nr.obs_names.values
samples_nr  = adata_nr.obs['Sample'].values
nr_arch     = adata_nr.obs['archetype'].values

print(f'Loading arch labels from {IN_COMBINED_LABELS}')
df_combined = pd.read_parquet(IN_COMBINED_LABELS)
labels_c22  = df_combined[df_combined['dataset'] == 'cheng22'].reset_index(drop=True)

print(f'Loading {INPUT_H5AD}')
adata = ad.read_h5ad(INPUT_H5AD)

mt_mask = np.array([g.lower().startswith('mt-') for g in adata.var_names])
adata = adata[:, ~mt_mask].copy()

ages = adata.obs['Age'].values
age_counts = pd.Series(ages).value_counts()
valid_ages = age_counts[age_counts > MIN_CELLS].index
adata = adata[np.isin(ages, valid_ages)].copy()
ages = adata.obs['Age'].values
gene_names = np.array(adata.var_names)

assert adata.raw is not None, 'adata.raw (raw counts) is required'
raw_counts = adata.raw[:, adata.var_names].X
raw_counts = raw_counts.toarray() if sp.issparse(raw_counts) else np.asarray(raw_counts)
raw_counts = np.rint(raw_counts).astype(np.int64)

labeled_mask = np.isin(ages, LABELED_AGES)
labeled_idx  = np.where(labeled_mask)[0]
assert len(labels_c22) == labeled_mask.sum(), (
    f'Labeled cell count mismatch: parquet {len(labels_c22)} vs adata {labeled_mask.sum()}')
arch_labels_all = labels_c22['archetype'].values          # positional with adata labeled cells

barcode_to_idx  = {b: i for i, b in enumerate(adata.obs_names.values)}
nr_idx_in_adata = np.array([barcode_to_idx[b] for b in nr_barcodes])

dr_mask_in_labeled = np.isin(ages[labeled_mask], DR_AGES)
dr_idx_in_adata    = labeled_idx[dr_mask_in_labeled]
dr_arch            = arch_labels_all[dr_mask_in_labeled]

# combined NR + DR cells (NR archetypes from embedding, DR from parquet)
condition_code  = np.array([0] * len(nr_idx_in_adata) + [1] * len(dr_idx_in_adata))
sample_combined = np.concatenate([samples_nr, adata.obs['Sample'].values[dr_idx_in_adata]])
combined_idx    = np.concatenate([nr_idx_in_adata, dr_idx_in_adata])
age_combined    = ages[combined_idx]
arch_combined   = np.concatenate([nr_arch, dr_arch])
arch_code       = np.array([int(a.replace('Arch', '')) for a in arch_combined])
print(f'  NR cells: {len(nr_idx_in_adata)}, DR cells: {len(dr_idx_in_adata)}')
print(f'  archetype distribution: {pd.Series(arch_combined).value_counts().to_dict()}')

raw_comb = raw_counts[combined_idx]

# =============================================================================
# Step 2 — select rows: top TOP_N_ARCH enriched markers per archetype ranked by
#          effect size (log2FC); significant, non-technical, present, de-duplicated
# =============================================================================
markers = pd.read_csv(MARKERS_TSV, sep='\t')
markers = markers[(markers['log2FC'] > 0) & markers['fdr'].notna() & (markers['fdr'] < 0.05)]
markers = markers[~markers['gene'].str.match(TECH_GENE_RE)]   # drop ribosomal/Gm/mito/metallothionein
present = set(gene_names)
row_arch_of, selected, row_genes = {}, set(), []
for a in range(1, N_ARCH + 1):
    sub = markers[markers['archetype'] == f'archetype_{a}'].sort_values('log2FC', ascending=False)
    taken = []
    for g in sub['gene']:
        if g in present and g not in selected:
            taken.append(g)
            selected.add(g)
            if len(taken) == TOP_N_ARCH:
                break
    for g in taken:
        row_arch_of[g] = a
    row_genes.extend(taken)
    if len(taken) < TOP_N_ARCH:
        print(f'  WARNING: Arch{a} only {len(taken)} markers available (present & unique)')
    print(f'  Arch{a}: took {len(taken)} markers')
print(f'  Total marker rows: {len(row_genes)}')

gene_col_of = {g: np.where(gene_names == g)[0][0] for g in row_genes}
row_gene_idx = np.array([gene_col_of[g] for g in row_genes])

# =============================================================================
# Step 3 — build (sample x archetype) pseudobulk columns: log1p(CP10k), ordered
# =============================================================================
cond_of_sample = {s: ('DR' if c == 1 else 'NR') for s, c in zip(sample_combined, condition_code)}
age_of_sample  = dict(zip(sample_combined, age_combined))
uniq_samples   = np.unique(sample_combined)
ordered_samples = sorted(
    uniq_samples,
    key=lambda s: (0 if cond_of_sample[s] == 'NR' else 1, AGE_ORDER[age_of_sample[s]], s),
)

# column order: condition (NR then DR) -> archetype -> age/sample
samples_by_cond = {c: [s for s in ordered_samples if cond_of_sample[s] == c]
                   for c in ['NR', 'DR']}
col_ids, col_sample, col_arch, col_cond = [], [], [], []
expr_cols = []   # each: log1p(CP10k) over the selected genes
for cond in ['NR', 'DR']:
    for a in range(1, N_ARCH + 1):
        for s in samples_by_cond[cond]:
            cmask = (sample_combined == s) & (arch_code == a)
            if int(cmask.sum()) < MIN_PB_CELLS:
                continue
            pb = raw_comb[cmask][:, row_gene_idx].sum(axis=0).astype(np.float64)
            total = raw_comb[cmask].sum()                 # total counts over ALL genes
            cp10k = pb / total * 1e4
            expr_cols.append(np.log1p(cp10k))             # ln(1 + CP10k)
            col_ids.append(f'{s}|A{a}')
            col_sample.append(s)
            col_arch.append(a)
            col_cond.append(cond)

expr = np.vstack(expr_cols).T                             # genes x columns
print(f'  Pseudobulk columns: {expr.shape[1]} (of max {N_ARCH * len(ordered_samples)})')

# --- z-score per gene across all columns ---
mu = expr.mean(axis=1, keepdims=True)
sd = expr.std(axis=1, keepdims=True)
zero_var = (sd[:, 0] == 0)
if zero_var.any():
    print(f'  WARNING: {int(zero_var.sum())} genes have zero variance across columns -> set to 0')
sd[sd == 0] = 1.0
z = (expr - mu) / sd

# --- order rows within each archetype block by hierarchical clustering of z-profiles ---
def cluster_leaf_order(mat):
    """Ward-linkage leaf order (optimal ordering) of the rows of `mat`; identity if < 3."""
    if mat.shape[0] < 3:
        return np.arange(mat.shape[0])
    return leaves_list(linkage(mat, method='ward', optimal_ordering=True))


row_arch_arr = np.array([row_arch_of[g] for g in row_genes])
row_order = []
for a in range(1, N_ARCH + 1):
    idx = np.where(row_arch_arr == a)[0]
    if len(idx):
        row_order.extend(idx[cluster_leaf_order(z[idx])])
row_order = np.array(row_order)
z = z[row_order]
row_genes = [row_genes[i] for i in row_order]
print('  Rows ordered by ward hierarchical clustering within archetype blocks')

# --- collapse to condition × archetype means (mean z across samples): simpler panel ---
col_arch_arr, col_cond_arr = np.array(col_arch), np.array(col_cond)
avg_cols, avg_arch, avg_cond, avg_labels = [], [], [], []
for cond in ['NR', 'DR']:
    for a in range(1, N_ARCH + 1):
        sel = (col_cond_arr == cond) & (col_arch_arr == a)
        if not sel.any():
            continue
        avg_cols.append(z[:, sel].mean(axis=1))
        avg_arch.append(a)
        avg_cond.append(cond)
        avg_labels.append(f'{cond} A{a}')
zavg = np.vstack(avg_cols).T                              # genes x (condition × archetype)
print(f'  Averaged columns (condition × archetype): {zavg.shape[1]}')

# =============================================================================
# Step 4 — two-panel heatmap: (A) per sample × archetype, (B) mean over samples
#          per condition × archetype. Shared rows (archetype blocks) + z-scale.
# =============================================================================
plt.rcParams['pdf.fonttype'] = 42
arch_palette = sns.color_palette('Dark2', N_ARCH)
arch_rgb = {a: tuple(arch_palette[a - 1]) for a in range(1, N_ARCH + 1)}
cond_rgb = {c: mcolors.to_rgb(col) for c, col in {'NR': COLOR_NR, 'DR': COLOR_DR}.items()}

# column annotation strips (row 0 = condition, row 1 = archetype)
annA = np.stack([[cond_rgb[c] for c in col_cond], [arch_rgb[a] for a in col_arch]])
annB = np.stack([[cond_rgb[c] for c in avg_cond], [arch_rgb[a] for a in avg_arch]])
row_strip = np.array([[arch_rgb[row_arch_of[g]]] for g in row_genes])   # (nrows, 1, 3)

nrows, nA, nB = len(row_genes), z.shape[1], zavg.shape[1]
fig = plt.figure(figsize=(2.0 + 0.15 * nA + 0.35 * nB + 2.5, 0.17 * nrows + 1.8))
gs = fig.add_gridspec(2, 6, width_ratios=[0.5, nA, 1.6, nB * 2.4, 0.8, 0.35],
                      height_ratios=[1.5, nrows], hspace=0.02, wspace=0.06)

axA_ann = fig.add_subplot(gs[0, 1])
axB_ann = fig.add_subplot(gs[0, 3])
ax_row  = fig.add_subplot(gs[1, 0])
axA     = fig.add_subplot(gs[1, 1])
axB     = fig.add_subplot(gs[1, 3])
ax_cbar = fig.add_subplot(gs[1, 5])

# column annotations
for axn, ann in [(axA_ann, annA), (axB_ann, annB)]:
    axn.imshow(ann, aspect='auto')
    axn.set_xticks([])
    axn.set_yticks([])
axA_ann.set_yticks([0, 1])
axA_ann.set_yticklabels(['condition', 'archetype'], fontsize=7)
axA_ann.tick_params(length=0)

# row (gene) color strip + gene labels
ax_row.imshow(row_strip, aspect='auto')
ax_row.set_xticks([])
ax_row.set_yticks(range(nrows))
ax_row.set_yticklabels(row_genes, fontsize=5)
ax_row.tick_params(length=0)
ax_row.set_ylabel('archetype markers (grouped by archetype)', fontsize=8)

# heatmaps
im = axA.imshow(z, aspect='auto', cmap='RdBu_r', vmin=VMIN, vmax=VMAX)
axA.set_yticks([])
axA.set_xticks(range(nA))
axA.set_xticklabels(col_sample, fontsize=5, rotation=90)
axA.set_xlabel('sample (NR then DR, by archetype)', fontsize=8)
axA.set_title('per sample × archetype', fontsize=9)

# thin vertical lines at each (condition, archetype) group boundary
for i in range(1, nA):
    if col_arch[i] != col_arch[i - 1] or col_cond[i] != col_cond[i - 1]:
        axA.axvline(i - 0.5, color='black', lw=0.5)

axB.imshow(zavg, aspect='auto', cmap='RdBu_r', vmin=VMIN, vmax=VMAX)
axB.set_yticks([])
axB.set_xticks(range(nB))
axB.set_xticklabels(avg_labels, fontsize=7, rotation=90)
axB.set_xlabel('mean over samples', fontsize=8)
axB.set_title('condition × archetype mean', fontsize=9)

fig.colorbar(im, cax=ax_cbar, label='z-score  log1p(CP10k)')

# legends: archetype (rows & cols) + condition (cols)
arch_handles = [mpatches.Patch(color=arch_rgb[a], label=f'Arch{a}') for a in range(1, N_ARCH + 1)]
cond_handles = [mpatches.Patch(color=cond_rgb[c], label=c) for c in ['NR', 'DR']]
fig.legend(handles=arch_handles, title='archetype', frameon=False,
           loc='upper left', bbox_to_anchor=(0.99, 0.92), fontsize=7, title_fontsize=8)
fig.legend(handles=cond_handles, title='condition (col)', frameon=False,
           loc='upper left', bbox_to_anchor=(0.99, 0.66), fontsize=7, title_fontsize=8)

fig.suptitle(f'Top {TOP_N_ARCH} archetype markers/archetype by log2FC '
             f'(cheng22 1-vs-rest, tech-filtered) × sample-archetype pseudobulk, z-scored per gene',
             y=0.99, fontsize=11)
fig.savefig(OUT_PDF, bbox_inches='tight')
plt.close(fig)
print(f'Saved {OUT_PDF}')
print('Done.')
