"""Rfx4 pseudobulk expression barplot: Astrocytes vs Neurons (cheng22 P28 NR, matplotlib PDF).

The saved `.X` in this h5ad is ALREADY normalized (log2(CP10k+1)-like: non-integer,
max ~6.5), so it must NOT be used for pseudobulk. Raw integer counts live in `adata.raw`.
We therefore:
  1. pull raw counts from adata.raw,
  2. pseudobulk PER (subclass, replicate) = sum raw counts over cells of that subclass
     within one replicate; replicate = the `sample` column (P28_1a/1b/2a/2b, n=4),
  3. normalize each pseudobulk to CPM, then log2(CPM+1),
  4. barplot Rfx4 for a subset of subclasses (Astro + 3 neuron types), ordered by
     decreasing mean expression, with one dot per replicate on each bar.

Astrocytes are expected to have the highest Rfx4 expression.

Reads:
  links/astro/cheng22_AllSubclasses_P28NR.h5ad
Outputs:
  local_data/fig/astro/46.Rfx4_expr_barplot_by_subclass.pdf
"""

import os
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

# --- file paths ---
SCRIPTS_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
INPUT_H5AD   = os.path.join(PROJECT_ROOT, 'links', 'astro', 'cheng22_AllSubclasses_P28NR.h5ad')
FIG_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
OUT_PDF      = os.path.join(FIG_DIR, '46.Rfx4_expr_barplot_by_subclass.pdf')

# --- config ---
GENE         = 'Rfx4'
SUBCLASS_COL = 'Subclass'
REPLICATE_COL = 'sample'                       # replicates: P28_1a, P28_1b, P28_2a, P28_2b
KEEP_SC      = ['Astro', 'L2/3', 'Pvalb']  # astrocytes + 2 neuron subclasses
CPM_SCALE    = 1e6
HIGHLIGHT_SC = 'Astro'
COLOR_HIGH   = '#d62728'   # highlight astrocytes
COLOR_OTHER  = '#8c9bab'   # neutral gray-blue for the rest

os.makedirs(FIG_DIR, exist_ok=True)

# =============================================================================
# Step 1 — load and grab raw counts (pseudobulk must use raw, not normalized .X)
# =============================================================================
print(f'Loading {INPUT_H5AD}')
adata = ad.read_h5ad(INPUT_H5AD)
print(f'  {adata.n_obs} cells x {adata.n_vars} genes')

assert adata.raw is not None, 'adata.raw (raw counts) is required for pseudobulk'
raw = adata.raw[:, adata.var_names].X
raw = raw.toarray() if sp.issparse(raw) else np.asarray(raw)
raw = np.rint(raw).astype(np.int64)   # ensure integer raw counts

var_names = np.array(adata.var_names)
gene_hits = np.where(var_names == GENE)[0]
if len(gene_hits) == 0:
    raise ValueError(f'Gene {GENE!r} not found in adata.var_names')
gene_idx = gene_hits[0]

for col in (SUBCLASS_COL, REPLICATE_COL):
    if col not in adata.obs.columns:
        raise ValueError(f'Column {col!r} not found in adata.obs')
subclasses = adata.obs[SUBCLASS_COL].astype(str).values
replicates = adata.obs[REPLICATE_COL].astype(str).values
rep_levels = sorted(np.unique(replicates))
print(f'  replicates ({REPLICATE_COL}): {rep_levels}')

missing = [sc for sc in KEEP_SC if sc not in set(subclasses)]
if missing:
    raise ValueError(f'Requested subclasses not present: {missing}')

# =============================================================================
# Step 2 — pseudobulk per (subclass, replicate): sum raw counts -> CPM -> log2(CPM+1)
# =============================================================================
# per_rep[sc] = list of (replicate, log2(CPM+1)) across replicates
per_rep = {sc: [] for sc in KEEP_SC}
for sc in KEEP_SC:
    for rep in rep_levels:
        mask = (subclasses == sc) & (replicates == rep)
        n_cells = int(mask.sum())
        if n_cells == 0:
            print(f'  WARNING: no cells for {sc} / {rep}; skipping this replicate')
            continue
        pb = raw[mask].sum(axis=0)                 # per-gene summed raw counts
        lib_size = pb.sum()
        cpm_gene = pb[gene_idx] / lib_size * CPM_SCALE
        logcpm = float(np.log2(cpm_gene + 1.0))
        per_rep[sc].append((rep, n_cells, logcpm))

# bar height = mean of per-replicate log2(CPM+1); sort subclasses by decreasing mean
rows = []
for sc in KEEP_SC:
    vals = [v for _, _, v in per_rep[sc]]
    rows.append((sc, float(np.mean(vals)), vals))
rows.sort(key=lambda r: r[1], reverse=True)
sc_order   = [r[0] for r in rows]
mean_val   = [r[1] for r in rows]

print(f'\n{GENE} pseudobulk log2(CPM+1) by subclass x replicate:')
for sc, m, _ in rows:
    print(f'  {sc:<8} mean={m:6.3f}')
    for rep, n, v in per_rep[sc]:
        print(f'      {rep:<8} n_cells={n:>5}  log2(CPM+1)={v:6.3f}')

top_sc = sc_order[0]
if top_sc == HIGHLIGHT_SC:
    print(f'\nSanity check OK: {HIGHLIGHT_SC} has the highest {GENE} expression.')
else:
    print(f'\nNOTE: top subclass is {top_sc!r}, not {HIGHLIGHT_SC!r}.')

# =============================================================================
# Step 3 — barplot with per-replicate dots (vector PDF)
# =============================================================================
plt.rcParams['pdf.fonttype'] = 42   # editable vector text

fig, ax = plt.subplots(figsize=(6, 5))
x = np.arange(len(sc_order))
colors = [COLOR_HIGH if sc == HIGHLIGHT_SC else COLOR_OTHER for sc in sc_order]
ax.bar(x, mean_val, color=colors, edgecolor='black', linewidth=0.6, zorder=1)

# one dot per replicate on each bar (seaborn stripplot, long-form data)
dots_df = pd.DataFrame(
    [(sc, rep, v) for sc in sc_order for rep, _, v in per_rep[sc]],
    columns=['subclass', 'replicate', 'logcpm'],
)
sns.swarmplot(data=dots_df, x='subclass', y='logcpm', order=sc_order,
              color='black', size=6, linewidth=0.5,
              edgecolor='white', ax=ax, zorder=3)

ax.set_ylabel(f'{GENE}  log2(CPM + 1)')
ax.set_xlabel('Subclass')
ax.set_title(f'{GENE} pseudobulk expression: astrocytes vs neurons\n(cheng22 P28 NR, dots = replicates)')
ax.set_xticks(x)
ax.set_xticklabels(sc_order, rotation=45, ha='right')
sns.despine(ax=ax)

fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved {OUT_PDF}')
print('Done.')
