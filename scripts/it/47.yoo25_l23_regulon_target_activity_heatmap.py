"""L2/3 regulon-target activity heatmap across postnatal stages (Yoo25 AllAges mouse IT).

Heatmap counterpart of script 46: each ROW is a regulon, each COLUMN is a postnatal-stage
pseudobulk (all 11 ages in one grid, ordered by (day, condition) with each dark-reared age
placed immediately AFTER its matched normal age, e.g. P12, P12DR, P14, P14DR, ...). Cell
color = that regulon's activity at that age, normalized PER ROW by its own MAX (each
regulon divided by its peak across ages, so max->1 and the min stays at its true
fraction of the peak) so each regulon's temporal SHAPE is comparable while keeping the
baseline meaningful; the raw activity is retained in the TSV.

Activity pipeline is identical to script 46 (pseudobulk-by-Age, NOT per-cell):
  1. Subset to L2/3 (Subclass == 'L2/3').
  2. Counts source: adata.raw.X (raw integer counts) indexed by adata.raw.var_names.
  3. Parse Age 'P<day>[DR]' -> (day:int, condition:'DR'|'normal'). Keep ALL ages.
  4. Pseudobulk by Age: sum raw counts over all L2/3 cells sharing an Age -> gene x age.
  5. Normalize per age: CPM = counts/age_total*1e6; expr = log1p(CPM).
  6. Min-max scale each gene to [0,1] jointly across ALL age columns (normal + DR share
     one scale; zero-range genes contribute 0).
  7. Regulon activity per age = mean of the [0,1]-scaled values over that regulon's target
     genes present in the dataset (intersection with raw.var_names; n used / total printed).

Reads:
  links/it/superdupermegaRNA_yoo25_IT_AllAges.h5ad
  local_data/res/it/40.yoo25_L2_3_regulon_targets.tsv
Outputs:
  local_data/fig/it/47.yoo25_l23_regulon_target_activity_heatmap.pdf
  local_data/res/it/47.yoo25_l23_regulon_target_activity_heatmap.tsv
"""

import os
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42     # editable vector text in PDF
plt.rcParams['svg.fonttype'] = 'none'  # editable vector text in SVG

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IN_H5AD     = os.path.join(PROJECT_ROOT, 'links', 'it', 'superdupermegaRNA_yoo25_IT_AllAges.h5ad')
IN_REG      = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it', '40.yoo25_L2_3_regulon_targets.tsv')
OUT_FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it')
OUT_RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')

# Regulon rows, top-to-bottom in this order.
SELECTED_REGULONS = ['Fos', 'Fosb', 'Fosl2', 'Egr1', 'Egr2', 'Egr3', 'Egr4',
                     'Rfx3', 'Meis2', 'Nfib', 'Tcf12', 'Jdp2', 'Satb1']
DIRECTION    = '+/+'           # activating regulons only (matches scripts 42/46/49)
SUBCLASS_COL = 'Subclass'
KEEP_SUBCLASS = 'L2/3'
AGE_COL      = 'Age'
CP_TARGET    = 1e6             # CPM
CMAP         = 'RdBu_r'        # per-row [0,1] color: blue (low) -> red (high)

OUT_PDF = os.path.join(OUT_FIG_DIR, '47.yoo25_l23_regulon_target_activity_heatmap.pdf')
OUT_TSV = os.path.join(OUT_RES_DIR, '47.yoo25_l23_regulon_target_activity_heatmap.tsv')

_AGE_RE = re.compile(r'^P(\d+)(DR)?$')


def parse_age(age):
    """'P<day>[DR]' -> (day:int, condition:'DR'|'normal'). Fail-fast on unparseable."""
    m = _AGE_RE.match(str(age))
    if m is None:
        raise ValueError(f"Cannot parse Age value {age!r}; expected 'P<day>' optionally '+DR'")
    return int(m.group(1)), ('DR' if m.group(2) else 'normal')


def main():
    os.makedirs(OUT_FIG_DIR, exist_ok=True)
    os.makedirs(OUT_RES_DIR, exist_ok=True)

    # --- regulon target sets (+/+ only): TF -> set of target symbols ---
    reg = pd.read_csv(IN_REG, sep='\t')
    reg = reg[reg['regulation_direction'] == DIRECTION]
    tf_targets = {}
    for tf in SELECTED_REGULONS:
        key = f'{tf}_{DIRECTION}'
        targets = set(reg.loc[reg['regulon'] == key, 'Gene'])
        if not targets:
            raise ValueError(f"No {DIRECTION} regulon '{key}' found in {IN_REG}")
        tf_targets[tf] = targets

    print(f'Loading {IN_H5AD}')
    adata = ad.read_h5ad(IN_H5AD)

    # --- subset to L2/3 ---
    adata = adata[(adata.obs[SUBCLASS_COL] == KEEP_SUBCLASS).values].copy()
    if adata.n_obs == 0:
        raise ValueError(f"No {KEEP_SUBCLASS} cells found in column {SUBCLASS_COL!r}")
    print(f'{adata.n_obs} {KEEP_SUBCLASS} cells (expect 38,539)')

    if adata.raw is None:
        raise ValueError("adata.raw is None; raw integer counts required for pseudobulk")
    raw_var = list(adata.raw.var_names)
    name2idx = {g: i for i, g in enumerate(raw_var)}
    print(f'Using adata.raw ({adata.raw.shape[1]} genes) as raw-count source')

    # --- parse Age -> (day, condition) per cell; keep all ages present ---
    ages = adata.obs[AGE_COL].astype(str).values
    age_info = {a: parse_age(a) for a in sorted(set(ages))}
    # order ages by (day, condition) so each DR age sits right after its matched normal age
    cond_rank = {'normal': 0, 'DR': 1}
    ordered_ages = sorted(age_info, key=lambda a: (age_info[a][0], cond_rank[age_info[a][1]]))
    print(f'{len(ordered_ages)} ages: {ordered_ages}')

    # --- pseudobulk by Age: sum raw counts over cells sharing an Age ---
    Xraw = adata.raw.X
    n_genes = adata.raw.shape[1]
    pb = np.zeros((n_genes, len(ordered_ages)), dtype=np.float64)  # gene x age
    n_cells = {}
    for j, a in enumerate(ordered_ages):
        mask = ages == a
        n_cells[a] = int(mask.sum())
        block = Xraw[mask]
        block = block.toarray() if sp.issparse(block) else np.asarray(block)
        col_sum = np.asarray(block, dtype=np.float64).sum(axis=0).ravel()
        pb[:, j] = col_sum
        print(f'  {a:8s}: {n_cells[a]:6d} cells pooled')

    # --- normalize per age: CPM -> log1p ---
    age_totals = pb.sum(axis=0)  # (n_ages,)
    if np.any(age_totals == 0):
        zero = [ordered_ages[j] for j in np.where(age_totals == 0)[0]]
        raise ValueError(f"Zero-count pseudobulk for age(s): {zero}")
    cpm = pb / age_totals[None, :] * CP_TARGET
    logexpr = np.log1p(cpm)  # gene x age, log(1 + CPM)

    # --- min-max scale each gene to [0,1] jointly across ALL age columns ---
    gmin = logexpr.min(axis=1, keepdims=True)
    gmax = logexpr.max(axis=1, keepdims=True)
    grng = gmax - gmin
    scaled = np.where(grng > 0, (logexpr - gmin) / np.where(grng > 0, grng, 1.0), 0.0)

    # --- regulon activity per age = mean of [0,1] scaled target expression ---
    # activity matrix: regulon (row) x age (col), same age order as ordered_ages
    act_mat = np.zeros((len(SELECTED_REGULONS), len(ordered_ages)), dtype=np.float64)
    rows = []
    for i, tf in enumerate(SELECTED_REGULONS):
        targets = tf_targets[tf]
        used = [g for g in sorted(targets) if g in name2idx]
        if not used:
            raise ValueError(f"{tf}: 0/{len(targets)} targets present in raw.var_names")
        print(f'  {tf:8s}: {len(used)}/{len(targets)} targets used')
        cols = [name2idx[g] for g in used]
        activity = scaled[cols, :].mean(axis=0)  # (n_ages,)
        act_mat[i, :] = activity

    # --- per-row max normalize (divide each regulon by its own max; max->1, min stays >0) ---
    rmax = act_mat.max(axis=1, keepdims=True)
    act_norm = np.where(rmax > 0, act_mat / np.where(rmax > 0, rmax, 1.0), 0.0)

    for i, tf in enumerate(SELECTED_REGULONS):
        used = [g for g in sorted(tf_targets[tf]) if g in name2idx]
        for j, a in enumerate(ordered_ages):
            day, cond = age_info[a]
            rows.append(dict(regulon=tf, n_targets_total=len(tf_targets[tf]), n_targets_used=len(used),
                             age=a, day=day, condition=cond, n_cells=n_cells[a],
                             activity=float(act_mat[i, j]),
                             activity_rowmaxnorm=float(act_norm[i, j])))

    # --- heatmap: regulons (rows) x ages (cols), per-row max-normalized color ---
    ncols = len(ordered_ages)
    nrows = len(SELECTED_REGULONS)
    fig, ax = plt.subplots(figsize=(0.62 * ncols + 2.2, 0.42 * nrows + 1.6))
    im = ax.imshow(act_norm, aspect='auto', cmap=CMAP, vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(ncols))
    ax.set_xticklabels(ordered_ages, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(nrows))
    ax.set_yticklabels(SELECTED_REGULONS, fontsize=9)
    ax.set_xlabel('postnatal stage (age)')
    ax.set_ylabel('regulon')

    # thin gridlines between cells
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.5)
    ax.tick_params(which='minor', length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label('regulon activity\n(per-row max-normalized)', fontsize=8)

    ax.set_title(f'Regulon-target activity over postnatal stage — {KEEP_SUBCLASS} IT '
                 f'(Yoo25 AllAges)', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches='tight')
    plt.close(fig)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_TSV, sep='\t', index=False)
    print(f'Wrote {OUT_PDF}')
    print(f'Wrote {OUT_TSV} ({len(out)} rows)')


if __name__ == '__main__':
    main()
