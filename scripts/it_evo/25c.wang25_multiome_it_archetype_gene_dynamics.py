"""Per-gene breakdown of the archetype gene sets scored in scripts/it_evo/25.

Script 25 plots one line per archetype — the average over its gene set. This unpacks that
average: for every (panel, archetype, gene) it reports the per-stage means and ranks the
genes by how dynamic they are across stages, so it is clear which genes drive an archetype's
temporal shape and which only set its baseline level.

Same cells, same gene sets and same normalization as script 25: EN-L{2/3,4,5,6}-IT cells in
V1, mouse archetype markers mapped to human orthologs and kept only if they are also human
HVGs for that layer, per-gene log2(CP10k+1) max-scaled by the NORM_MAX_PCTILE percentile
across cells. Script 25's archetype score is exactly the mean of the `norm_<stage>` columns
here over an archetype's genes.

Per-gene metrics (computed on the max-scaled per-stage means, i.e. what enters the average):
  contrib         mean score across all cells — how much a gene lifts the archetype's level
  dyn_range       1 - trough/peak; 0 = flat across stages, ->1 = off in at least one stage
  fold            peak / trough (inf if the trough is exactly 0)
  range_over_sem  (peak - trough) / the largest per-stage SEM — dynamic range against noise,
                  the metric to trust when a stage has few cells
Genes are ranked within each archetype by range_over_sem (rank 1 = most dynamic).

Reads:
  links/it_evo/wang25_human_multiome_gex_it.h5ad
  local_data/res/it/{34,33,35,36}.follow.two_{L23,L4,L5IT,L6IT}_archetype_markers.tsv
  local_data/res/it_evo/02.human_{L23,L4,L5IT,L6IT}_varimax_loadings.tsv   (human HVG list)
  data/human_mouse_orthologs.tsv
Outputs:
  local_data/res/it_evo/25c.wang25_multiome_it_archetype_gene_dynamics.tsv
"""

import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IN_H5AD       = os.path.join(PROJECT_ROOT, 'links', 'it_evo', 'wang25_human_multiome_gex_it.h5ad')
IN_MARKER_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
IN_HVG_TMPL   = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo',
                             '02.human_{token}_varimax_loadings.tsv')
IN_ORTHOLOGS  = os.path.join(PROJECT_ROOT, 'data', 'human_mouse_orthologs.tsv')
OUT_RES_DIR   = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')

OUT_TSV = os.path.join(OUT_RES_DIR, '25c.wang25_multiome_it_archetype_gene_dynamics.tsv')

# Panels: (label, human Type, layer token for the human HVG list, mouse marker TSV basename,
# archetype letters) — kept in sync with script 25.
PANELS = [
    ('L2/3',  'EN-L2_3-IT', 'L23',  '34.follow.two_L23_archetype_markers.tsv',  ['A', 'B', 'C']),
    ('L4',    'EN-L4-IT',   'L4',   '33.follow.two_L4_archetype_markers.tsv',   ['A', 'B', 'C']),
    ('L5 IT', 'EN-L5-IT',   'L5IT', '35.follow.two_L5IT_archetype_markers.tsv', ['A', 'B']),
    ('L6 IT', 'EN-L6-IT',   'L6IT', '36.follow.two_L6IT_archetype_markers.tsv', ['A', 'B', 'C']),
]

KEEP_REGION = 'V1'             # matches script 25 (V1 only)
TYPE_COL    = 'Type'
REGION_COL  = 'Region'
GROUP_COL   = 'Group'
CP_TARGET   = 1e4              # CP10k
NORM_MAX_PCTILE = 99           # per-gene max-scaling percentile (maps to 1), as script 25

# Developmental stages in temporal order; every panel must have all of them.
STAGE_ORDER = ['Second_trimester', 'Third_trimester', 'Infancy', 'Adolescence']

N_SHOW = 3                     # top/bottom genes printed per archetype


def load_ortholog_map():
    """1-to-1 mouse->human symbol map (same construction as script 25)."""
    ortho = pd.read_csv(IN_ORTHOLOGS, sep='\t')
    ortho = ortho.drop_duplicates('human_symbol').drop_duplicates('mouse_symbol')
    return dict(zip(ortho['mouse_symbol'], ortho['human_symbol']))


def gene_stage_metrics(gene, X, depth, stage, name2idx):
    """Per-stage means and dynamic-range metrics for one gene (script-25 normalization)."""
    l2 = np.log2(X[:, name2idx[gene]] / depth * CP_TARGET + 1.0)   # (n_cells,)
    hi = np.percentile(l2, NORM_MAX_PCTILE)
    norm = np.clip(l2 / (hi if hi > 0 else 1.0), 0.0, 1.0)

    df = pd.DataFrame({'stage': stage, 'raw': l2, 'norm': norm, 'det': l2 > 0})
    grp = df.groupby('stage', observed=True)
    n_by = grp.size().reindex(STAGE_ORDER)
    nrm = grp['norm'].mean().reindex(STAGE_ORDER)
    sem = (grp['norm'].std(ddof=1) / np.sqrt(n_by)).reindex(STAGE_ORDER)
    raw = grp['raw'].mean().reindex(STAGE_ORDER)
    det = grp['det'].mean().reindex(STAGE_ORDER)

    peak, trough = nrm.max(), nrm.min()
    if peak <= 0:
        raise ValueError(f"Gene {gene} has a non-positive peak per-stage mean; cannot rank")
    rec = {
        'p99_log2cp10k': hi,
        'pct_detected': 100.0 * det.mean(),
        'contrib': norm.mean(),
        'dyn_range': (peak - trough) / peak,
        'fold': peak / trough if trough > 0 else np.inf,
        'range_over_sem': (peak - trough) / sem.max(),
        'peak_stage': nrm.idxmax(),
        'trough_stage': nrm.idxmin(),
    }
    for s in STAGE_ORDER:
        rec[f'norm_{s}'] = nrm[s]
        rec[f'sem_{s}'] = sem[s]
        rec[f'raw_{s}'] = raw[s]
        rec[f'pct_det_{s}'] = 100.0 * det[s]
        rec[f'n_cells_{s}'] = n_by[s]
    return rec


def main():
    os.makedirs(OUT_RES_DIR, exist_ok=True)

    mouse_to_human = load_ortholog_map()
    print(f'{len(mouse_to_human)} 1-to-1 mouse->human orthologs')

    print(f'Loading {IN_H5AD}')
    adata = ad.read_h5ad(IN_H5AD)
    human_genes = set(adata.var_names)

    rows = []
    for label, human_type, token, marker_file, letters in PANELS:
        # --- cells: this IT type, V1 only ---
        sub = adata[((adata.obs[TYPE_COL] == human_type)
                     & (adata.obs[REGION_COL] == KEEP_REGION)).values].copy()
        if sub.n_obs == 0:
            raise ValueError(f"No {human_type} cells in region {KEEP_REGION}")

        stage = sub.obs[GROUP_COL].astype(str).values
        missing = set(STAGE_ORDER) - set(stage)
        unknown = set(stage) - set(STAGE_ORDER)
        if missing or unknown:
            raise ValueError(f"[{label}] stage mismatch — missing {missing}, unknown {unknown}")
        print(f'[{label}] {sub.n_obs} {human_type} cells ({KEEP_REGION}); per-stage n: '
              + ', '.join(f'{s}={int((stage == s).sum())}' for s in STAGE_ORDER))

        X = sub.X
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        X = X.astype(np.float64)
        depth = X.sum(axis=1)
        if np.any(depth == 0):
            raise ValueError(f"Found {human_type} cells with zero total counts; cannot normalize")
        name2idx = {g: i for i, g in enumerate(sub.var_names)}

        # --- mouse markers -> human orthologs present in the data and in this layer's HVGs ---
        markers = pd.read_csv(os.path.join(IN_MARKER_DIR, marker_file), sep='\t')
        hvg = set(pd.read_csv(IN_HVG_TMPL.format(token=token), sep='\t', index_col=0).index)
        keep = human_genes & hvg

        for i, letter in enumerate(letters):
            arch_id = f'archetype_{i + 1}'
            mouse_genes = markers.loc[markers['archetype'] == arch_id, 'gene'].tolist()
            if not mouse_genes:
                raise ValueError(f"No markers for {arch_id} in {marker_file}")
            pairs = [(g, mouse_to_human[g]) for g in mouse_genes
                     if g in mouse_to_human and mouse_to_human[g] in keep]
            if not pairs:
                raise ValueError(f"No ortholog-mapped human-HVG genes found in the data for "
                                 f"{label} {letter} ({arch_id}, {len(mouse_genes)} mouse markers)")

            arch_rows = []
            for mouse_gene, human_gene in pairs:
                rec = {'panel': label, 'human_type': human_type, 'arch_id': arch_id,
                       'archetype': letter, 'n_genes_used': len(pairs),
                       'mouse_gene': mouse_gene, 'human_gene': human_gene}
                rec.update(gene_stage_metrics(human_gene, X, depth, stage, name2idx))
                arch_rows.append(rec)

            arch = pd.DataFrame(arch_rows).sort_values('range_over_sem', ascending=False)
            arch.insert(7, 'rank', np.arange(1, len(arch) + 1))
            rows.append(arch)

            top = ', '.join(f'{r.human_gene} ({r.range_over_sem:.1f}, peak {r.peak_stage})'
                            for r in arch.head(N_SHOW).itertuples())
            bot = ', '.join(f'{r.human_gene} ({r.range_over_sem:.1f})'
                            for r in arch.tail(N_SHOW).itertuples())
            print(f'  {letter} ({arch_id}, {len(pairs)} genes)')
            print(f'    most dynamic:  {top}')
            print(f'    least dynamic: {bot}')

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT_TSV, sep='\t', index=False)
    print(f'Wrote {OUT_TSV} ({len(out)} rows)')


if __name__ == '__main__':
    main()
