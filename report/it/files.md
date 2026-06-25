# Archetype-associated gene files — IT scripts 33–36

The archetype-associated genes (markers) live in `local_data/res/it/`, one TSV per
subclass, produced by the `XX.follow.two_*_archetype_scores.py` scripts.

| Script | Subclass | Markers file |
|---|---|---|
| `scripts/it/33.follow.two_L4_archetype_scores.py`   | L4 IT  | `local_data/res/it/33.follow.two_L4_archetype_markers.tsv` |
| `scripts/it/34.follow.two_L23_archetype_scores.py`  | L2/3 IT | `local_data/res/it/34.follow.two_L23_archetype_markers.tsv` |
| `scripts/it/35.follow.two_L5IT_archetype_scores.py` | L5 IT  | `local_data/res/it/35.follow.two_L5IT_archetype_markers.tsv` |
| `scripts/it/36.follow.two_L6IT_archetype_scores.py` | L6 IT  | `local_data/res/it/36.follow.two_L6IT_archetype_markers.tsv` |

## How they're produced

Each `XX.follow.*_archetype_scores.py` script:

1. Loads the two-dataset (cheng22 P28NR + yoo25 P21) Harmony embedding coords.
2. Fits PCHA on the top-5 Harmony PCs at the subclass's optimal `NOC`.
3. Takes the top 300 cells nearest each archetype.
4. Runs a conservative one-vs-each Wilcoxon test over the shared expressed /
   nonzero-variance gene set (mito removed).
5. Filters markers on `frac_in ≥ 0.25`, `fdr < 0.001`, `log2FC > log2(1.5)`.

## Markers file schema

Columns: `gene, archetype, log2FC, pval, fdr, frac_in, frac_out`

Archetype-associated genes are the rows, grouped by the `archetype` column
(`archetype_1`, `archetype_2`, …).

## Related outputs (same folder, per script)

- `XX.follow.two_*_archetype_scores.tsv` — per-cell archetype scores derived from these markers
- `XX.follow.two_*_pcha_xp.tsv` — per-cell PCHA coordinates
- `XX.follow.two_*_pcha_aa.tsv` — archetype coordinates in PCHA space

# Regulon → target-gene tables — IT script 40

Clean regulon membership tables derived from the Yoo25 SCENIC+ eRegulon metadata,
one TSV per IT layer, produced by `scripts/it/40.yoo25_IT_regulon_targets.py`.

| Layer | Output file | Regulons | +/+ | -/+ | direct | extended-only | Genes | Target pairs |
|---|---|---|---|---|---|---|---|---|
| L2/3 IT | `local_data/res/it/40.yoo25_L2_3_regulon_targets.tsv` | 139 | 96 | 43 | 68 | 71 | 3,795 | 11,569 |
| L4 IT   | `local_data/res/it/40.yoo25_L4_regulon_targets.tsv`   | 107 | 72 | 35 | 66 | 41 | 3,537 | 9,433 |
| L5 IT   | `local_data/res/it/40.yoo25_L5IT_regulon_targets.tsv` | 54  | 33 | 21 | 41 | 13 | 2,708 | 4,398 |
| L6 IT   | `local_data/res/it/40.yoo25_L6IT_regulon_targets.tsv` | 49  | 36 | 13 | 30 | 19 | 2,558 | 4,679 |

`Genes` = number of distinct target genes in the layer; `Target pairs` = rows
(one per unique regulon–gene pair). `+/+` and `-/+` are the per-layer regulon
counts by sign pattern (activating vs repressing TF→gene); `direct` and
`extended-only` split the same regulons by annotation source. Only regulons with
a positive R2G (second) sign are kept (`+/+`, `-/+`); `+/-` and `-/-` regulons are
dropped.

## How they're produced

Each layer's SCENIC+ output is split into `direct` and `extended` eRegulon
metadata files. The script:

1. Normalizes each regulon to **TF + sign pattern** (e.g. `Arnt_direct_+/+` →
   `Arnt_+/+`), so `direct` and `extended` versions share a key. `Arnt_+/+` and
   `Arnt_-/+` are distinct regulons.
2. If a regulon exists in both `direct` and `extended`, keeps its **direct**
   targets; otherwise keeps the **extended** targets.
3. Collapses the multiple-enhancer-region rows to unique `(regulon, Gene)` pairs.
4. Keeps only regulons whose R2G (second) sign is positive (`+/+`, `-/+`),
   dropping `+/-` and `-/-`.

## File schema

Columns: `regulon, TF, regulation_direction, source, Gene`

One row per regulon–target-gene pair (long format), grouped by the `regulon`
column. `source` is `direct` or `extended`; `regulation_direction` is the sign
pattern (`+/+` or `-/+` — only positive-R2G regulons are retained).
