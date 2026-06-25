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
