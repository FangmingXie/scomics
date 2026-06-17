# cell_feature_matrix.h5

10x Xenium-style sparse matrix in HDF5 format (version 2).

## Top-level attributes

| Key | Value |
|-----|-------|
| `filetype` | `matrix` |
| `version` | `2` |
| `library_ids` | `0000000_0000000` |
| `original_gem_groups` | `[1]` |

## HDF5 structure

```
/                              (root)
└── matrix/                    (group)
    ├── barcodes               shape=(787007,)      dtype=|S10
    ├── data                   shape=(307932198,)   dtype=int32   # CSC values
    ├── indices                shape=(307932198,)   dtype=int64   # CSC row indices
    ├── indptr                 shape=(787008,)      dtype=int64   # CSC col pointers
    ├── shape                  shape=(2,)           dtype=int32   # [n_features, n_barcodes]
    └── features/              (group)
        ├── id                 shape=(13780,)       dtype=|S31    # e.g. ENSMUSG00000052595
        ├── name               shape=(13780,)       dtype=|S31    # e.g. A1cf
        ├── feature_type       shape=(13780,)       dtype=|S25
        ├── genome             shape=(13780,)       dtype=|S7     # all "Unknown"
        └── _all_tag_keys      shape=(1,)           dtype=|S6     # ["genome"]
```

## Matrix dimensions

| Item | Count |
|------|-------|
| Features (rows) | 13,780 |
| Barcodes / cells (cols) | 787,007 |
| Non-zero entries | 307,932,198 |

## Feature types

| Feature type | Count |
|--------------|-------|
| Gene Expression | 5,006 |
| Unassigned Codeword | 8,096 |
| Negative Control Codeword | 609 |
| Negative Control Probe | 40 |
| Genomic Control | 21 |
| Deprecated Codeword | 8 |

## Notes

- Genome field is `Unknown` for all features (typical for spatial/Xenium panels).
- Barcodes are alphanumeric strings with a gem-group suffix (e.g. `aaaaabgn-1`).
- Feature IDs are Ensembl mouse gene IDs (`ENSMUSG…`); names are gene symbols.
- The matrix is stored in **CSC (Compressed Sparse Column)** format: `data`, `indices`, `indptr`.

---

# cells.parquet

Per-cell metadata table: **787,007 rows × 14 columns**. One row per cell; `cell_id` matches the barcodes in `cell_feature_matrix.h5`.

## Columns

| Column | dtype | Description |
|--------|-------|-------------|
| `cell_id` | str | Unique cell identifier (matches HDF5 barcodes) |
| `x_centroid` | float64 | X spatial coordinate (µm); range 3.98–10,062 |
| `y_centroid` | float64 | Y spatial coordinate (µm); range 62.34–22,325 |
| `transcript_counts` | int64 | Gene expression transcript count; mean 560, max 7,979 |
| `control_probe_counts` | int64 | Negative control probe counts |
| `genomic_control_counts` | int64 | Genomic control counts |
| `control_codeword_counts` | int64 | Negative control codeword counts |
| `unassigned_codeword_counts` | int64 | Unassigned codeword counts |
| `deprecated_codeword_counts` | int64 | Deprecated codeword counts |
| `total_counts` | int64 | Total molecular counts (all feature types) |
| `cell_area` | float64 | Cell area (µm²); mean 49, range 1.4–1,285 |
| `nucleus_area` | float64 | Nucleus area (µm²); mean 30.5; 368 cells are NaN |
| `nucleus_count` | int64 | Number of nuclei per cell; mostly 1 |
| `segmentation_method` | str | Segmentation approach (3 categories; see below) |

## segmentation_method

| Value | Count |
|-------|-------|
| Segmented by interior stain (18S) | 707,353 |
| Segmented by nucleus expansion of 5.0 µm | 51,914 |
| Segmented by boundary stain (ATP1A1+CD45+E-Cadherin) | 27,740 |

## nucleus_count

| Nuclei | Cells |
|--------|-------|
| 0 | 368 |
| 1 | 785,247 |
| 2 | 1,249 |
| 3 | 121 |
| 4 | 16 |
| 5 | 4 |
| 6 | 2 |
