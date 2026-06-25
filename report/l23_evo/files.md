# Regulon → target-gene table — human (l23_evo script 27)

Clean regulon membership table derived from the Wang25 human SCENIC+ eRegulon
supplementary table, produced by `scripts/l23_evo/27.human_wang25_regulon_targets.py`.
This is the human analog of the mouse Yoo25 tables in `report/it/files.md`
(`scripts/it/40.yoo25_IT_regulon_targets.py`).

| Species | Output file | Regulons | +/+ | -/+ | direct | extended-only | Genes | Target pairs |
|---|---|---|---|---|---|---|---|---|
| Human (Wang25) | `local_data/res/l23_evo/27.human_wang25_regulon_targets.tsv` | 582 | 385 | 197 | 534 | 48 | 7,412 | 56,280 |

`Genes` = number of distinct target genes; `Target pairs` = rows (one per unique
regulon–gene pair). `+/+` and `-/+` are the regulon counts by sign pattern
(activating vs repressing TF→gene). Only regulons with a positive R2G (second) sign are kept
(`+/+`, `-/+`); `+/-` and `-/-` regulons are dropped — this dataset happens to
contain only `+/+` and `-/+` regulons, so none are removed. There is no
direct/extended overlap, so all 582 regulons are retained.

## How it's produced

The input is a single Excel file (`human_regulons_SuppTable13.xlsx`), read with the
Python standard library (`zipfile` + `ElementTree`) to avoid an openpyxl
dependency. Sheet **13b** (eRegulon target genes) is used; each row is one eRegulon
with a comma-separated `Target genes` cell. The script:

1. Parses the eRegulon name `<TF>[_extended]_<sign>_<sign>` (e.g. `ETV1_+_+` =
   direct, `RORB_extended_+_+` = extended, `NKX6-2_-_+`). There is no explicit
   "direct" tag — a regulon is direct unless it carries `_extended_`. Signs are
   joined with `/` (`+_+` → `+/+`) so the schema matches the mouse tables.
2. Normalizes each regulon to **TF + sign pattern** (e.g. `RORB_extended_+/+` →
   `RORB_+/+`). If a regulon exists in both direct and extended, keeps its
   **direct** targets; otherwise keeps the **extended** targets.
3. Explodes the comma-separated gene list to unique `(regulon, Gene)` pairs.
4. Keeps only regulons whose R2G (second) sign is positive (`+/+`, `-/+`),
   dropping `+/-` and `-/-`.

## File schema

Columns: `regulon, TF, regulation_direction, source, Gene`

One row per regulon–target-gene pair (long format), grouped by the `regulon`
column. `source` is `direct` or `extended`; `regulation_direction` is the sign
pattern (`+/+` or `-/+` — only positive-R2G regulons are retained). Identical
schema to the mouse Yoo25 tables.
