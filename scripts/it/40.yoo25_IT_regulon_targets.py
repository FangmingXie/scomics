"""Clean regulon -> target-gene tables from Yoo25 SCENIC+ eRegulons (IT layers).

The SCENIC+ output for each IT layer (L2/3, L4, L5IT, L6IT) is split into two
metadata files, "direct" and "extended". Each row is a TF -> enhancer-region ->
target-gene triplet, and the `eRegulon_name` encodes the regulon as
`<TF>_<direct|extended>_<sign>` (e.g. `Arnt_direct_+/+`, `Aff4_extended_+/+`),
where the sign pattern is one of `+/+`, `+/-`, `-/+`, `-/-` (TF2G / R2G regulation
directions). "Direct" uses direct TF-motif annotations; "extended" uses
extended/indirect annotations.

For each layer this script builds one tidy (long) table of regulons and their
unique target genes:
  - A regulon is identified by TF + sign pattern (direct/extended normalized away),
    e.g. `Arnt_+/+`. `Arnt_+/+` and `Arnt_-/+` are distinct regulons.
  - If a regulon exists in both direct and extended, keep its DIRECT targets;
    otherwise keep its EXTENDED targets.
  - The same gene appears once per enhancer region, so target pairs are deduped to
    unique (regulon, Gene).

Reads (per layer, where <L> is one of L2_3, L4, L5IT, L6IT):
  links/it/jainlab26_regulon/Yoo25_regulon_Sanjana/<L>/<L>_direct_e_regulon_metadata.csv
  links/it/jainlab26_regulon/Yoo25_regulon_Sanjana/<L>/<L>_extended_e_regulon_metadata.csv
Outputs (per layer):
  local_data/res/it/40.yoo25_<L>_regulon_targets.tsv
"""

import os
import pandas as pd

REGULON_DIR = "links/it/jainlab26_regulon/Yoo25_regulon_Sanjana"
LAYERS = ["L2_3", "L4", "L5IT", "L6IT"]
OUT_TSV_TMPL = "local_data/res/it/40.yoo25_{layer}_regulon_targets.tsv"

OUT_COLS = ["regulon", "TF", "regulation_direction", "source", "Gene"]

# keep only regulons with a positive R2G (second) sign; drop +/- and -/-
KEEP_DIRECTIONS = {"+/+", "-/+"}


def load_and_annotate(csv_path, source):
    """Load an eRegulon metadata CSV and add normalized regulon-identity columns."""
    df = pd.read_csv(csv_path, index_col=0)

    # normalized regulon key: drop the direct/extended tag, e.g. Arnt_direct_+/+ -> Arnt_+/+
    df["regulon"] = (
        df["eRegulon_name"]
        .str.replace("_direct_", "_", regex=False)
        .str.replace("_extended_", "_", regex=False)
    )
    # sign pattern is the suffix after the last underscore, e.g. +/+
    df["regulation_direction"] = df["eRegulon_name"].str.rsplit("_", n=1).str[-1]
    df["source"] = source

    # fail fast if the expected encoding is violated
    if not set(df["regulation_direction"].unique()) <= {"+/+", "+/-", "-/+", "-/-"}:
        raise ValueError(
            f"Unexpected sign pattern(s) in {csv_path}: "
            f"{sorted(df['regulation_direction'].unique())}"
        )

    # keep only regulons whose R2G (second) sign is positive
    df = df[df["regulation_direction"].isin(KEEP_DIRECTIONS)].copy()
    return df


def process_layer(layer):
    """Build and write the clean regulon -> target-gene table for one IT layer."""
    direct_csv = os.path.join(REGULON_DIR, layer, f"{layer}_direct_e_regulon_metadata.csv")
    extended_csv = os.path.join(REGULON_DIR, layer, f"{layer}_extended_e_regulon_metadata.csv")
    out_tsv = OUT_TSV_TMPL.format(layer=layer)

    direct = load_and_annotate(direct_csv, "direct")
    extended = load_and_annotate(extended_csv, "extended")

    direct_keys = set(direct["regulon"].unique())

    # keep all direct regulons; from extended keep only regulons absent from direct
    extended_only = extended[~extended["regulon"].isin(direct_keys)]
    combined = pd.concat([direct, extended_only], ignore_index=True)

    # collapse multiple-region rows -> unique (regulon, target gene) pairs
    table = (
        combined[OUT_COLS]
        .drop_duplicates(subset=["regulon", "Gene"])
        .sort_values(["regulon", "Gene"])
        .reset_index(drop=True)
    )

    # sanity checks
    n_direct = table.loc[table["source"] == "direct", "regulon"].nunique()
    n_extended = table.loc[table["source"] == "extended", "regulon"].nunique()
    n_pos = table.loc[table["regulation_direction"] == "+/+", "regulon"].nunique()
    n_neg = table.loc[table["regulation_direction"] == "-/+", "regulon"].nunique()
    assert table.duplicated(subset=["regulon", "Gene"]).sum() == 0, "duplicate (regulon, Gene) pairs remain"
    assert n_direct == len(direct_keys), f"direct regulon count mismatch: {n_direct} != {len(direct_keys)}"

    os.makedirs(os.path.dirname(out_tsv), exist_ok=True)
    table.to_csv(out_tsv, sep="\t", index=False)

    print(f"[{layer}] regulons total         : {table['regulon'].nunique()}")
    print(f"[{layer}]   from direct          : {n_direct}")
    print(f"[{layer}]   from extended (only) : {n_extended}")
    print(f"[{layer}]   +/+ (activating)     : {n_pos}")
    print(f"[{layer}]   -/+ (repressing)     : {n_neg}")
    print(f"[{layer}] unique target genes    : {table['Gene'].nunique()}")
    print(f"[{layer}] target pairs (rows)    : {len(table)}")
    print(f"[{layer}] wrote -> {out_tsv}")


def main():
    for layer in LAYERS:
        process_layer(layer)


if __name__ == "__main__":
    main()
