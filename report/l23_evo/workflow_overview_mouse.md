# Mouse L2/3 IT Characterization — Conceptual Workflow (Cheng22, P28)

## Overview

Independent characterization of mouse L2/3 IT subtype structure using the same analytical framework as the human Jorstad23 pipeline (Scripts 18–20). Rather than projecting mouse cells into human space, this pipeline builds a mouse-native varimax + PCHA representation to identify mouse-specific archetypes and their marker genes.

**Input**: Cheng22 P28 mouse cortex snRNA-seq, filtered to L2/3 IT → ~4,044 cells across 3 annotated subtypes (L2/3_A, L2/3_B, L2/3_C) from multiple samples.

---

## Script 18 — Dimensionality Reduction and Variance Partitioning

**Goal**: Identify a low-dimensional subspace that captures subtype variation while separating out technical confounders (library size, batch).

**Steps**:
- Normalize raw counts to log2(CP10k + 1)
- Select top 2,000 highly variable genes (by variance)
- Center/scale genes → PCA (10 components)
- **Variance partitioning on raw PCs** (partial R²): for each PC, decompose explained variance into contributions from cell type, sample (batch), and library size. Reveals that PC1 is cell-type dominated (type R² = 0.64), PC3 is batch-dominated (sample R² = 0.48), and PC4 is library-size dominated (libsize R² = 0.58) — technical signals are mixed across PCs.
- **Varimax rotation** of the gene loading matrix: rotates within the PCA subspace to maximize loading sparsity per component. Separates the confounders into dedicated axes rather than spreading them across all components.
- **Variance partitioning on rotated VX components**: confirms that varimax concentrated library-size variation into VX3 (libsize R² = 0.59) and batch variation into VX5 (sample R² = 0.58), while freeing VX1/VX2/VX6/VX7/VX8/VX10 to carry predominantly cell-type signal (type R² = 0.10–0.65).

**Key intermediate**: varimax component scores (VX1–VX10) with interpretable factor separation; pre/post-varimax variance partition reports for visual inspection.

---

## Script 19 — Archetype Number Selection

**Goal**: Determine the optimal number of archetypes (NOC) for the mouse L2/3 IT VX subspace.

**Input**: Varimax scores restricted to the 6 cell-type-informative components (VX1, VX2, VX6, VX7, VX8, VX10) — selected based on high cell-type R² and low batch/library-size R² in the Script 18 variance partition.

**Steps**:
- Construct a low-dimensional representation of the selected VX subspace via internal PCA (NDIM = 5; fits 6 components, drops the last)
- Sweep NOC from 2 to 6; for each NOC, fit PCHA and compute three metrics:
  - **Explained variance (EV)**: fraction of VX subspace variance captured by the archetype simplex
  - **Archetype Relative Variation (ARV)**: instability of archetype positions across bootstrap replicates; low ARV = reproducible archetypes
  - **Effective EV** = EV × (1 − ARV): penalizes EV by instability; peaks at the most informative stable NOC
- Repeat across samples to assess per-sample reproducibility (ARV_rep)

**NOC = 3 selected**: highest effective EV (0.66) with low ARV (0.036), indicating three stable, information-rich archetypes. At NOC = 4, ARV spikes to 0.42, signaling that a fourth archetype cannot be reliably placed — the data does not support finer splitting.

---

## Script 20 — Archetype Fitting and Marker Gene Discovery

**Goal**: Fit the final archetype model and identify the gene expression signatures of each archetype.

**Steps**:
- Fit PCHA at NOC = 3 on the 6-component VX subspace (NDIM = 5): places three archetype vertices spanning the cell cloud; each cell receives barycentric coordinates within the simplex
- **Archetype cell assignment**: for each archetype vertex, select the 300 nearest cells in VX space (Euclidean distance) as its representative population
- **Wilcoxon one-vs-rest enrichment**: for each archetype's 300 cells vs. all remaining cells, compute a rank-sum test per gene; correct for multiple testing (FDR-BH); retain genes with detection fraction ≥ 25% in the archetype group and FDR < 0.001
- Results: 1,468 total marker genes (554 / 373 / 541 per archetype), each archetype with a distinct gene signature

**Outputs and visualizations**:
- Marker gene table with log2FC, FDR, and detection fractions per archetype
- **PCHA scatter** (PCHA coordinate space): cells colored by subtype or sample, with archetype vertices and connecting simplex overlaid
- **PCA reference scatter** (raw PC1–PC3, pre-varimax): same metadata coloring, for comparison with the rotated space — illustrates how batch/library-size signals dominated the raw PCs before varimax
- **Gene expression scatter** (PCHA space): interactive dropdown over the top 5 marker genes per archetype (15 genes total, deduplicated), expression shown as log2(CP10k + 1) with colorbar clipped at 5th–95th percentile
