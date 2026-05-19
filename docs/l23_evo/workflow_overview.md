# L2/3 IT Evolution Analysis — Conceptual Workflow

## 1. Human L2/3 IT characterization (Jorstad23 dataset)

**Input**: ~47k human V1 L2/3 IT neurons (snRNA-seq, 6 pre-annotated clusters, 3 donors, 2 labs)

**Steps**:
- Standard dimensionality reduction: HVG selection → PCA → UMAP
- Variance partitioning (partial R²): reveals PC1 is dominated by library size, not biology; PC3/PC5 are the cleaner cell-type axes
- **Varimax rotation** of the gene loading matrix: rotates within the PCA subspace to maximize loading sparsity per component, separating biological variation (VX2, ~71% cell-type R²) from technical variation (VX1, ~70% library-size R²)
- Marker gene discovery per cluster (Wilcoxon one-vs-rest)

**Key intermediate**: varimax component scores (VX1–VX10) with interpretable factor separation

---

## 2. Archetype decomposition (PCHA)

**Input**: varimax scores (VX2/VX6–VX10 — subtype-informative components)

**Steps**:
- Sweep NOC (number of archetypes) → select NOC=4
- **PCHA** (Principal Convex Hull Analysis): finds a simplex in the VX subspace whose vertices (archetypes) span cell type variation; cells are positioned as convex combinations of archetypes
- Identify archetype marker genes (top 300 cells near each vertex → Wilcoxon enrichment)

**Key intermediate**: human PCHA coordinate space (5D) + 4 archetype vertices

---

## 3. Mouse-to-human mapping (Yoo25 mouse L2/3 IT)

**Goal**: position mouse cells within the human archetype space to assess evolutionary conservation of subtypes

Three strategies were tried (in order):

**Strategy A — Direct cosine neighbor assignment (Script 12)**
- Select high-loading genes from human VX components → find mouse orthologs (~188 shared genes)
- Cosine similarity in shared gene space → top-k human neighbors → weighted-average embedding
- *Failure*: species mean shift dominates; 50% of neighbor slots go to ~73 hub human cells

**Strategy B — Harmony joint embedding (Scripts 14–15)**
- Run independent varimax + PCHA on mouse data to identify mouse subtype-informative genes
- Build shared gene space from *both* human and mouse subtype-variation genes (via orthologs)
- Joint PCA → Harmony batch correction (species) → cosine neighbors in corrected space → embed mouse in human PCHA coords
- *Partial improvement*: 574 unique human cells used (vs. 73), but manifold still concentrated

**Strategy C — Sinkhorn-Knopp balanced assignment (Script 16)**
- Same gene space as Strategy A (188 shared genes)
- Replace top-k assignment with **SK balanced transport**: iteratively normalize the similarity matrix so each human cell receives equal total mouse mass — eliminates hubs by construction
- Temperature sweep (τ=0.05→0.002): τ=0.005 identified as optimal (~20-cell soft neighborhood per mouse cell, balanced human coverage)

**Output**: mouse cells embedded in human PCHA space, visualized alongside human cells with archetype vertices
