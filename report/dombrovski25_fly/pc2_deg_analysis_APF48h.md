# PC2 Structure and DEG Analysis — LPLC2 APF_48h

**Dataset:** Dombrovski 2025 fly scRNA-seq (`dombrovski25_fly.h5ad`)  
**Cell population:** LPLC2, APF_48h time point (1,442 cells)  
**Scripts:** 07 · 08 · 09

---

## Procedures

### Preprocessing (script 07)
- Filtered to LPLC2 cells; removed 38 mitochondrial genes (`mt:` prefix), leaving 17,786 genes.
- Selected top 2,000 highly variable genes (HVGs) for PCA.
- Normalized with CP10k → log2(1+x) → z-score per gene; regressed out log(nCount_RNA) to remove library-size effects.
- Ran PCA (20 components). PC1 explained 7.6% and PC2 2.3% of variance.
- Binned all 1,442 cells into 10 equal-sized quantile bins along PC2 (~144–145 cells per bin, `bin00` = lowest PC2, `bin09` = highest).
- Results saved to `local_data/res/dombrovski25_fly/07.lplc2_APF48h_pca.parquet`.

### DEG testing (script 08)
- Compared PC2 extremes: **Group A** (bin00 + bin01, low PC2, 289 cells) vs **Group B** (bin08 + bin09, high PC2, 289 cells).
- Tested all 11,869 expressed non-mt genes (5,917 zero-variance genes excluded).
- Test: Wilcoxon rank-sum on CP10k+log2(1+x) expression values (non-parametric; log-normalized values used so Log2FC is biologically interpretable).
- FDR correction: Benjamini-Hochberg. Significance threshold: FDR < 0.05 and |Log2FC| > 1.

### Visualization (script 09)
- Volcano plot: Log2FC vs −log10(FDR) with threshold lines and top-15 gene labels.
- PC2 vs expression scatter+line plot for the top 20 significant DEGs: individual cells as scatter points colored by PC2 bin; per-bin mean shown as a black line with colored markers. Gene dropdown for interactive exploration.

---

## Results

### Significant DEGs at varying Log2FC thresholds

| \|Log2FC\| threshold | Total | Up in high PC2 | Up in low PC2 |
|---|---|---|---|
| > log2(1.5) ≈ 0.585 | 185 | 118 | 67 |
| > 1.0 (2-fold) | 49 | 35 | 14 |

### Top hits

| Gene | Log2FC | FDR | Direction |
|---|---|---|---|
| Mp | +1.92 | 7.8×10⁻⁸⁵ | high PC2 |
| jus | +1.43 | 5.5×10⁻⁸² | high PC2 |
| mbl | +1.72 | 3.4×10⁻⁸¹ | high PC2 |
| Cow | +1.31 | 4.3×10⁻⁸¹ | high PC2 |
| dpr13 | −2.60 | 4.6×10⁻⁷⁸ | low PC2 |
| Cad87A | −1.62 | 3.3×10⁻⁷⁸ | low PC2 |
| SiaT | +1.77 | 2.3×10⁻⁷⁵ | high PC2 |
| beat-VI | −1.33 | 5.6×10⁻³³ | low PC2 |

### Genes of interest
| Gene | Log2FC | FDR | Sig at \|Log2FC\|>1 | Sig at \|Log2FC\|>log2(1.5) |
|---|---|---|---|---|
| dpr13 | −2.60 | 4.6×10⁻⁷⁸ | Yes | Yes |
| beat-VI | −1.33 | 5.6×10⁻³³ | Yes | Yes |
| DIP-ε | −0.09 | 1.3×10⁻⁴ | No | No |
| side-II | +0.37 | 4.2×10⁻¹² | No | No |

---

## Pv vs Sst — Outgroup Comparison (Mouse P56 VIS Cortex)

**Dataset:** Gao 2025 mouse DevVIS scRNA-seq (`DevVIS_scRNA_processed.h5ad`)  
**Cell population:** P56, VIS cortex — Pvalb Gaba (1,018 cells) vs Sst Gaba (1,102 cells)  
**Scripts:** 10 · 11

### Procedures

- Filtered to P56 cells; removed 13 mitochondrial genes (`mt-` prefix), leaving 32,272 genes.
- Library depth computed from raw count matrix (no `nCount_RNA` column in obs).
- Tested all 28,705 expressed non-mt genes (constant genes excluded).
- Test: Wilcoxon rank-sum on CP10k+log2(1+x) expression (same normalization as fly analysis).
- FDR correction: Benjamini-Hochberg.

### Significant DEGs at varying Log2FC thresholds

| \|Log2FC\| threshold | Total | Up in Sst Gaba | Up in Pvalb Gaba |
|---|---|---|---|
| > 0.5 | 462 | 215 | 247 |
| > 1.0 (2-fold) | 143 | 75 | 68 |

### Top hits (ranked by FDR)

| Gene | Log2FC | FDR | Direction |
|---|---|---|---|
| Sst | +4.07 | ~0 | Sst Gaba |
| Grin3a | +3.42 | ~0 | Sst Gaba |
| Synpr | +3.37 | ~0 | Sst Gaba |
| Cacna2d3 | +2.91 | ~0 | Sst Gaba |
| Kcnh7 | −2.47 | 7.5×10⁻³⁰¹ | Pvalb Gaba |
| Srrm4 | −1.76 | 2.2×10⁻³⁰² | Pvalb Gaba |
| Elfn1 | +1.91 | ~0 | Sst Gaba |
| Slc4a4 | −1.66 | ~0 | Pvalb Gaba |
| Myo1e | −1.20 | 4.4×10⁻³⁰⁶ | Pvalb Gaba |
| Cox6a2 | −1.07 | ~0 | Pvalb Gaba |

### Comparison: LPLC2 PC2 extremes vs Pv vs Sst

The Pv vs Sst difference is substantially larger than the within-type LPLC2 PC2 variation, as expected for a between-subtype comparison:

| Metric | Fly LPLC2 PC2 extremes | Mouse Pv vs Sst |
|---|---|---|
| Genes tested | 11,869 | 28,705 |
| DEGs \|Log2FC\| > 0.5 | 229 | 462 (~2×) |
| DEGs \|Log2FC\| > 1.0 | 49 | 143 (~3×) |
| % genes FDR < 0.05 | 15.9% | 31.3% (~2×) |
| Mean \|Log2FC\| (sig, >1) | 1.29 | 1.51 |
| Max \|Log2FC\| | 2.60 | 4.11 |

Pv vs Sst yields ~3× more significant DEGs at the 2-fold threshold and roughly double the genome-wide significant fraction. The fly PC2 comparison is notable given its much smaller gene set, tighter effect sizes, and that it captures continuous within-type heterogeneity rather than a discrete subtype boundary.

---

## Conclusions

- PC2 captures a biologically meaningful axis of variation within LPLC2 cells at APF_48h, distinct from library-size effects (which were regressed out).
- The high-PC2 pole is enriched for genes involved in neuronal identity and signaling (*Mp*, *mbl*, *SiaT*, *dpr* family members), while the low-PC2 pole is enriched for cell-recognition and adhesion molecules (*dpr13*, *Cad87A*, *beat-VI*).
- The *dpr/beat/side/DIP* family of Ig-domain proteins shows differential expression along PC2: *dpr13* and *beat-VI* are significantly enriched at the low end, while *DIP-ε* and *side-II* show trends in the same direction that fall below the fold-change threshold, suggesting a graded rather than binary expression pattern across the PC2 axis.
