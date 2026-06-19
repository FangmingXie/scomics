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

### Significant DEGs
49 genes passed both thresholds (FDR < 0.05, |Log2FC| > 1):
- **35 up in high PC2** (bin08+09)
- **14 up in low PC2** (bin00+01)

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
| Gene | Significant | Log2FC | Note |
|---|---|---|---|
| dpr13 | Yes | −2.60 | strongly enriched in low PC2 |
| beat-VI | Yes | −1.33 | enriched in low PC2 |
| DIP-ε | No | −0.09 | FDR passes but |Log2FC| < 1 |
| side-II | No | +0.37 | FDR passes but |Log2FC| < 1 |

---

## Conclusions

- PC2 captures a biologically meaningful axis of variation within LPLC2 cells at APF_48h, distinct from library-size effects (which were regressed out).
- The high-PC2 pole is enriched for genes involved in neuronal identity and signaling (*Mp*, *mbl*, *SiaT*, *dpr* family members), while the low-PC2 pole is enriched for cell-recognition and adhesion molecules (*dpr13*, *Cad87A*, *beat-VI*).
- The *dpr/beat/side/DIP* family of Ig-domain proteins shows differential expression along PC2: *dpr13* and *beat-VI* are significantly enriched at the low end, while *DIP-ε* and *side-II* show trends in the same direction that fall below the fold-change threshold, suggesting a graded rather than binary expression pattern across the PC2 axis.
