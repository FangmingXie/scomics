import os
import pandas as pd
import anndata as ad

# --- file paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'raw', 'dombrowski25_fly')
OUT_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'processed', 'dombrowski25_fly')
OUT_FILE = os.path.join(OUT_DIR, 'dombrowski25_fly.h5ad')

SAMPLES = {
    'APF_48h': 'GSM8837595',
    'APF_72h': 'GSM8837596',
    'APF_96h': 'GSM8837597',
}

# --- load and process each sample ---
adatas = []
for sample_name, gsm_id in SAMPLES.items():
    counts_file = os.path.join(RAW_DIR, f'{gsm_id}_{sample_name}_counts1.tsv.gz')
    meta2_file = os.path.join(RAW_DIR, f'{gsm_id}_{sample_name}_metadata2.tsv.gz')

    counts = pd.read_csv(counts_file, sep='\t', index_col=0)  # genes x cells
    meta2 = pd.read_csv(meta2_file, sep='\t')

    # transpose to cells x genes, keep only metadata2 cells
    counts_t = counts.T
    cells_keep = meta2['cell_barcode'].values
    counts_filtered = counts_t.loc[cells_keep]

    obs = meta2.set_index('cell_barcode')
    var = pd.DataFrame(index=counts_filtered.columns)

    adata = ad.AnnData(X=counts_filtered.values, obs=obs, var=var)
    # prefix barcodes with sample name to ensure uniqueness across samples
    adata.obs_names = [f'{sample_name}_{bc}' for bc in adata.obs_names]
    print(f'{sample_name}: {adata}')
    adatas.append(adata)

# --- merge and save ---
adata_all = ad.concat(adatas, merge='same')
print(f'\nMerged: {adata_all}')

os.makedirs(OUT_DIR, exist_ok=True)
adata_all.write_h5ad(OUT_FILE)
print(f'Saved {OUT_FILE}')
