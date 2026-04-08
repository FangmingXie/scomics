import os
import sys
import warnings
import pandas as pd
import anndata as ad
import scanpro

warnings.filterwarnings('ignore')

# --- file paths ---
SCRIPTS_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT        = os.path.dirname(SCRIPTS_DIR)
TARGET_FILE         = os.path.join(PROJECT_ROOT, 'local_data', 'source', 'morcom26_cux2mice', 'P26_EN-L2-3-CTX_EN-L4-5-CTX_EN-L2-mix.h5ad')
LABEL_TRANSFER_FILE = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'morcom26', '16.label_transfer.tsv')
OUT_RES_DIR         = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'morcom26')
OUT_FIG_DIR         = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'morcom26_cux2mice')
OUT_TSV_FILE        = os.path.join(OUT_RES_DIR, '20.scanpro_results.tsv')
OUT_FRAC_HTML       = os.path.join(OUT_FIG_DIR, '20.fraction_barplot.html')

# --- config ---
PRED_TYPE_COL  = 'pred_type'
PRED_TYPES     = ['L2/3_A', 'L2/3_B', 'L2/3_C']
KEEP_CELLTYPES = ['EN-L2-3-CTX', 'EN-L2-mix']

sys.path.insert(0, SCRIPTS_DIR)
from viz import stacked_bar_html

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)


def fraction_table(df, group_col):
    counts = df.groupby([group_col, PRED_TYPE_COL]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=PRED_TYPES, fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0)


# --- load ---
print(f'Loading label transfer results: {LABEL_TRANSFER_FILE}')
df = pd.read_csv(LABEL_TRANSFER_FILE, sep='\t', index_col=0)

print(f'Loading target metadata: {TARGET_FILE}')
target = ad.read_h5ad(TARGET_FILE)
df['samples']   = target.obs.loc[df.index, 'samples'].values
df['Condition'] = target.obs.loc[df.index, 'Condition'].values

# --- filter to EN-L2-3-CTX + EN-L2-mix cells and L2/3 subtypes only ---
df = df[df['celltype'].isin(KEEP_CELLTYPES)].copy()
df = df[df[PRED_TYPE_COL].isin(PRED_TYPES)].copy()
print(f'  {len(df)} cells after filtering to celltypes={KEEP_CELLTYPES} and pred_types={PRED_TYPES}')

# --- fraction barplot ---
overall    = df[PRED_TYPE_COL].value_counts(normalize=True).reindex(PRED_TYPES, fill_value=0)
frac_cond  = fraction_table(df, 'Condition')
frac_samp  = fraction_table(df, 'samples')
frac_ct    = fraction_table(df, 'celltype')
conditions = sorted(df['Condition'].unique())
samples    = sorted(df['samples'].unique())
celltypes  = sorted(df['celltype'].unique())

overall_df = pd.DataFrame([overall.values], index=['Overall'], columns=PRED_TYPES)
stacked_bar_html(
    panel_data=[
        ('Overall',       ['Overall'], overall_df),
        ('Per condition', conditions,  frac_cond),
        ('Per sample',    samples,     frac_samp),
        ('Per celltype',  celltypes,   frac_ct),
    ],
    celltypes=PRED_TYPES,
    title='Predicted L2/3 subtype fractions — EN-L2-3-CTX & EN-L2-mix (cheng22_yoo25 → morcom26)',
    out_path=OUT_FRAC_HTML,
)

# --- scanpro: test proportion changes across conditions ---
print('Running scanpro...')
result = scanpro.scanpro(
    data=df,
    clusters_col=PRED_TYPE_COL,
    conds_col='Condition',
    samples_col='samples',
    transform='logit',
    robust=True,
    verbosity=1,
)

print(result.results)

result.results.index.name = 'pred_type'
result.results.to_csv(OUT_TSV_FILE, sep='\t')
print(f'Saved scanpro results -> {OUT_TSV_FILE}')

print('Done.')
