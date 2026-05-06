# Scanpro test for archetype abundance changes between NR and DR samples (cheng22).
# Loads combined parquet from script 26; merges P28 and P38 into a single NR vs DR test.
# P28_dl is excluded (separate DL condition).

import os
import pandas as pd
from scanpro import scanpro

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
PARQUET_IN  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '26.combined_labels.parquet')
OUT_NR_DR   = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '28.scanpro_NR_vs_DR.csv')

NR_AGES = ['P28', 'P38']
DR_AGES = ['P28_dr', 'P38_dr']

df = pd.read_parquet(PARQUET_IN)
df = df[df['dataset'] == 'cheng22'].reset_index(drop=True)
print(f'Loaded {len(df)} cheng22 cells')
print(f'  Age groups: {df["age"].value_counts().to_dict()}')

sub = df[df['age'].isin(NR_AGES + DR_AGES)].copy()
sub['condition'] = sub['age'].isin(NR_AGES).map({True: 'NR', False: 'DR'})
print(f'\nRunning scanpro: NR={NR_AGES} vs DR={DR_AGES}')
print(f'  Cells: {sub["condition"].value_counts().to_dict()}')
print(f'  Donors: {sub.groupby("condition")["donor_name"].nunique().to_dict()}')

res = scanpro(sub, clusters_col='archetype', conds_col='condition', samples_col='donor_name')
res.results.to_csv(OUT_NR_DR)
print(f'\nResults:\n{res.results}')
print(f'Saved {OUT_NR_DR}')
