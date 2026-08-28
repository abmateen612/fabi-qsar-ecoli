# Data

`fabi_ecoli_ic50_raw.csv` — raw bioactivity data pulled directly from the 
ChEMBL REST API for target CHEMBL1857 (*E. coli* FabI), filtered to 
standard_type = IC50 at query time. 192 records, unprocessed.

Cleaning, deduplication, and pIC50 conversion are handled in 
`fabi_qsar_pipeline.py`, not applied to this raw file.
