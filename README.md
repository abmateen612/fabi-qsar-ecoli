# FabI QSAR Model — E. coli

A ligand-based QSAR model predicting inhibitory potency (pIC50) against 
FabI (enoyl-ACP reductase) from *Escherichia coli*, a validated antibacterial 
drug target in fatty acid biosynthesis.

## Pipeline

ChEMBL IC50 data → clean & filter → dedupe & convert to pIC50 → 
RDKit 2D descriptors → Random Forest regression → validation 
(cross-validation, y-randomization, applicability domain)

## Data

- Source: ChEMBL (target CHEMBL1857, *E. coli* FabI)
- 192 raw IC50 records → 129 after cleaning (exact values, nM units only) 
  → 110 unique compounds after deduplication

## Model

- Random Forest Regressor (200 trees), 8 RDKit 2D descriptors 
  (MolWt, LogP, TPSA, H-bond donors/acceptors, rotatable bonds, aromatic rings, ring count)
- 5-fold cross-validation mean R² = 0.438
- Held-out test R² = 0.556
- Y-randomization control: shuffled-label R² = -0.319 (confirms model 
  captures real structure-activity signal, not noise)
- Applicability domain (leverage/Williams plot): 2/110 compounds flagged 
  outside domain

## Results

See `results/` for feature importance, predicted-vs-actual, and 
applicability domain plots.

## Requirements

See `requirements.txt`

## Usage

Run `fabi_qsar_pipeline.py` — full pipeline from raw data to validated model.
