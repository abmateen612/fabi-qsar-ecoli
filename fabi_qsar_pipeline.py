# Step 1: Pull FabI bioactivity data from ChEMBL (direct REST API, no client library)

import requests
import pandas as pd

# --- user-configurable ---
TARGET_CHEMBL_ID = "CHEMBL1857"   # we can change chembl id when needed
ACTIVITY_TYPE = "IC50"
OUTPUT_CSV = r"D:\QSAR_model\fabi_ecoli_ic50_raw.csv"
# -------------------------

base_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
all_records = []
url = base_url
params = {
    "target_chembl_id": TARGET_CHEMBL_ID,
    "standard_type": ACTIVITY_TYPE,
    "limit": 10000
}

while url:
    resp = requests.get(url, params=params if url == base_url else None)
    data = resp.json()
    all_records.extend(data["activities"])
    next_page = data["page_meta"].get("next")
    url = f"https://www.ebi.ac.uk{next_page}" if next_page else None
    
    
### Step 2: Load and clean the raw data.

df = pd.DataFrame(all_records)
print(f"Total rows pulled: {len(df)}")
print(df[["molecule_chembl_id", "canonical_smiles", "standard_value",
          "standard_units", "assay_description"]].head())

df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved to {OUTPUT_CSV}")


df = pd.read_csv(r"D:\QSAR_model\fabi_ecoli_ic50_raw.csv")

# keep only exact IC50 values (drop >, < censored data)
df = df[df["standard_relation"] == "="]

# keep only rows with valid SMILES and numeric IC50 in nM
df = df.dropna(subset=["canonical_smiles", "standard_value"])
df = df[df["standard_units"] == "nM"]

print(f"Rows after cleaning: {len(df)}")
df.to_csv(r"D:\QSAR_model\fabi_ecoli_ic50_clean.csv", index=False)


### Step 3: Deduplicate compounds and convert IC50 → pIC50.

import numpy as np

# average IC50 for compounds tested more than once
df = df.groupby(["molecule_chembl_id", "canonical_smiles"], as_index=False)["standard_value"].mean()

# convert nM to pIC50 (-log10 of molar concentration)
df["pIC50"] = -np.log10(df["standard_value"] * 1e-9)

print(f"Unique compounds: {len(df)}")
df.to_csv(r"D:\QSAR_model\fabi_ecoli_pic50.csv", index=False)

### Step 4: Compute RDKit descriptors.

from rdkit import Chem
from rdkit.Chem import Descriptors

def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return pd.Series({
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumAromaticRings": Descriptors.NumAromaticRings(mol),
        "RingCount": Descriptors.RingCount(mol)
    })

desc_df = df["canonical_smiles"].apply(get_descriptors)
df = pd.concat([df, desc_df], axis=1)
df = df.dropna()  # drop rows where SMILES failed to parse

print(f"Rows with valid descriptors: {len(df)}")
df.to_csv(r"D:\QSAR_model\fabi_ecoli_descriptors.csv", index=False)


### Step 5: Train a baseline Random Forest model with cross-validation.

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

feature_cols = ["MolWt", "LogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds", "NumAromaticRings", "RingCount"]
X = df[feature_cols]
y = df["pIC50"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)

# 5-fold CV on training set (better estimate of performance than a single split, given small N)
cv_scores = cross_val_score(model, X_train, y_train, cv=KFold(5, shuffle=True, random_state=42), scoring="r2")
print(f"CV R2 scores: {cv_scores}")
print(f"Mean CV R2: {cv_scores.mean():.3f}")

model.fit(X_train, y_train)
test_r2 = r2_score(y_test, model.predict(X_test))
print(f"Test R2: {test_r2:.3f}")


### Step 6: y-randomization control.

y_shuffled = np.random.RandomState(42).permutation(y_train)
cv_scores_random = cross_val_score(model, X_train, y_shuffled, cv=KFold(5, shuffle=True, random_state=42), scoring="r2")
print(f"Y-randomized mean CV R2: {cv_scores_random.mean():.3f}")


### Step 7: Feature importance plot

import matplotlib.pyplot as plt

importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values()

plt.figure(figsize=(6, 4))
importances.plot(kind="barh")
plt.xlabel("Feature Importance")
plt.title("RF Feature Importance - FabI QSAR")
plt.tight_layout()
plt.savefig(r"D:\QSAR_model\feature_importance.png", dpi=300)
plt.show()

### Step 8: Predicted vs. actual scatter plot

y_pred = model.predict(X_test)

plt.figure(figsize=(5, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")  # ideal fit line
plt.xlabel("Actual pIC50")
plt.ylabel("Predicted pIC50")
plt.title(f"Predicted vs Actual (Test R2 = {test_r2:.3f})")
plt.tight_layout()
plt.savefig(r"D:\QSAR_model\pred_vs_actual.png", dpi=300)
plt.show()

### Step 9: Applicability domain (leverage / Williams plot)

X_arr = X.values
hat_matrix = X_arr @ np.linalg.pinv(X_arr.T @ X_arr) @ X_arr.T
leverage = np.diag(hat_matrix)

n, p = X_arr.shape
h_star = 3 * (p + 1) / n  # standard warning threshold

plt.figure(figsize=(6, 4))
plt.scatter(leverage, model.predict(X) - y, alpha=0.7)
plt.axvline(h_star, color="r", linestyle="--", label=f"h* = {h_star:.3f}")
plt.xlabel("Leverage")
plt.ylabel("Residual (pred - actual)")
plt.title("Williams Plot - Applicability Domain")
plt.legend()
plt.tight_layout()
plt.savefig(r"D:\QSAR_model\applicability_domain.png", dpi=300)
plt.show()

print(f"Compounds outside applicability domain: {(leverage > h_star).sum()} / {n}")