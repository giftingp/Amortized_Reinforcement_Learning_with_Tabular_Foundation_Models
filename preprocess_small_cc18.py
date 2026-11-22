"""
preprocess_cc18_for_mothernet.py
---------------------------------
Load the CC‑18 test datasets pickle and preprocess them for evaluation.
Follows the MotherNet paper (Section 4.1):
  - 30 datasets with ≤2000 samples.
  - 50/50 train‑test split, repeated 5 times (stratified).
  - Missing values are replaced with zeros (no other imputation).
  - Only numeric features are kept; categorical features are ignored for simplicity
    (the paper includes them after one‑hot encoding, but for a quick benchmark
    we follow the original TabPFN protocol that uses only numeric features).
  - Data are saved as .npz files in ./cc18_preprocessed/.
"""

import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------
PKL_PATH = "C:/Users/Philip/Desktop/Teknisk fysik/MEX/V.20 Working Model/cc_test_datasets_multiclass_full.pkl"
OUT_DIR = "./cc18_preprocessed"
N_SPLITS = 5
RANDOM_SEEDS = [42, 43, 44, 45, 46]

os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------------
# Load the pickle
# --------------------------------------------------------------------------------
with open(PKL_PATH, 'rb') as f:
    datasets = pickle.load(f)

print(f"Loaded {len(datasets)} datasets from pickle.")

# --------------------------------------------------------------------------------
# Preprocess each dataset
# --------------------------------------------------------------------------------
for entry in datasets:
    name = entry[0]
    X_full = entry[1].cpu().numpy()      # shape (N, d)
    y_full = entry[2].cpu().numpy()      # shape (N,)

    X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)
    n_classes = len(np.unique(y_full))
    n_features = X_full.shape[1]
    total_samples = len(y_full)

    print(f"\nDataset: {name}")
    print(f"  Samples: {total_samples}, Features: {n_features}, Classes: {n_classes}")

    for split_idx, seed in enumerate(RANDOM_SEEDS):
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=0.5, random_state=seed, stratify=y_full
        )
        fname = os.path.join(OUT_DIR, f"{name}_split_{split_idx+1}_seed_{seed}.npz")
        np.savez_compressed(fname,
                            X_train=X_train, y_train=y_train,
                            X_test=X_test, y_test=y_test,
                            classes=n_classes,
                            features=n_features,
                            dataset_name=name,
                            split=split_idx+1,
                            seed=seed)
        print(f"  Saved split {split_idx+1} -> {fname}")

print(f"\nAll datasets preprocessed and saved to '{OUT_DIR}'.")