"""
Generate CSV inputs for a simulated active learning workflow.

Draw SAMPLES_PER_CLASS records from each diagnosis class to form the initial
labeled training set. Save the remaining feature rows as an unlabeled pool
and their ground-truth labels separately for simulated label queries.
Each class in the input dataset must contain at least SAMPLES_PER_CLASS records.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set the random seed for reproducibility
np.random.seed(42)

# Path configuration
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "datasets" / "blood_cells_features" / "blood_cells_data.csv"
OUTPUT_DIR = BASE_DIR / "datasets" / "active_learning_data"

# Create the output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Number of samples to draw per class
SAMPLES_PER_CLASS = 100

print("=" * 50)
print("Active learning data generator")
print("=" * 50)

# Read the source dataset
print(f"\nReading source data: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"Total samples: {len(df)}")
print(f"Features: {len(df.columns) - 1}")  # Exclude the target column
print(f"Target column: diagnosis")

# Inspect the class distribution
print("\nClass distribution in the source dataset:")
print(df['diagnosis'].value_counts())

# Stratified sampling: draw the same number of samples from each class
train_indices = []
for label in df['diagnosis'].unique():
    label_indices = df[df['diagnosis'] == label].index.tolist()
    sampled = np.random.choice(label_indices, size=SAMPLES_PER_CLASS, replace=False)
    train_indices.extend(sampled)

# Create the training set and unlabeled pool
train_df = df.loc[train_indices].copy()
pool_indices = df.index.difference(train_indices)
pool_df = df.loc[pool_indices].copy()

# Remove target labels from the unlabeled pool
pool_df_unlabeled = pool_df.drop(columns=['diagnosis'])

# Save the labeled training set
train_output = OUTPUT_DIR / "initial_train_labeled.csv"
train_df.to_csv(train_output, index=False)
print(f"\nInitial training data saved: {train_output}")
print(f"  - Samples: {len(train_df)}")
print(f"  - Class distribution:")
print(train_df['diagnosis'].value_counts().to_string())

# Save the unlabeled pool without target labels
pool_output = OUTPUT_DIR / "unlabeled_pool.csv"
pool_df_unlabeled.to_csv(pool_output, index=False)
print(f"\nUnlabeled sample pool saved: {pool_output}")
print(f"  - Samples: {len(pool_df_unlabeled)}")
print(f"  - Features: {len(pool_df_unlabeled.columns)}")

# Also save the ground-truth pool labels for simulated active learning queries
pool_labels_output = OUTPUT_DIR / "pool_true_labels.csv"
pool_df[['diagnosis']].to_csv(pool_labels_output, index=False)
print(f"\nGround-truth labels for the unlabeled pool saved: {pool_labels_output}")
print(f"  (Used to simulate label queries during active learning)")

print("\n" + "=" * 50)
print("Data generation completed.")
print("=" * 50)
print(f"\nOutput directory: {OUTPUT_DIR}")
print("\nGenerated files:")
print(f"  1. initial_train_labeled.csv  - Initial labeled training data")
print(f"  2. unlabeled_pool.csv         - Unlabeled sample pool")
print(f"  3. pool_true_labels.csv       - Ground-truth pool labels for simulation")
