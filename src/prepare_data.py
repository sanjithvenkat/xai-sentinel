
import pandas as pd
from pathlib import Path

input_path = Path("data/traffic.csv")
output_path = Path("data/traffic_sample.csv")

print("Reading dataset...")
df = pd.read_csv(input_path, low_memory=False)
df.columns = df.columns.str.strip()

print("\nOriginal dataset shape:", df.shape)
print("\nLabel counts:")
print(df["Label"].value_counts(dropna=False))

# Remove rows with missing labels
df = df.dropna(subset=["Label"]).copy()

# Keep a reproducible sample, preserving representation of each label
sample_parts = []
for label, group in df.groupby("Label", sort=False):
    sample_parts.append(group.sample(
        n=min(len(group), 10000),
        random_state=42
    ))

sample = pd.concat(sample_parts, ignore_index=True)
sample = sample.sample(frac=1, random_state=42).reset_index(drop=True)

sample.to_csv(output_path, index=False)

print("\nSample shape:", sample.shape)
print("\nSample label counts:")
print(sample["Label"].value_counts())
print(f"\nSaved to: {output_path}")
