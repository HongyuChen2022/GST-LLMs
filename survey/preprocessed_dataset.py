import pandas as pd
import numpy as np

# settings
NUM_VERSIONS = 10
ROWS_PER_DATASET_PER_VERSION = 5
SEED = 40

# load the two datasets
df1 = pd.read_csv("survey/ds_style_instructions.csv")
df2 = pd.read_csv("survey/ds_no_instructions.csv")

# add source labels
df1["source_dataset"] = "style_instructions"
df2["source_dataset"] = "no_instructions"

# add item ids
df1["item_id"] = [f"style_{i+1}" for i in range(len(df1))]
df2["item_id"] = [f"noinst_{i+1}" for i in range(len(df2))]

# shuffle each dataset once
rng = np.random.default_rng(SEED)
df1 = df1.iloc[rng.permutation(len(df1))].reset_index(drop=True)
df2 = df2.iloc[rng.permutation(len(df2))].reset_index(drop=True)

all_versions = []

for version in range(1, NUM_VERSIONS + 1):
    start = (version - 1) * ROWS_PER_DATASET_PER_VERSION
    end = start + ROWS_PER_DATASET_PER_VERSION

    part1 = df1.iloc[start:end].copy()
    part2 = df2.iloc[start:end].copy()

    part1["survey_version"] = version
    part2["survey_version"] = version

    combined = pd.concat([part1, part2], ignore_index=True)

    # shuffle inside each version
    combined = combined.sample(frac=1, random_state=SEED + version).reset_index(drop=True)
    combined["pair_in_version"] = range(1, len(combined) + 1)

    all_versions.append(combined)

final_df = pd.concat(all_versions, ignore_index=True)

# save
final_df.to_csv("survey/versioned_dataset.csv", index=False)

print("Saved to survey/versioned_dataset.csv")
print(final_df[["survey_version", "pair_in_version", "source_dataset", "item_id"]].head(20))