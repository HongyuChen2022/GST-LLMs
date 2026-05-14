import pandas as pd
import os

os.makedirs("survey_versions", exist_ok=True)

# =========================================================
# LOAD MAIN DATASETS
# =========================================================

fine_df = pd.read_csv("style_instruction.csv")
noinst_df = pd.read_csv("no_instruction.csv")


full_df = pd.concat([fine_df, noinst_df], ignore_index=True)

# feminine texts
fem_df = full_df[[
    "item_id",
    "data_source",
    "source_dataset",
    "feminine_style"
]].copy()

fem_df = fem_df.rename(columns={"feminine_style": "short_text"})
fem_df["style_condition"] = "feminine"

# masculine texts
masc_df = full_df[[
    "item_id",
    "data_source",
    "source_dataset",
    "masculine_style"
]].copy()

masc_df = masc_df.rename(columns={"masculine_style": "short_text"})
masc_df["style_condition"] = "masculine"

all_data = pd.concat([fem_df, masc_df], ignore_index=True)

all_data["is_attention_check"] = False

# =========================================================
# LOAD ATTENTION CHECKS
# =========================================================

attention_df = pd.read_csv("attention_checks.csv")

attention_df["is_attention_check"] = True
attention_df["item_id"] = "attention_check"
attention_df["data_source"] = "attention_check"
attention_df["source_dataset"] = "attention_check"
attention_df["style_condition"] = "attention_check"

# =========================================================
# CREATE 5 SURVEY VERSIONS
# =========================================================

for version in range(5):

    version_df = pd.concat([

        all_data[
            (all_data["source_dataset"] == "style_instructions") &
            (all_data["style_condition"] == "feminine")
        ].sample(10, random_state=version),

        all_data[
            (all_data["source_dataset"] == "style_instructions") &
            (all_data["style_condition"] == "masculine")
        ].sample(10, random_state=version),

        all_data[
            (all_data["source_dataset"] == "no_instructions") &
            (all_data["style_condition"] == "feminine")
        ].sample(10, random_state=version),

        all_data[
            (all_data["source_dataset"] == "no_instructions") &
            (all_data["style_condition"] == "masculine")
        ].sample(10, random_state=version),

        attention_df

    ]).sample(frac=1, random_state=version + 100).reset_index(drop=True)

    version_df.to_csv(
        f"survey_versions/version_{version}.csv",
        index=False
    )

print("Created 5 survey versions with 3 attention checks each.")