import pandas as pd
import random
import os

os.makedirs("survey_versions", exist_ok=True)

fine_df = pd.read_csv("style_instruction.csv")
noinst_df = pd.read_csv("no_instruction.csv")


full_df = pd.concat([fine_df, noinst_df], ignore_index=True)

# feminine
fem_df = full_df[[
    "item_id",
    "data_source",
    "source_dataset",
    "feminine_style"
]].copy()

fem_df = fem_df.rename(columns={"feminine_style": "short_text"})
fem_df["style_condition"] = "feminine"

# masculine
masc_df = full_df[[
    "item_id",
    "data_source",
    "source_dataset",
    "masculine_style"
]].copy()

masc_df = masc_df.rename(columns={"masculine_style": "short_text"})
masc_df["style_condition"] = "masculine"

all_data = pd.concat([fem_df, masc_df], ignore_index=True)

# split into 5 versions
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

    ]).sample(frac=1, random_state=version).reset_index(drop=True)

    version_df.to_csv(
        f"version_{version}.csv",
        index=False
    )

print("Created 5 survey versions.")