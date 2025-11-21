# combine_predictions.py
# Combine predictions into one file
import pandas as pd
import glob
from pathlib import Path


def main():
    
    dfs = []

    for file in glob.glob("predictions/*.csv"):
        df = pd.read_csv(file)

        stem = Path(file).stem

        fold = stem[-1]

        model = stem[:-5]

        df["model"] = model
        df["fold"] = fold

        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv("prediction_results_per_model.csv", index=False)
    
if __name__ == "__main__":
    main()