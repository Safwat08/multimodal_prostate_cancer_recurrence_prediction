import pandas as pd
import glob
from pathlib import Path

def main():

    files = glob.glob("results/*.csv")

    rows = [] 

    for file in files:
        path = Path(file)
        data = pd.read_csv(path)

        model_name = path.stem.split("_fold")[0]

        data["Model/Variant"] = model_name

        rows.append(
            data[["Model/Variant", "best_val_cindex", "fold"]]
            .rename(columns={"best_val_cindex": "C-index",
                            "fold": "Fold"})
        )

    cindex = pd.concat(rows, ignore_index=True)

    cindex.to_csv("Cindices_per_fold.csv", index=False)
    
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