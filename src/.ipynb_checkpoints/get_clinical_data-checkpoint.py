import json
import pandas as pd
import glob
from pathlib import Path

def main():
    
    files = glob.glob("clinical_data/*.json")

    rows = []

    for f in files:
        with open(f, "r") as fp:
            data = json.load(fp)


           # Add filename without extension
            data["patient_id"] = Path(f).stem


            rows.append(data)

    df = pd.DataFrame(rows)

    df = pd.DataFrame(rows)
    other_cols = [c for c in df.columns if c != "patient_id"]
    df = df[["patient_id"] + other_cols]


    df.to_csv("clinical_data.csv", index=False)

if __name__ == "__main__":
    main()