from src.utils import preprocess_data
import pandas as pd
import numpy as np
from sksurv.util import Surv
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
import joblib
import os

def main():
    
    # Load data
    df = pd.read_csv("clinical_data.csv")
    splits = pd.read_csv("data_split_5fold.csv")

    # Merge folds
    df = df.merge(splits, on="patient_id")

    n_folds = 5

    fold_results = []

    # Clinical Linear
    print(f"\n### Training and Validating Model: clinical_linear_model ###")
    print(f"Data type: tabular")
    for fold in range(n_folds):

        train_df, val_df = preprocess_data(df, fold)

        # survival labels
        y_train = Surv.from_arrays(
            event=train_df["BCR"].astype(bool).values,
            time=train_df['time_to_follow-up/BCR'].values
        )

        X_train = train_df.drop(columns=["BCR", "time_to_follow-up/BCR", "fold", "patient_id"])
        X_val = val_df.drop(columns=["BCR", "time_to_follow-up/BCR", "fold", "patient_id"])

        # fit Cox model
        model = CoxnetSurvivalAnalysis(l1_ratio = 0.0001)
        model.fit(X_train, y_train)

        # predict on validation
        risk_val = model.predict(X_val)

        # compute C-index
        val_cindex = concordance_index_censored(
            val_df["BCR"].astype(bool).values,
            val_df["time_to_follow-up/BCR"].values,
            risk_val
        )[0]

        print(
            f"Fold {fold+1}: "
            f"val_cindex = {val_cindex:.4f}"
        )

        fold_results.append({
            "model": "clinical_linear_model",
            "fold": fold + 1,
            "best_epoch": "NA",
            "best_val_cindex": val_cindex,
        })

        df_pred = pd.DataFrame({
            "patient_id": val_df["patient_id"].values,
            "event": val_df["BCR"].astype(int).values,
            "time": val_df["time_to_follow-up/BCR"].values,
            "predicted_risk": risk_val,
        })
        
        os.makedirs("predictions", exist_ok=True)
        
        # save predictions
        pred_path = f"predictions/clinical_linear_model_fold{fold+1}.csv"
        df_pred.to_csv(pred_path, index=False)

        print(f"Saved predictions to {pred_path}")
        
        os.makedirs("models", exist_ok=True)
        
        # save model
        model_path = f"models/clinical_linear_model_fold{fold+1}.joblib"
        joblib.dump(model, model_path)
        print(f"Saved model to {model_path}")

        os.makedirs("results", exist_ok=True)    
        
        # save results
        results_df = pd.DataFrame(fold_results)
        results_path = f"results/clinical_linear_model_fold_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"Saved fold results to {results_path}")

if __name__ == "__main__":
    main()