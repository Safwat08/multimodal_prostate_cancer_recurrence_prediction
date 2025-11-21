from monai.data import Dataset as Dataset
from monai.data import DataLoader as DataLoader
from monai.data import ITKReader
import pandas as pd
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ResampleToMatchd,
    CropForegroundd,
    ResizeWithPadOrCropd,
    NormalizeIntensityd,
    ConcatItemsd,
    RandAffined,
    RandGaussianNoised,
    RandBiasFieldd,
    RandAdjustContrastd,
    NormalizeIntensityd,
    MapTransform

)
import numpy as np
import torch
from pathlib import Path
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torchsurv.loss import cox
from torchsurv.metrics.cindex import ConcordanceIndex

import os
from typing import Dict, Any, Optional, Type

import pandas as pd
import torch
import wandb

# Set target spacing
target_spacing = (1.0, 1.0, 3.0)

# Set target shape
target_shape   = (160, 160, 48)

# Set keys
keys_img = ["adc", "hbv", "t2w"]
keys_all = ["adc", "hbv", "t2w", "mask"]

# Train_transforms
train_transforms = Compose([
    LoadImaged(keys=keys_all,
               reader=ITKReader(),
               image_only=True),
    EnsureChannelFirstd(keys=keys_all),
    Orientationd(keys=keys_all, axcodes="RAS"),
    Spacingd(
        keys=["t2w"],
        pixdim=target_spacing,
        mode=("trilinear",),
    ),
    Spacingd(
        keys=["mask"],
        pixdim=target_spacing,
        mode=("nearest",),
    ),
    ResampleToMatchd(
        keys=["adc", "hbv"],
        key_dst="t2w",
        mode=("trilinear", "trilinear"),
    ),
    CropForegroundd(
        keys=keys_all,
        source_key="mask",
        margin=(16, 16, 2),
    ),
    ResizeWithPadOrCropd(
        keys=keys_all,
        spatial_size=target_shape,
    ),
        RandAffined(
        keys=keys_img,
        spatial_size=target_shape,
        rotate_range=(0.087, 0.087, 0.087),
        translate_range=(5, 5, 1),
        scale_range=(0.05, 0.05, 0.0),
        mode=("trilinear",) * len(keys_img),
        prob=0.5,
    ),
    NormalizeIntensityd(
        keys=keys_img,
        nonzero=True,
        channel_wise=True,
    ),
    ConcatItemsd(
        keys=["adc", "hbv", "t2w"],
        name="image",
        dim=0,
    ),
])

# Val_transforms
val_transforms = Compose([
    LoadImaged(keys=keys_all,
               reader=ITKReader(),
               image_only=True),
    EnsureChannelFirstd(keys=keys_all),
    Orientationd(keys=keys_all, axcodes="RAS"),
    Spacingd(
        keys=["t2w"],
        pixdim=target_spacing,
        mode=("trilinear",),
    ),
    Spacingd(
        keys=["mask"],
        pixdim=target_spacing,
        mode=("nearest",),
    ),
    ResampleToMatchd(
        keys=["adc", "hbv"],
        key_dst="t2w",
        mode=("trilinear", "trilinear"),
    ),
    CropForegroundd(
        keys=keys_all,
        source_key="mask",
        margin=(16, 16, 2),
    ),
    ResizeWithPadOrCropd(
        keys=keys_all,
        spatial_size=target_shape,
    ),
    NormalizeIntensityd(
        keys=keys_img,
        nonzero=True,
        channel_wise=True,
    ),
    ConcatItemsd(
        keys=["adc", "hbv", "t2w"],
        name="image",
        dim=0,
    ),
])

def preprocess_data(df, fold):
    """
    Preprocess clinical dataframe and generate train/validation splits for a 
    specific fold in cross-validation. The preprocessing pipeline includes:

        - Cleaning or correcting inconsistent clinical entries
        - Splitting data according to fold annotations
        - One-hot encoding of categorical variables
        - Scaling of continuous features (StandardScaler)
        
     Args:
        df (pd.DataFrame):
            Input dataframe containing clinical features, an 'event' column,
            a 'time_to_event' column, and a 'fold' column used for CV splits.
        fold (int):
            Fold index to treat as the validation split. All other folds form 
            the training dataset.

    Returns:
        train_df (pd.DataFrame):
            Fully preprocessed training dataframe.
        val_df (pd.DataFrame):
            Fully preprocessed validation dataframe.
    """
    
    # "earlier_therapy" column dropped: Only one unique entry for a few patients
    # "BCR_PCA" column dropped: Column used to define event "BCR", i.e. highly correlated could cause data leakage
    df = df.drop(columns=["earlier_therapy", "BCR_PSA"])

    # "capsular_penetration" column "x" entries replaced with "0", i.e. x and 0 are considered equal
    df["capsular_penetration"] = df["capsular_penetration"].replace("x", 0)

    # "tertiary_gleason" columns NaNs were filled with 0, i.e., 0 on an ordinal scale of 1 -5
    df[["tertiary_gleason"]] = df[["tertiary_gleason"]].fillna(0)

    # Convert pT mapping to ordinal scale
    pT_mapping = {
    "1": 1,
    "1a": 2,
    "1b": 3,
    "1c": 4,
    "2": 5,
    "2a": 6,
    "2b": 7,
    "2c": 8,
    "3": 9,
    "3a": 10,
    "3b": 11,
    "4": 12,
    "4b":12 # 4b does not exist in clinical definitions, assumed to mean 4, could be an error and so could also mean 2b/3b
    }

    df["pT_stage"] = df["pT_stage"].map(pT_mapping)
    
    # Split datasets
    train_df = df[df["fold"] != fold].copy().reset_index(drop=True)
    val_df   = df[df["fold"] == fold].copy().reset_index(drop=True)

    # OneHotEncode categorical columns
    categorical_cols = ["positive_lymph_nodes"]

    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )
    
    # Fit encoder on training dataset
    train_cat = encoder.fit_transform(train_df[categorical_cols])
    val_cat   = encoder.transform(val_df[categorical_cols])

    encoded_train_df = pd.DataFrame(
        train_cat,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=train_df.index
    )
    encoded_val_df = pd.DataFrame(
        val_cat,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=val_df.index
    )
    
    train_df = train_df.drop(columns=categorical_cols)
    val_df   = val_df.drop(columns=categorical_cols)

    train_df = pd.concat([train_df, encoded_train_df], axis=1)
    val_df   = pd.concat([val_df, encoded_val_df], axis=1)

    # StandardScaler for numerical columns
    scaler = StandardScaler()
    numeric_cols = ["age_at_prostatectomy", "pre_operative_PSA"]
    
    # Fit scaler on training dataset
    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])

    return(train_df, val_df)

def build_file_list(df, tabular = False, imaging = False):
    """
    Build a list of dictionary entries for use with MONAI Dataset objects.
    Each entry corresponds to a single patient and may include:

        - Patient ID
        - Survival labels (time, event)
        - Paths to MRI modalities (ADC, HBV, T2W) and segmentation mask
        - Preprocessed clinical feature vectors

    Args:
        df (pd.DataFrame):
            Input dataframe containing at least:
            - 'patient_id'
            - 'BCR' (event indicator)
            - 'time_to_follow-up/BCR' (time to event or censoring)
        tabular (bool):
            If True, include clinical features as a NumPy array in the output.
        imaging (bool):
            If True, include file paths to MRI modalities and segmentation masks.

    Returns:
        file_list (List[dict]):
            A list of dictionaries, each containing all metadata needed
            by MONAI transforms and Dataset/DataLoader pipelines. The keys may 
            include:
                - "id", "time", "event"
                - "adc", "hbv", "t2w", "mask"        (if imaging=True)
                - "features"                          (if tabular=True)
    """
    # Initiate empty list
    file_list = []

    # Iterate over dataframe
    for _, row in df.iterrows():
        
        # Identify patient_id
        patient_id = int(row["patient_id"])
        
        # Identify time to event/censoring
        time = float(row["time_to_follow-up/BCR"])
        
        # Identify event
        event = float(row["BCR"])
        
        # Make up initial entry
        entry = ({
            "id": patient_id,
            "time": time,
            "event": event,
        })
        
        # Update entry based on flags
        if imaging == True:
            
            # Include paths to MRI modalities (ADC, HBV, T2W) and segmentation mask
            entry.update({
                "adc":  f"radiology/mpMRI/{patient_id}/{patient_id}_0001_adc.mha",
                "hbv":  f"radiology/mpMRI/{patient_id}/{patient_id}_0001_hbv.mha",
                "t2w":  f"radiology/mpMRI/{patient_id}/{patient_id}_0001_t2w.mha",
                "mask": f"radiology/prostate_mask_t2w/{patient_id}_0001_mask.mha",
            })
            
        if tabular == True:
            
            # Include clinical features that exlcude "id", "fold", event and time-to-event columns
            feature_cols = [
                c for c in df.columns
                if c not in ["patient_id", "fold", "BCR", "time_to_follow-up/BCR"]
            ]
            
            entry.update({
                "features": row[feature_cols].to_numpy(dtype=np.float32)
            })
           
        # Append final entry
        file_list.append(entry)
        
    return file_list

def get_dataloaders(df, fold, data_type, batch_size=4, num_workers=0):
    
    """
    Construct PyTorch DataLoaders for a given cross-validation fold using
    either imaging-only, tabular-only, or multimodal clinical + MRI data.

    This function:
        1. Selects the desired modality (tabular, imaging, multimodal)
        2. Preprocesses the clinical dataframe for the selected fold
        3. Builds MONAI-compatible metadata dictionaries
        4. Wraps them into MONAI `Dataset` objects
        5. Returns PyTorch DataLoaders for training and validation

    Args:
        df (pd.DataFrame):
            Full clinical dataframe containing patient IDs, event/time labels,
            fold assignments, and optionally clinical feature columns.
        fold (int):
            Fold index used for identify validation set. All other folds
            become the training set.
        data_type (str):
            One of:
                - "imaging"   : MRI-only modeling
                - "tabular"   : clinical-only modeling
                - "multimodal": MRI + clinical fusion
        batch_size (int, default=4):
            Batch size for the training DataLoader.
        num_workers (int, default=0):
            Number of worker processes for data loading.

    Returns:
        train_loader (DataLoader):
            PyTorch DataLoader providing shuffled training samples.
        val_loader (DataLoader):
            PyTorch DataLoader providing full-batch validation samples
            (batch_size = len(validation_set)).
    """       

    # Set flags based on "data_type"
    if data_type == "imaging":
        tab_bool = False
        img_bool = True

    elif data_type == "tabular":
        tab_bool = True
        img_bool = False

    elif data_type == "multimodal":
        tab_bool = True
        img_bool = True

    else:
        raise ValueError(f"Unknown data_type: {data_type}")   
    

    # Preprocess and split dataframe based on fold
    train_df, val_df = preprocess_data(df, fold)
    
    # Initiate empty file lists
    train_files = []
    val_files = []
    
    # Build list of dictionary entries to use with MONAI Dataset
    train_files = build_file_list(train_df, tabular = tab_bool, imaging = img_bool)
    val_files   = build_file_list(val_df, tabular = tab_bool, imaging = img_bool)
    
    # Include tranformations im data_type is imaging or multimodal
    if img_bool == True:
        train_ds = Dataset(train_files, transform=train_transforms)
        val_ds = Dataset(val_files, transform=val_transforms)
    else:
        # No transformations needed if not imaging data
        train_ds = Dataset(train_files)
        val_ds = Dataset(val_files)
    
    # Build training and validation data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False, drop_last=False, num_workers=num_workers)
    
    return train_loader, val_loader

def one_epoch_run(model, optimizer, train_loader, val_loader, data_type, device):
    """
    Run a single training + validation epoch for a survival model using Cox
    proportional hazards loss, supporting imaging-only, tabular-only, or
    multimodal (MRI + clinical) inputs.

    The function:
        - Selects the correct inputs for the model based on `data_type`
        - Trains on all batches from `train_loader`
        - Evaluates on `val_loader` without gradient updates
        - Computes average train/val Cox loss and validation C-index
        - Aggregates per-patient predictions into a DataFrame

    Args:
        model (nn.Module):
            PyTorch model that accepts either:
                - imaging only: model(images)
                - tabular only: model(features)
                - multimodal: model(mri_images=images, clinical_features=features)
            and returns a tensor of log-risk scores.
        optimizer (torch.optim.Optimizer):
            Optimizer used to update model parameters.
        train_loader (DataLoader):
            DataLoader yielding training batches. Each batch is expected to be
            a dictionary with keys:
                - "time"   : survival time
                - "event"  : event indicator (0/1 or bool)
                - "image"  : MRI tensor (B, C, D, H, W) if imaging is used
                - "features": clinical features (B, F) if tabular is used
        val_loader (DataLoader):
            DataLoader yielding validation batches with the same structure as
            `train_loader`, plus an "id" key for patient IDs.
        data_type (str):
            One of:
                - "imaging"   : MRI-only model
                - "tabular"   : clinical-only model
                - "multimodal": MRI + clinical fusion model
        device (torch.device or str):
            Device on which to run the model (e.g., "cuda" or "cpu").

    Returns:
        avg_train_loss (float):
            Mean Cox negative partial log-likelihood over training batches.
        avg_val_loss (float):
            Mean Cox negative partial log-likelihood over validation batches.
        val_cindex (float):
            Concordance index computed on the full validation set.
        df_pred (pd.DataFrame):
            DataFrame containing per-patient predictions with columns:
                - "patient_id"
                - "time"
                - "event"
                - "predicted_risk"
    
    """
    
    # Set flags based on data_type
    if data_type == "imaging":
        tab_bool = False
        img_bool = True

    elif data_type == "tabular":
        tab_bool = True
        img_bool = False

    elif data_type == "multimodal":
        tab_bool = True
        img_bool = True

    else:
        raise ValueError(f"Unknown data_type: {data_type}")
        
    # Train loop
    model.train()
    running_train_loss = 0.0
    n_train_batches = 0

    for batch in train_loader:
        # Get inputs based on data_type
        if img_bool:
            images = batch["image"].to(device)
        else:
            images = None

        if tab_bool:
            features = batch["features"].to(device)
        else:
            features = None

        # Get time-to-event and events
        times  = batch["time"].float()
        events = batch["event"].bool()

        # Clear gradients per batch
        optimizer.zero_grad()

        # Model risk based on data_type
        if data_type == "multimodal":
            risk = model(mri_images=images, clinical_features=features)
        elif data_type == "imaging":
            risk = model(images)
        elif data_type == "tabular":
            risk = model(features)
        
        # Flatten risk scores from shape (B, 1) to (B,)
        risk = risk.squeeze(-1)
            
        # Compute Cox negative partial log-likelihood
        train_loss = cox.neg_partial_log_likelihood(risk, events, times)
        
        # Backpropagate 
        train_loss.backward()
        
        # Update graidents
        optimizer.step()
        
        # Add to running loss and batch
        running_train_loss += train_loss.item()
        n_train_batches += 1

    # Calculate average loss across batch    
    avg_train_loss = running_train_loss / max(n_train_batches, 1)

    # Validation loop
    model.eval()
    running_val_loss = 0.0
    n_val_batches = 0
    
    # Initiate empty lists 
    all_risk   = []
    all_times  = []
    all_events = []
    all_ids = []

    # Disable gradient computation for validation
    with torch.no_grad():
        for batch in val_loader:
            
            # Get inputs based on data_type
            if img_bool:
                images = batch["image"].to(device)
            else:
                images = None

            if tab_bool:
                features = batch["features"].to(device)
            else:
                features = None
            
            # Get time-to-event and events
            times  = batch["time"].float().to(device)
            events = batch["event"].bool().to(device)
            
            ids = batch["id"].cpu().tolist()

            # Model risk based on data_type
            if data_type == "multimodal":
                risk = model(mri_images=images, clinical_features=features)
            elif data_type == "imaging":
                risk = model(images)
            elif data_type == "tabular":
                risk = model(features)
            
            # Flatten risk scores from shape (B, 1) to (B,)
            risk = risk.squeeze(-1)
            
            # Compute Cox negative partial log-likelihood
            val_loss = cox.neg_partial_log_likelihood(risk, events, times)
            
            # Add to running loss and batch
            running_val_loss += val_loss.item()
            n_val_batches += 1

            # Append risk, time, events and ids
            all_risk.append(risk.detach().cpu())
            all_times.append(times.detach().cpu())
            all_events.append(events.detach().cpu())
            all_ids.extend(ids) 

    # Get average validation loss
    avg_val_loss = running_val_loss / max(n_val_batches, 1)

    # Concat all risks, times and events
    all_risk   = torch.cat(all_risk, dim=0)
    all_times  = torch.cat(all_times, dim=0)
    all_events = torch.cat(all_events, dim=0)
    
    # Get C-index
    cindex_metric = ConcordanceIndex()
    val_cindex = cindex_metric(all_risk, all_events, all_times).item()
    
    # Return predictions
    os.makedirs("predictions", exist_ok=True)
    
    risk_np   = all_risk.numpy().flatten()
    times_np  = all_times.numpy()
    events_np = all_events.numpy().astype(int)

    df_pred = pd.DataFrame({
        "patient_id": all_ids,
        "time": times_np,
        "event": events_np,
        "predicted_risk": risk_np,
    })
    
    return avg_train_loss, avg_val_loss, val_cindex, df_pred



def train_model_over_folds(
    model_class: Type[torch.nn.Module],
    model_name: str,
    data_type: str,
    model_kwargs: Optional[Dict[str, Any]] = None,
    clinical_csv: str = "clinical_data.csv",
    splits_csv: str = "data_split_5fold.csv",
    project_name: str = "mri_clinical_multimodal_prostate_survival",
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 4,
    num_workers: int = 4,
    num_epochs: int = 10,
    n_folds: int = 5,
    device: Optional[torch.device] = None,
):
    """
    Train and evaluate a survival model using K-fold cross-validation.

    This function:
        - Loads clinical metadata and fold assignments
        - Creates PyTorch DataLoaders for each fold
        - Instantiates and trains a new model per fold
        - Tracks train/val Cox loss, C-index, and best-epoch weights
        - Logs metrics to Weights & Biases (one run per fold)
        - Saves: 
        -- best model weights per fold
        -- per-patient predictions per fold
        -- fold-level summary CSV

    Args:
        model_class (Type[nn.Module]):
            The model class to instantiate (e.g., Multimodel_Model,
            MRI_CNN_Model, Clinical_MLP_Model).
        model_name (str):
            Model name used in logging, file names, and W&B grouping.
        data_type (str):
            One of {"multimodal", "imaging", "tabular"}.
            Controls which inputs are passed into each model and how
            get_dataloaders() prepares the dataset.
        model_kwargs (dict, optional):
            Keyword arguments passed to model_class(**model_kwargs).
        clinical_csv (str):
            Path to clinical metadata CSV containing patient_id, event, time,
            and clinical features.
        splits_csv (str):
            Path to 5-fold split CSV containing patient_id and fold assignment.
        project_name (str):
            Weights & Biases project name.
        lr (float):
            Learning rate for Adam.
        weight_decay (float):
            L2 weight decay regularization.
        batch_size (int):
            Batch size for training DataLoader.
        num_workers (int):
            Number of workers for data loading.
        num_epochs (int):
            Number of epochs per fold.
        n_folds (int):
            Total number of folds (default 5).
        device (torch.device, optional):
            Device to train on. If None, selects CUDA if available.
    
    No Returns
    """

    # Handle defaults
    if model_kwargs is None:
        model_kwargs = {}

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make sure dirs exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("predictions", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Load + merge data
    df = pd.read_csv(clinical_csv)
    splits = pd.read_csv(splits_csv)
    df = df.merge(splits, on="patient_id")

    # Initiate empty fold_results list
    fold_results = []

    print(f"\n### Training and Validating Model: {model_name} ###")
    print(f"Data type: {data_type}")

    for fold in range(n_folds):
        print(f"\n## FOLD {fold+1}/{n_folds} ##")

        # wandb init
        run = wandb.init(
            project=project_name,
            name=f"{model_name}_fold{fold+1}",
            group=model_name,
            config={
                "fold": fold + 1,
                "model": model_name,
                "data_type": data_type,
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "epochs": num_epochs,
                "loss": "cox_neg_partial_log_likelihood",
            },
            reinit=True,
        )

        # DataLoaders
        train_loader, val_loader = get_dataloaders(
            df,
            fold=fold,
            data_type=data_type,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # Model
        model = model_class(**model_kwargs).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        best_val_cindex = -1.0
        best_epoch = -1
        best_state = None
        best_preds_df = None

        # Epoch loop
        for epoch in range(num_epochs):
            print(f"\n# Fold {fold+1} | Epoch {epoch+1}/{num_epochs} #")

            train_loss, val_loss, val_cindex, df_pred = one_epoch_run(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                data_type=data_type,
                device=device,
            )

            print(
                f"Fold {fold+1} | Epoch {epoch+1}: "
                f"train_loss = {train_loss:.4f} | "
                f"val_loss = {val_loss:.4f} | "
                f"val_cindex = {val_cindex:.4f}"
            )

            wandb.log(
                {
                    "fold": fold + 1,
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_cindex": val_cindex,
                }
            )

            # Track best epoch
            if val_cindex > best_val_cindex:
                best_val_cindex = val_cindex
                best_epoch = epoch + 1
                best_state = model.state_dict()
                best_preds_df = df_pred

        # Save per-fold artifacts
        fold_results.append(
            {
                "model": model_name,
                "fold": fold + 1,
                "best_epoch": best_epoch,
                "best_val_cindex": best_val_cindex,
            }
        )

        # Best weights
        model_path = f"models/{model_name}_fold{fold}.pt"
        torch.save(
            best_state,
            model_path
        )

        print(f"\nSaved best model per fold to {model_path}")
        
        # Best predictions
        prediction_path = f"predictions/{model_name}_fold{fold}.csv"
        best_preds_df.to_csv(
            prediction_path,
            index=False,
        )
        
        print(f"\nSaved predictions per fold to {model_path}")

        # wandb summary
        run.summary["best_val_cindex"] = best_val_cindex
        run.summary["best_epoch"] = best_epoch
        wandb.finish()

    # Save fold_results CSV
    results_df = pd.DataFrame(fold_results)
    results_path = f"results/{model_name}_fold_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved fold results {results_path}")
    
    