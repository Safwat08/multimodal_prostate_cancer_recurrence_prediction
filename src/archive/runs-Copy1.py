#!/usr/bin/env python

import argparse
from src.utils import train_model_over_folds
from src.models import Multimodal_Model, Clinical_MLP_Model, MRI_CNN_Model, 


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train prostate cancer recurrence models."
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--full_multimodal",
        action="store_true",
        help="Train the Full Multimodal model.",
    )
    group.add_argument(
        "--multimodal_mri_ablated",
        action="store_true",
        help="Train the Multimodal model MRI ablated.",
    )
    group.add_argument(
        "--multimodal_clinical_ablated",
        action="store_true",
        help="Train the Multimodal model clinical ablated.",
    )
    group.add_argument(
        "--clinical_mlp",
        action="store_true",
        help="Train the Clinical MLP unimodal model.",
    )
    group.add_argument(
        "--mri_cnn",
        action="store_true",
        help="Train the MRI CNN unimdal model.",
    )
    group.add_argument(
        "--clinical_linear",
        action="store_true",
        help="Train the Clinical Linear model.",
    )
    group.add_argument(
        "--clinical_mlp_1e_4",
        action="store_true",
        help="Train the Clinical MLP unimodal model at learning rate 1e-4.",
    )
    group.add_argument(
        "--full_multimodal_dual_lr",
        action="store_true",
        help="Train the Full Multimodal model using the dual LR optimizer",
    )
    group.add_argument(
        "--mri_cnn_precropped",
        action="store_true",
        help="Train the MRI CNN unimodal model using precropped preprocessing",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Train ALL models.",
    )
    return parser.parse_args()


# Individual model flags
run_full_multimodal = args.full_multimodal
run_multimodal_mri_ablated = args.multimodal_mri_ablated
run_multimodal_clinical_ablated = args.multimodal_clinical_ablated
run_clinical_mlp = args.clinical_mlp
run_mri_cnn = args.mri_cnn
run_clinical_linear = args.clinical_linear
run_clinical_mlp_1e_4 = args.clinical_mlp_1e_4
run_full_multimodal_dual_lr = args.full_multimodal_dual_lr
run_mri_cnn_precropped = args.mri_cnn_precropped

run_all_flag = args.all

# Compute run_all correctly
individual_flags = [
    run_full_multimodal,
    run_multimodal_mri_ablated,
    run_multimodal_clinical_ablated,
    run_clinical_mlp,
    run_mri_cnn,
    run_clinical_linear,
    run_clinical_mlp_1e_4,
    run_full_multimodal_dual_lr,
    run_mri_cnn_precropped,
]

run_all = run_all_flag or not any(individual_flags)

# Clinical MLP Model
train_model(model_class = Clinical_MLP_Model,
                       model_name = "clinical_mlp_model",
                       data_type = "tabular",
                       model_kwargs = {"input_dim": 14,
                                       "feature_dim": 32,
                                       "dropout": 0.1,
                                       "output_dim": 1},
                       clinical_csv = "clinical_data.csv",
                       splits_csv = "data_split_5fold.csv",
                       project_name = "prostate_cancer_recurrence_mri_clinical_multimodal",
                       lr = 1e-3,
                       weight_decay = 1e-4,
                       batch_size = 4,
                       num_workers = 4,
                       num_epochs = 10,
                       n_folds = 5)

# MRI CNN Model
if run_all or run_mri_cnn:
train_model(model_class = MRI_CNN_Model,
            model_name = "mri_cnn_model",
            data_type = "imaging",
            model_kwargs = {"output_dim": 1,
                                       "feature_dim": 16,
                                       "dropout": 0.3,
                                       "use_adc": True,
                                       "use_hbv": True,
                                       "use_t2w": True},
                       clinical_csv = "clinical_data.csv",
                       splits_csv = "data_split_5fold.csv",
                       project_name = "prostate_cancer_recurrence_mri_clinical_multimodal",
                       lr = 1e-4,
                       weight_decay = 1e-4,
                       batch_size = 4,
                       num_workers = 4,
                       num_epochs = 10,
                       n_folds = 5)

if run_all or run_full_multimodal:
train_model(model_class = Multimodal_Model,
                       model_name = "multimodal_model_full",
                       data_type = "multimodal",
                       model_kwargs = {"mri_feature_dim": 16,
                                       "mri_dropout": 0.3,
                                       "clinical_input_dim": 14,
                                       "clinical_feature_dim": 32,
                                       "clinical_dropout": 0.1,
                                       "predictor_feature_dim": 32,
                                       "predictor_dropout": 0.3,
                                       "use_adc": True,
                                       "use_hbv": True,
                                       "use_t2w": True,
                                       "ablate_mri": False,
                                       "ablate_clinical": False},
                       clinical_csv = "clinical_data.csv",
                       splits_csv = "data_split_5fold.csv",
                       project_name = "prostate_cancer_recurrence_mri_clinical_multimodal",
                       lr = 1e-4,
                       weight_decay = 1e-4,
                       batch_size = 4,
                       num_workers = 4,
                       num_epochs = 10,
                       n_folds = 5)

# Multimodal_Model clinical ablated
if run_all or run_multimodal_clinical_ablated:
train_model(model_class = Multimodal_Model,
                       model_name = "multimodal_model_clinical_ablated",
                       data_type = "multimodal",
                       model_kwargs = {"mri_feature_dim": 16,
                                       "mri_dropout": 0.3,
                                       "clinical_input_dim": 14,
                                       "clinical_feature_dim": 32,
                                       "clinical_dropout": 0.1,
                                       "predictor_feature_dim": 32,
                                       "predictor_dropout": 0.3,
                                       "use_adc": True,
                                       "use_hbv": True,
                                       "use_t2w": True,
                                       "ablate_mri": False,
                                       "ablate_clinical": True},
                       clinical_csv = "clinical_data.csv",
                       splits_csv = "data_split_5fold.csv",
                       project_name = "prostate_cancer_recurrence_mri_clinical_multimodal",
                       lr = 1e-4,
                       weight_decay = 1e-4,
                       batch_size = 4,
                       num_workers = 4,
                       num_epochs = 10,
                       n_folds = 5)

# Multimodal_Model MRI ablated
if run_all or run_multimodal_mri_ablated:
train_model_over_folds(model_class = Multimodal_Model,
                       model_name = "multimodal_model_mri_ablated",
                       data_type = "multimodal",
                       model_kwargs = {"mri_feature_dim": 16,
                                       "mri_dropout": 0.3,
                                       "clinical_input_dim": 14,
                                       "clinical_feature_dim": 32,
                                       "clinical_dropout": 0.1,
                                       "predictor_feature_dim": 32,
                                       "predictor_dropout": 0.3,
                                       "use_adc": True,
                                       "use_hbv": True,
                                       "use_t2w": True,
                                       "ablate_mri": True,
                                       "ablate_clinical": False},
                       clinical_csv = "clinical_data.csv",
                       splits_csv = "data_split_5fold.csv",
                       project_name = "prostate_cancer_recurrence_mri_clinical_multimodal",
                       lr = 1e-4,
                       weight_decay = 1e-4,
                       batch_size = 4,
                       num_workers = 4,
                       num_epochs = 10,
                       n_folds = 5)

if run_all or run_clinical_mlp_1e_4:
# Clinical MLP Model
train_model(model_class = Clinical_MLP_Model,
                       model_name = "clinical_mlp_model_lr1e_4",
                       data_type = "tabular",
                       model_kwargs = {"input_dim": 14,
                                       "feature_dim": 32,
                                       "dropout": 0.1,
                                       "output_dim": 1},
                       clinical_csv = "clinical_data.csv",
                       splits_csv = "data_split_5fold.csv",
                       project_name = "prostate_cancer_recurrence_mri_clinical_multimodal",
                       lr = 1e-4,
                       weight_decay = 1e-4,
                       batch_size = 4,
                       num_workers = 4,
                       num_epochs = 10,
                       n_folds = 5)


if run_all or run_full_multimodal_dual_lr:
    
# Multimodal_Model with Dual LR optimizer
train_model(model_class = Multimodal_Model,
                       model_name = "multimodal_model_full_diff_lr",
                       data_type = "multimodal",
                       model_kwargs = {"mri_feature_dim": 16,
                                       "mri_dropout": 0.3,
                                       "clinical_input_dim": 14,
                                       "clinical_feature_dim": 32,
                                       "clinical_dropout": 0.1,
                                       "predictor_feature_dim": 32,
                                       "predictor_dropout": 0.3,
                                       "use_adc": True,
                                       "use_hbv": True,
                                       "use_t2w": True,
                                       "ablate_mri": False,
                                       "ablate_clinical": False},
                       clinical_csv = "clinical_data.csv",
                       splits_csv = "data_split_5fold.csv",
                       project_name = "prostate_cancer_recurrence_mri_clinical_multimodal",
                       use_diff_lr = True,
                       cnn_lr = 1e-4,
                       mlp_lr = 1e-3,
                       lr = 1e-4,
                       weight_decay = 1e-4,
                       batch_size = 4,
                       num_workers = 4,
                       num_epochs = 10,
                       n_folds = 5)

# MRI CNN precropped preprocessing
if run_all or run_mri_cnn_precropped:
    
train_models(model_class = MRI_CNN_Model,
             model_name = "mri_cnn_model_precropped",
               data_type = "imaging",
               model_kwargs = {"output_dim": 1,
                               "feature_dim": 16,
                               "dropout": 0.3,
                               "use_adc": True,
                               "use_hbv": True,
                               "use_t2w": True},
               clinical_csv = "clinical_data.csv",
               splits_csv = "data_split_5fold.csv",
               project_name = "prostate_cancer_recurrence_mri_clinical_multimodal",
             preprocessing_type = "precropped",
               lr = 1e-4,
               weight_decay = 1e-4,
               batch_size = 4,
               num_workers = 4,
               num_epochs = 10,
               n_folds = 5)






train_model_over_folds(model_class = Multimodal_Model,
                       model_name = "multimodal_model",
                       data_type = "multimodal",
                       model_kwargs = {"mri_feature_dim": 16,
                                       "mri_dropout": 0.3,
                                       "clinical_input_dim": 14,
                                       "clinical_feature_dim": 32,
                                       "clinical_dropout": 0.1,
                                       "predictor_feature_dim": 32,
                                       "predictor_dropout": 0.3,
                                       "use_adc": True,
                                       "use_hbv": True,
                                       "use_t2w": True,
                                       "ablate_mri": False,
                                       "ablate_clinical": False},
                       clinical_csv = "clinical_data.csv",
                       splits_csv = "data_split_5fold.csv",
                       project_name = "prostate_cancer_recurrence_mri_clinical_multimodal",
                       lr = 1e-4,
                       weight_decay = 1e-4,
                       batch_size = 4,
                       num_workers = 4,
                       num_epochs = 10,
                       n_folds = 5)

# Clinical MLP Model
train_model_over_folds(model_class = Clinical_MLP_Model,
                       model_name = "clinical_mlp_model",
                       data_type = "tabular",
                       model_kwargs = {"input_dim": 14,
                                       "feature_dim": 32,
                                       "dropout": 0.1,
                                       "output_dim": 1},
                       clinical_csv = "clinical_data.csv",
                       splits_csv = "data_split_5fold.csv",
                       project_name = "prostate_cancer_recurrence_mri_clinical_multimodal",
                       lr = 1e-3,
                       weight_decay = 1e-4,
                       batch_size = 4,
                       num_workers = 4,
                       num_epochs = 10,
                       n_folds = 5)


# MRI CNN Model
train_model_over_folds(model_class = MRI_CNN_Model,
                       model_name = "mri_cnn_model",
                       data_type = "imaging",
                       model_kwargs = {"output_dim": 1,
                                       "feature_dim": 16,
                                       "dropout": 0.3,
                                       "use_adc": True,
                                       "use_hbv": True,
                                       "use_t2w": True},
                       clinical_csv = "clinical_data.csv",
                       splits_csv = "data_split_5fold.csv",
                       project_name = "prostate_cancer_recurrence_mri_clinical_multimodal",
                       lr = 1e-4,
                       weight_decay = 1e-4,
                       batch_size = 4,
                       num_workers = 4,
                       num_epochs = 10,
                       n_folds = 5)


# Clinical MLP
train_model_over_folds(model_class = Clinical_MLP,
                       model_name = "clinical_mlp",
                       data_type = "tabular",
                       clinical_csv = "clinical_data.csv",
                       splits_csv = "data_split_5fold.csv",
                       project_name = "mri_clinical_multimodal_prostate_survival",
                       lr = 1e-3,
                       weight_decay = 1e-4,
                       batch_size = 4,
                       num_workers = 4,
                       num_epochs = 10,
                       n_folds = 5)

# MRI_CNN
train_model_over_folds(model_class = MRI_CNN,
                       model_name = "mri_cnn",
                       data_type = "imaging",
                       clinical_csv = "clinical_data.csv",
                       splits_csv = "data_split_5fold.csv",
                       project_name = "mri_clinical_multimodal_prostate_survival",
                       lr = 1e-4,
                       weight_decay = 1e-4,
                       batch_size = 4,
                       num_workers = 4,
                       num_epochs = 10,
                       n_folds = 5)

# Multimodal_CNN_MLP
train_model_over_folds(model_class = Multimodal_CNN_MLP,
                       model_name = "multimodal_cnn_mlp",
                       data_type = "multimodal",
                       clinical_csv = "clinical_data.csv",
                       splits_csv = "data_split_5fold.csv",
                       project_name = "mri_clinical_multimodal_prostate_survival",
                       lr = 1e-4,
                       weight_decay = 1e-4,
                       batch_size = 4,
                       num_workers = 4,
                       num_epochs = 10,
                       n_folds = 5)

# Multimodal_CNN_MLP Clinical Ablation
train_model_over_folds(model_class = Multimodal_CNN_MLP,
                       model_name = "multimodal_cnn_mlp_clinical_ablation",
                       data_type = "multimodal",
                       model_kwargs = {"ablate_clinical" = True},
                       clinical_csv = "clinical_data.csv",
                       splits_csv = "data_split_5fold.csv",
                       project_name = "mri_clinical_multimodal_prostate_survival",
                       lr = 1e-4,
                       weight_decay = 1e-4,
                       batch_size = 4,
                       num_workers = 4,
                       num_epochs = 10,
                       n_folds = 5)

# Multimodal_CNN_MLP Clinical Ablation
train_model_over_folds(model_class = Multimodal_CNN_MLP,
                       model_name = "multimodal_cnn_mlp_mri_ablation",
                       data_type = "multimodal",
                       model_kwargs = {"ablate_mri" = True},
                       clinical_csv = "clinical_data.csv",
                       splits_csv = "data_split_5fold.csv",
                       project_name = "mri_clinical_multimodal_prostate_survival",
                       lr = 1e-3,
                       weight_decay = 1e-4,
                       batch_size = 4,
                       num_workers = 4,
                       num_epochs = 10,
                       n_folds = 5)


