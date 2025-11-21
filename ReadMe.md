This github repo assesses a multimodal architechture to integrate multiparametric MRI images and clinical tabular data. Please read the attached "Techincal_Report.pdf" for further details

# Huggingface
This repository contains several trained model variants used for prostate cancer recurrence prediction.
Each model was trained for 10 epochs per fold in a 5-fold cross-validation setup.
For every fold, the best-performing checkpoint (based on validation C-index) was saved.

All model files follow this naming pattern: <model_name>_foldX.pt, where X denotes the fold (0-4)

Available Models
1. Clinical Linear Model 
Description: Regularized Cox proportional hazards model using clinical data
Prefix: clinical_linear_model

2. Clinical MLP
Description: Fully connected neural network using clinical data. Learning rate 1e-3
Prefix: clinical_mlp_model

3. MRI CNN
Description: 3D CNN trained on full MRI volumes (ADC, HBV, T2W + mask channel). Preprocessing includes mask as a seperate channel
Prefix: mri_cnn_model

4. Full Multimodal Model
Description: Multimodal model integrating Clinical MLP backbone to process clinical data and MRI CNN backbone to process MRI volumes. Feature embeddings from each branch passed through a risk prediction MLP
Prefix: multimodal_model_full

5. Multimodal Model — MRI Ablated
Description: Multimodal model with MRI CNN branch ablated
Prefix: mutlimodal_model_mri_ablated

6. Multimodal Model — Clinical Ablated
Description: Multimodal model with Clinical MLP branch ablated
Prefix: mutlimodal_model_clinical_ablated

7. Clinical MLP - Learning Rate 1e-4
Description: Fully connected neural network using clinical data (Same as #2) but with a Learning rate 1e-4
Prefix: clinical_mlp_model_lr1e_4

8. MRI CNN - Precropped
Description: 3D CNN trained on full MRI volumes (ADC, HBV, T2W). However, channels cropped using T2W-mask prior to concatinating into 1 tensor. 
Prefix: mri_cnn_model_precropped

1. Full Multimodal Model (Dual Learning Rate Optimizer)
Description: Same as #4 but uses a dual learning-rate optimizer (higher LR for the clinical MLP, lower LR for the MRI CNN) to balance learning between modalities.
** The strongest-performing variant across all folds.
prefix: multimodal_model_full_diff_lr





# Results
The final results are available as a csv file in "Cindices_per_fold.csv"

To replicate teh results please follow 

# Datasets


# Environment/Requirements
Install requirements using the following command:
conda create -n env "m31_technical" using requirements.yml

# Preprocessing 
Create "clinical_data.csv" using this command:
python src/get_clinical.data.py

This merges all the json files for the clinical patients and produces a "clinical_data.csv" file

# Training and Inference
To run training and inference run the following commands:
This will save the predictions per fold in "predictions", the best epoch model per fold in "models", the C-index values per fold in "results"

1. To train and run the Clincal Linear model:
python src/train_clinical_linear.py

2. To train and run the Clinical MLP unimodal model:
python src/train --clinicaL_mlp

3. To train and run the MRI CNN unimodal model:
python src/train --mri_cnn

# Summarize Results:
To summarize results run the following command
python src/merge_csv.py 

This will create two new csv files "Cindex_per_fold.csv" and "prediction_results_per_model" 

# Visualization:
To recreate visualizations used in report run the following command
python src/make_figures.py

