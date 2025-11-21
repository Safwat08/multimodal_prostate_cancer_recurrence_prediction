import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os 

def main():

    os.makedirs("figures", exist_ok=True)

    cindex = pd.read_csv('Cindices_per_fold.csv')

    selected_models = [
        "multimodal_model_full",
        "multimodal_model_clinical_ablated",
        "multimodal_model_mri_ablated",
        "clinical_mlp_model",
        "clinical_linear_model",
        "mri_cnn_model",
    ]

    # Figure 1
    cindex_fig = cindex[cindex["Model"].isin(selected_models)].copy()

    rename_map = {
        "clinical_linear_model": "Clinical Linear",
        "clinical_mlp_model": "Unimodal Clinical MLP",
        "mri_cnn_model": "Unimodal MRI CNN",
        "multimodal_model_full": "Full Multimodal",
        "multimodal_model_clinical_ablated": "Multimodal (Clinical-Ablated)",
        "multimodal_model_mri_ablated": "Multimodal (MRI-Ablated)",
    }

    ordered_labels = list(rename_map.values())

    cindex_fig["labels"] = cindex_fig["Model"].map(rename_map)

    cindex_fig["labels"] = pd.Categorical(
        cindex_fig["labels"],
        categories=ordered_labels,
        ordered=True
    )

    plt.figure(figsize=(4, 5))

    custom_colors = [
       "#ff7f0e", 
        "#76B041", 
        "#d62728", 
        "#1f77b4", 
        "#9467bd", 
        "#8c564b",
    ]

    sns.boxplot(
        data=cindex_fig,
        x="labels",
        y="C-index",
        order=ordered_labels,  
        palette=custom_colors)   


    plt.title("")
    plt.suptitle("")   
    plt.xlabel("Model")
    plt.ylabel("C-index")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.tight_layout()


    plt.savefig("figures/Figure1_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Figure 2
    selected_models = [
        "clinical_mlp_model",
        "clinical_mlp_model_lr1e_4",
    ]

    cindex_fig = cindex[cindex["Model"].isin(selected_models)].copy()

    rename_map = {
        "clinical_mlp_model": "1e-3",
        "clinical_mlp_model_lr1e_4": "1e-4",
    }

    ordered_labels = list(rename_map.values())

    cindex_fig["labels"] = cindex_fig["Model"].map(rename_map)

    cindex_fig["labels"] = pd.Categorical(
        cindex_fig["labels"],
        categories=ordered_labels,
        ordered=True
    )

    plt.figure(figsize=(2, 5))

    custom_colors = [
        "#76B041",
        "#228B22",
    ]



    sns.boxplot(
        data=cindex_fig,
        x="labels",
        y="C-index",
        order=ordered_labels,
        palette=custom_colors
    )


    plt.title("")
    plt.suptitle("")
    plt.xlabel("Learning Rate")
    plt.ylabel("C-index")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.tight_layout()

    plt.savefig("figures/Figure2_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Figure 3
    selected_models = [
        "multimodal_model_full",
        "multimodal_model_full_diff_lr",
    ]

    cindex_fig = cindex[cindex["Model"].isin(selected_models)].copy()

    rename_map = {
        "multimodal_model_full": "Single LR",
        "multimodal_model_full_diff_lr": "Dual LR",
    }

    ordered_labels = list(rename_map.values())

    cindex_fig["labels"] = cindex_fig["Model"].map(rename_map)

    cindex_fig["labels"] = pd.Categorical(
        cindex_fig["labels"],
        categories=ordered_labels,
        ordered=True
    )

    plt.figure(figsize=(2, 5))

    custom_colors = [
        "#1f77b4", 
        "#1BA6A6", 
    ]


    sns.boxplot(
        data=cindex_fig,
        x="labels",
        y="C-index",
        order=ordered_labels,
        palette=custom_colors 
    )


    plt.title("")
    plt.suptitle("") 
    plt.xlabel("Optimizer")
    plt.ylabel("C-index")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.tight_layout()

    plt.savefig("figures/Figure3_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Figure 4
    selected_models = [
        "mri_cnn_model",
        "mri_cnn_model_precropped",
    ]

    cindex_fig = cindex[cindex["Model"].isin(selected_models)].copy()

    rename_map = {
        "mri_cnn_model": "Uncropped/4channels",
        "mri_cnn_model_precropped": "Precropped/3channels",
    }

    ordered_labels = list(rename_map.values())

    cindex_fig["labels"] = cindex_fig["Model"].map(rename_map)

    cindex_fig["labels"] = pd.Categorical(
        cindex_fig["labels"],
        categories=ordered_labels,
        ordered=True
    )


    plt.figure(figsize=(2.5, 6))

    custom_colors = [
        "#d62728",  
        "#E75480",  
    ]

    sns.boxplot(
        data=cindex_fig,
        x="labels",
        y="C-index",
        order=ordered_labels,   
        palette=custom_colors   
    )

    plt.title("")
    plt.suptitle("")
    plt.xlabel("Preprocessing")
    plt.ylabel("C-index")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.tight_layout()

    plt.savefig("figures/Figure4_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()