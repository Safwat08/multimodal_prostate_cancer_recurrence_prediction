import torch
import torch.nn as nn
from monai.networks.blocks import Convolution
from monai.networks.layers import Act, Norm
    
# Clinical data MLP model backbone
class Clinical_MLP_Backbone(nn.Module):
    def __init__(self, 
                 input_dim: int = 14,
                 feature_dim: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x) 
        return (x)

# Clinical data MLP model with backbone
class Clinical_MLP_Model(nn.Module):
    def __init__(self,
                 input_dim: int = 14,
                 feature_dim: int = 32,
                 dropout: float = 0.1,
                 output_dim: int = 1):
        super().__init__()
        
        self.backbone = Clinical_MLP_Backbone(input_dim = input_dim,
                                            feature_dim = feature_dim,
                                            dropout = dropout)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.net(x) 
        return x
    
# MRI image CNN model backbone
class MRI_CNN_Backbone(nn.Module):
    def __init__(self,
                 feature_dim: int = 16,
                 dropout: float = 0.3,
                 use_adc: bool = True,
                 use_hbv: bool = True,
                 use_t2w: bool = True):
        super().__init__()
   
        self.use_adc = use_adc
        self.use_hbv = use_hbv
        self.use_t2w = use_t2w

        self.enc1 = Convolution(
            spatial_dims=3,
            in_channels=4,
            out_channels=feature_dim,
            kernel_size=3,
            strides=1,
            act=Act.RELU,
            norm=Norm.INSTANCE,
            dropout=None,
        )
        
        self.enc2 = Convolution(
            spatial_dims=3,
            in_channels=feature_dim,
            out_channels=feature_dim*2,
            kernel_size=3,
            strides=2,
            act=Act.RELU,
            norm=Norm.INSTANCE,
            dropout=None,
        )
        
        self.enc3 = Convolution(
            spatial_dims=3,
            in_channels=feature_dim*2,
            out_channels=feature_dim*4,
            kernel_size=3,
            strides=2,
            act=Act.RELU,
            norm=Norm.INSTANCE,
            dropout=None,
        )

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        adc = x[:, 0:1, ...] if self.use_adc else torch.zeros_like(x[:, 0:1, ...])
        hbv = x[:, 1:2, ...] if self.use_hbv else torch.zeros_like(x[:, 1:2, ...])
        t2w = x[:, 2:3, ...] if self.use_t2w else torch.zeros_like(x[:, 2:3, ...])
        mask = x[:, 3:4, ...]

        x_new = torch.cat([adc, hbv, t2w, mask], dim=1)

        x_new = self.enc1(x_new)
        x_new = self.enc2(x_new)
        x_new = self.enc3(x_new)

        x_new = self.global_pool(x_new)
        x_new   = torch.flatten(x_new, 1)
        x_new = self.dropout(x_new)

        return x_new        

# MRI image CNN model with backbone
class MRI_CNN_Model(nn.Module):
    def __init__(self,
                 output_dim: int = 1,
                 feature_dim: int = 16,
                 dropout: float = 0.3,
                 use_adc: bool = True,
                 use_hbv: bool = True,
                 use_t2w: bool = True):
        super().__init__()
        
        self.backbone = MRI_CNN_Backbone(feature_dim = feature_dim,
                                         dropout = dropout,
                                         use_adc = use_adc,
                                         use_hbv = use_hbv,
                                         use_t2w = use_t2w)


        
        self.fc = nn.Linear(feature_dim*4, output_dim)
        
    def forward(self, x):
        
        x = self.backbone(x)
        
        x = self.fc(x)

        return x    
        
# Multimodal Model with MRI_CNN and Clinical_MLP backbones
class Multimodal_Model(nn.Module):
    def __init__(
        self,
        mri_feature_dim: int = 16,
        mri_dropout:float = 0.3,
        clinical_input_dim: int = 14,
        clinical_feature_dim: int = 32,
        clinical_dropout: float = 0.1,
        predictor_feature_dim: int = 32,
        predictor_dropout: float = 0.3,
        use_adc: bool = True,
        use_hbv: bool = True,
        use_t2w: bool = True,
        ablate_mri: bool = False,
        ablate_clinical: bool = False):
        
        super().__init__()

        self.ablate_mri = ablate_mri
        self.ablate_clinical = ablate_clinical

        self.mri_encoder = MRI_CNN_Backbone(feature_dim = mri_feature_dim,
                                            dropout = mri_dropout,
                                   use_adc = use_adc, 
                                   use_hbv = use_hbv, 
                                   use_t2w = use_t2w)
        
        self.clin_encoder = Clinical_MLP_Backbone(input_dim = clinical_input_dim,
                                                  feature_dim = clinical_feature_dim,
                                                 dropout = clinical_dropout)
        
        self.mri_feature_dim = mri_feature_dim
        self.clinical_feature_dim = clinical_feature_dim
        
        fusion_dim = mri_feature_dim*4 + clinical_feature_dim

        self.risk_predictor = nn.Sequential(
            nn.Linear(fusion_dim, predictor_feature_dim),
            nn.ReLU(),
            nn.Dropout(predictor_dropout),
            nn.Linear(predictor_feature_dim, 1)
        )

    def forward(self, mri_images, clinical_features):
        if mri_images is not None:
            size   = mri_images.size(0)
            device = mri_images.device
            dtype  = mri_images.dtype
        elif clinical_features is not None:
            size   = clinical_features.size(0)
            device = clinical_features.device
            dtype  = clinical_features.dtype
        else:
            raise ValueError("At least one of mri_images or clinical_features must be provided.")
            
        # ----- MRI ablation -----
        if self.ablate_mri:
            mri_z = torch.zeros(size, 
                                self.mri_feature_dim*4, 
                                device=device, 
                                dtype=dtype)
        else:
            mri_z = self.mri_encoder(mri_images)

        # ----- Clinical ablation -----
        if self.ablate_clinical:
            clin_z = torch.zeros(size, 
                                 self.clinical_feature_dim,
                                 device=device, 
                                 dtype=dtype)
        else:
            clin_z = self.clin_encoder(clinical_features)

        # ----- Fusion -----
        z = torch.cat([mri_z, clin_z], dim=1)
        risk = self.risk_predictor(z)
        return risk
    
    
    
