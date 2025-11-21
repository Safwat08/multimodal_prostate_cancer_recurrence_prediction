import torch
import torch.nn as nn
from monai.networks.nets import ResNet
from monai.networks.blocks import Convolution
from monai.networks.layers import Act, Norm

# Clinical data MLP model original
class Clinical_MLP(nn.Module):
    def __init__(self, 
                 input_dim: int = 14,
                 output_dim: int = 1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        x = self.net(x)
        return x
    
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

# MRI image CNN model original
class MRI_CNN(nn.Module):
    def __init__(self,
                 output_dim: int = 1,
                 use_adc: bool = True,
                 use_hbv: bool = True,
                 use_t2w: bool = True):
        super().__init__()
   
        self.use_adc = use_adc
        self.use_hbv = use_hbv
        self.use_t2w = use_t2w

        self.enc1 = Convolution(
            spatial_dims=3,
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            strides=1,
            act=Act.RELU,
            norm=Norm.InstanceNorm,
            dropout=None,
        )
        
        self.enc2 = Convolution(
            spatial_dims=3,
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            strides=2,
            act=Act.RELU,
            norm=Norm.InstanceNorm,
            dropout=None,
        )
        
        self.enc3 = Convolution(
            spatial_dims=3,
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            strides=2,
            act=Act.RELU,
            norm=Norm.InstanceNorm,
            dropout=None,
        )

        self.global_pool = nn.AdaptiveAvgPool3d(1)

        self.dropout = nn.Dropout(0.3)
        
        self.fc = nn.Linear(64, output_dim)
        
    def forward(self, x):
        
        adc = x[:, 0:1, ...] if self.use_adc else torch.zeros_like(x[:, 0:1, ...])
        hbv = x[:, 1:2, ...] if self.use_hbv else torch.zeros_like(x[:, 1:2, ...])
        t2w = x[:, 2:3, ...] if self.use_t2w else torch.zeros_like(x[:, 2:3, ...])

        x_new = torch.cat([adc, hbv, t2w], dim=1)

        x_new = self.enc1(x_new)
        x_new = self.enc2(x_new)
        x_new = self.enc3(x_new)

        x_new = self.global_pool(x_new)
        
        x_new = torch.flatten(x_new, 1)
        
        x_new = self.dropout(x_new)
        
        x_new = self.fc(x_new)

        return x_new    
    
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
            in_channels=3,
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

        x_new = torch.cat([adc, hbv, t2w], dim=1)

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
    
    
    
# Multimodal Model
class Multimodal_CNN_MLP(nn.Module):
    def __init__(
        self,
        mri_output_dim = 32,
        clin_output_dim = 32,
        use_adc: bool = True,
        use_hbv: bool = True,
        use_t2w: bool = True,
        ablate_mri: bool = False,
        ablate_clinical: bool = False):
        
        super().__init__()

        self.ablate_mri = ablate_mri
        self.ablate_clinical = ablate_clinical

        self.mri_encoder = MRI_CNN(output_dim=mri_output_dim, 
                                   use_adc = use_adc, 
                                   use_hbv = use_hbv, 
                                   use_t2w = use_t2w)
        self.clin_encoder = Clinical_MLP(output_dim=clin_output_dim)
        
        self.mri_output_dim = mri_output_dim
        self.clin_output_dim = clin_output_dim
        
        fusion_dim = mri_output_dim + clin_output_dim

        self.risk_predictor = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)   # risk score (log-hazard)
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
                                self.clin_output_dim, 
                                device=device, 
                                dtype=dtype)
        else:
            mri_z = self.mri_encoder(mri_images)

        # ----- Clinical ablation -----
        if self.ablate_clinical:
            clin_z = torch.zeros(size, 
                                 self.mri_output_dim,
                                 device=device, 
                                 dtype=dtype)
        else:
            clin_z = self.clin_encoder(clinical_features)

        # ----- Fusion -----
        z = torch.cat([mri_z, clin_z], dim=1)
        risk = self.risk_predictor(z)
        return risk
    
    
    


# MRI image ResNet model
class MRI_ResNet(nn.Module):
    def __init__(self, input_chan, output_dim):
        super().__init__()
        self.backbone = ResNet(
            spatial_dims=3,
            n_input_channels=input_chan,
            block="basic",
            layers=(1, 1, 1, 1),
            block_inplanes=(16, 32, 64, 128),
            num_classes=output_dim
        )

    def forward(self, x):
        return self.backbone(x).squeeze(-1)
    
    
    
class MRI_Backbone(nn.Module):
    """
    Your basic 3D CNN feature extractor.
    Output: [B, feat_dim] (64)
    """
    def __init__(self, input_chan: int = 1):
        super().__init__()

        self.enc1 = Convolution(
            spatial_dims=3,
            in_channels=input_chan,
            out_channels=16,
            kernel_size=3,
            strides=1,
            act=Act.RELU,
            norm=Norm.BATCH,
            dropout=None,
        )
        
        self.enc2 = Convolution(
            spatial_dims=3,
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            strides=2,
            act=Act.RELU,
            norm=Norm.BATCH,
            dropout=None,
        )
        
        self.enc3 = Convolution(
            spatial_dims=3,
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            strides=2,
            act=Act.RELU,
            norm=Norm.BATCH,
            dropout=None,
        )

        self.global_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.global_pool(x)     
        x = torch.flatten(x, 1)
        return x    

    
class MRI_CNN_Nested(nn.Module):
    """
    Your original early-fusion model:
    - expects input [B, in_channels, D, H, W] (e.g. in_channels=3 for ADC+HBV+T2W)
    - outputs [B] log-risk
    """
    def __init__(self, input_chan: int, output_dim: int = 1):
        super().__init__()
        self.backbone = MRI_Backbone(input_chan=input_chan)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.backbone(x)        # [B, 64]
        x = self.dropout(x)
        x = self.fc(x).squeeze(-1)  # [B]
        return x    

class MRI_CNN_Fusion(nn.Module):
    def __init__(self, output_dim=1,
                 keep_adc=True,
                 keep_hbv=True,
                 keep_t2w=True):
        super().__init__()
            
        self.keep_adc = keep_adc
        self.keep_hbv = keep_hbv
        self.keep_t2w = keep_t2w
        
        # One backbone per modality (1-channel each)
        self.adc_backbone = MRI_Backbone(input_chan=1)
        self.hbv_backbone = MRI_Backbone(input_chan=1)
        self.t2w_backbone = MRI_Backbone(input_chan=1)

        n_modalities = int(keep_adc) + int(keep_hbv) + int(keep_t2w)
        if n_modalities == 0:
            raise ValueError("At least one of keep_adc/keep_hbv/keep_t2w must be True.")

        fused_dim = 64 * n_modalities  # each backbone -> 64-dim feature
        
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        # x: [B, 3, D, H, W]
        feat_list = []

        if self.keep_adc:
            adc = x[:, 0:1, ...]               # [B, 1, D, H, W]
            feat_adc = self.adc_backbone(adc)  # [B, 64]
            feat_list.append(feat_adc)
            
        if self.keep_hbv:
            hbv = x[:, 1:2, ...]
            feat_hbv = self.hbv_backbone(hbv)  # [B, 64]
            feat_list.append(feat_hbv)
            
        if self.keep_t2w:
            t2w = x[:, 2:3, ...]
            feat_t2w = self.t2w_backbone(t2w)  # [B, 64]
            feat_list.append(feat_t2w)
        
        # concatenate along feature dimension
        fused = torch.cat(feat_list, dim=1)  # [B, fused_dim]
        fused = self.dropout(fused)
        risk = self.fc(fused).squeeze(-1)    # [B]

        return risk
    
# Multimodal Model
class Mulitmodal_ResNet_MLP(nn.Module):
    def __init__(
        self,
        clin_input_dim,
        clin_output_dim,
        MRI_input_chan,
        MRI_output_dim,
        ablate: str = "none",
    ):
        super().__init__()

        self.ablate = ablate

        self.mri_encoder = MRI_ResNet(input_chan=MRI_input_chan, output_dim=MRI_output_dim)
        self.clin_encoder = Clinical_MLP(clin_input_dim, clin_output_dim)

        self.mri_latent_dim  = MRI_output_dim
        self.clin_latent_dim = clin_output_dim

        fusion_dim = MRI_output_dim + clin_output_dim

        self.risk_predictor = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)   # risk score (log-hazard)
        )

    def forward(self, mri_images, clinical_features):

        # ----- Get batch size + device -----
        if mri_images is not None:
            size      = mri_images.size(0)
            device = mri_images.device
            dtype  = mri_images.dtype
        else:
            size     = clinical_features.size(0)
            device = clinical_features.device
            dtype  = clinical_features.dtype

        # ----- MRI ablation -----
        if self.ablate == "no_mri":
            mri_z = torch.zeros(size, self.mri_latent_dim, device=device, dtype=dtype)
        else:
            mri_z = self.mri_encoder(mri_images)

        # ----- Clinical ablation -----
        if self.ablate == "no_clinical":
            clin_z = torch.zeros(size, self.clin_latent_dim,
                                 device=device, dtype=dtype)
        else:
            clin_z = self.clin_encoder(clinical_features)

        # ----- Fusion -----
        z = torch.cat([mri_z, clin_z], dim=1)
        risk = self.risk_predictor(z)
        return risk
    
    
    
# Multimodal Model
class Mulitmodal_CNNFusion_MLP(nn.Module):
    def __init__(
        self,
        clin_input_dim,
        clin_output_dim,
        MRI_input_chan,
        MRI_output_dim,
        ablate: str = "none",
        keep_adc=True,
        keep_hbv=True,
        keep_t2w=True
    ):
        super().__init__()

        self.ablate = ablate

        self.mri_encoder = MRI_CNN(input_chan=MRI_input_chan, output_dim=MRI_output_dim)
        self.clin_encoder = Clinical_MLP(clin_input_dim, clin_output_dim)

        self.mri_latent_dim  = MRI_output_dim
        self.clin_latent_dim = clin_output_dim

        fusion_dim = MRI_output_dim + clin_output_dim

        self.risk_predictor = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)   # risk score (log-hazard)
        )

    def forward(self, mri_images, clinical_features):

        # ----- Get batch size + device -----
        if mri_images is not None:
            size      = mri_images.size(0)
            device = mri_images.device
            dtype  = mri_images.dtype
        else:
            size     = clinical_features.size(0)
            device = clinical_features.device
            dtype  = clinical_features.dtype

        # ----- MRI ablation -----
        if self.ablate == "no_mri":
            mri_z = torch.zeros(size, self.mri_latent_dim, device=device, dtype=dtype)
        else:
            mri_z = self.mri_encoder(mri_images)

        # ----- Clinical ablation -----
        if self.ablate == "no_clinical":
            clin_z = torch.zeros(size, self.clin_latent_dim,
                                 device=device, dtype=dtype)
        else:
            clin_z = self.clin_encoder(clinical_features)

        # ----- Fusion -----
        z = torch.cat([mri_z, clin_z], dim=1)
        risk = self.risk_predictor(z)
        return risk