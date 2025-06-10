import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torchmetrics.classification import MultilabelAveragePrecision
from torchgeo.models.scale_mae import scalemae_large_patch16, ScaleMAELarge16_Weights
from teachers.config import CONFIG




class ScaleMAE(pl.LightningModule):

    def __init__(self, args=None):
        super().__init__()

        # Supporta sia Namespace che dict, e fallback se args è None
        args = vars(args) if isinstance(args, argparse.Namespace) else args or {}

        # Parametri con valori di default per test
        self.lr = args.get("lr", 1e-3)
        self.wd = args.get("wd", 1e-4)
        self.image_size = args.get("image_size", 224)
        self.num_classes = args.get("num_classes", 19)
        self.use_weight = args.get("use_weight", False)
        self.finetuning_bands = args.get("finetuning_bands", "rgb")
        self.concat = args.get("concat", False)

        self.save_hyperparameters()  # salva quelli passati

        # Backbone
        weights = ScaleMAELarge16_Weights.FMOW_RGB
        self.backbone = scalemae_large_patch16(weights=weights)
        self.classifier = nn.Linear(self.backbone.embed_dim, self.num_classes)

        # Metriche
        self.metric = MultilabelAveragePrecision(num_labels=self.num_classes)

        # Bande e normalizzazione
        self.bands = CONFIG[self.finetuning_bands]["bands"]
       
        # Class weights
        self.class_weights = torch.ones(self.num_classes)
        if self.use_weight:
            self.class_weights = self._get_class_weights()
        
    
    
       


    def forward(self, x):

        features = self.backbone.forward_features(x) # (B, 197, D)
        cls_token = features[:, 0, :]  # (B, D)
        return self.classifier(cls_token)  # (B, num_classes)

    
    def forward_features(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        
        features = self.backbone.forward_features(x)  # (B, 1+N, D)

        
        
        cls_token = features[:, 0]       # (B, D)
        patch_tokens = features[:, 1:]   # (B, N, D)
        gp = patch_tokens.mean(dim=1)    # global pooled features (B, D)
        
        return {
            "x_norm_clstoken": gp,                   # token globale (può anche essere cls_token)
            "x_norm_patchtokens": patch_tokens,      # patch tokens finali
            "x_prenorm": features,                   # tutti i token
            "x_prenorm_clstoken": cls_token,         # cls token
            "x_prenorm_patchtokens": patch_tokens,   # patch tokens
        }





def scalemae_RGB(checkpoint_path):
    model = ScaleMAE.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model

def scalemae_VEG(checkpoint_path):
    model = ScaleMAE.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model
    
def scalemae_GEO(checkpoint_path):
    model = ScaleMAE.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model