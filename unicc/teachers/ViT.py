import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import copy


import argparse
from teachers.config import CONFIG
from teachers.vision_transformer import get_model


from torchmetrics.classification import MultilabelAveragePrecision

class ViT(pl.LightningModule):
    def __init__(self, checkpoint_path, in_chans, finetuning_bands, use_ema: bool = False, ema_momentum: float = 0.999):
        super().__init__()
        # Supporta sia Namespace che dict, e fallback se args è None
        self.save_hyperparameters()  # logs hyperparameters for reproducibility
        
        self.bands = CONFIG[finetuning_bands]["bands"]
        self.mean = CONFIG[finetuning_bands]["mean"]
        self.std = CONFIG[finetuning_bands]["std"]
    
        
        self.encoder = get_model(
            arch =  "vit_large",
            patch_size = 16,
            drop_path_rate=0.0,
            img_size=224,
            in_chans=in_chans  # adjust based on your data (e.g., 3 for RGB, 12 for multispectral)
        )
        
        if checkpoint_path:
            state = torch.load(checkpoint_path)

            # Cerca i pesi nel posto giusto
            if isinstance(state, dict):
                if "model" in state:
                    state_dict = state["model"]
                elif "state_dict" in state:
                    state_dict = state["state_dict"]
                else:
                    state_dict = state
            else:
                state_dict = state  # fallback
            
            # Filtra solo i pesi del ViT encoder
            encoder_state_dict = {
                k.replace("module.encoder.", ""): v
                for k, v in state_dict.items()
                if k.startswith("module.encoder.")
            }

            # Caricamento con tolleranza
            missing, unexpected = self.encoder.load_state_dict(encoder_state_dict, strict=False)
            print(f"[INFO] Loaded encoder from {checkpoint_path}")
            print(f"[INFO] Missing keys: {missing}")
            print(f"[INFO] Unexpected keys: {unexpected}")


        


        for param in self.encoder.parameters():
            param.requires_grad = False
        self.eval()

        self.image_size = 224

        # --- EMA support ----------------------------------------------------
        self._is_ema = use_ema
        if self._is_ema:
            self._momentum = ema_momentum
            # crea un nuovo modello identico e carica pesi
            self._ema = get_model(
                arch="vit_large",
                patch_size=16,
                drop_path_rate=0.0,
                img_size=224,
                in_chans=in_chans
            ).eval()
            self._ema.load_state_dict(self.encoder.state_dict())
            for p in self._ema.parameters():
                p.requires_grad = False
    
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def update_ema(self, student_encoder, momentum: float | None = None):
        """Aggiorna i pesi _ema come EMA del modello passato (di solito lo student)."""
        if not self._is_ema:
            return
        m = self._momentum if momentum is None else momentum
        for p_ema, p_src in zip(self._ema.parameters(), student_encoder.parameters()):
            p_ema.data.mul_(m).add_(p_src.data, alpha=1. - m)


    def forward(self, x):
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False) # x: (B, 3, 120, 120) → (B, 3, 224, 224)
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        features = self.encoder.forward_features(x)
        cls_token = features["x_norm_clstoken"] 
        return self.classifier(cls_token)
    

    
    
    def forward_features(self, x):

      x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
      x = (x - self.mean.to(x.device)) / self.std.to(x.device)
      
  
      # Forward raw features (prenorm, no masking, no register token)
      features = self.encoder.patch_embed(x)
      B = x.shape[0]
      cls_tokens = self.encoder.cls_token.expand(B, -1, -1)
      x = torch.cat((cls_tokens, features), dim=1)
      x = x + self.encoder.interpolate_pos_encoding(x, self.image_size, self.image_size)
  
      for blk in self.encoder.blocks:
          x = blk(x)
  
      x_norm = self.encoder.norm(x)
  
      return {
          "x_norm_clstoken": x_norm[:, 0],              # [B, D]
          "x_norm_patchtokens": x_norm[:, 1:],          # [B, N, D]
          "x_prenorm": x,                               # [B, 1+N, D]
          "x_prenorm_clstoken": x[:, 0],
          "x_prenorm_patchtokens": x[:, 1:]
      }



def vit_tiny(checkpoint_path):
    model = ViT(checkpoint_path, in_chans=12, finetuning_bands='all')

    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model

def ViT_large(checkpoint_path):
    model = ViT(checkpoint_path, in_chans=9, finetuning_bands='nove', use_ema = True)

    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model


def ViT_RGB(checkpoint_path):
    
    model = ViT(checkpoint_path, in_chans=3, finetuning_bands='rgb')
    

    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model
    
def ViT_VEG(checkpoint_path):
    model = ViT(checkpoint_path, in_chans=3, finetuning_bands='veg')

    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model
    
def ViT_GEO(checkpoint_path):
    model = ViT(checkpoint_path, in_chans=3, finetuning_bands='geo')

    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model