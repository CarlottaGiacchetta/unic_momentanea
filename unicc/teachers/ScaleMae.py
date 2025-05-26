import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torchmetrics.classification import MultilabelAveragePrecision
from torchgeo.models.scale_mae import scalemae_large_patch16, ScaleMAELarge16_Weights
from teachers.config import CONFIG




class ScaleMAETeacher(nn.Module):
    """
    Compatibile con i teacher ViT (dino/deit/dbot):
    - stesse chiavi nel dict
    - x_norm_clstoken = LayerNorm(gap(patch_tokens))
    - patch token & cls token *pre-norm* negli altri campi
    """

    def __init__(self, finetuning_bands: str, ckpt: str | None = None):
        super().__init__()


        # ---------- backbone ----------
        weights = ScaleMAELarge16_Weights.FMOW_RGB
        self.backbone = scalemae_large_patch16(weights=weights)
        # *patch* backbone per distillazione
        self.backbone.fc_norm = nn.LayerNorm(self.backbone.embed_dim)
        if hasattr(self.backbone, "norm"):
            delattr(self.backbone, "norm")


        # freeze
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.eval()

    


    # --------------------------------------------------------
    @torch.no_grad()
    def forward_features(self, x: torch.Tensor, resize=True):
        if resize:
            x = F.interpolate(x, size=(224, 224),
                              mode="bilinear", align_corners=False)

        # ---------- stesso flow del VisionTransformer teacher ----------
        B = x.size(0)
        x = self.backbone.patch_embed(x)

        cls_tok = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tok, x), dim=1)
        x = self.backbone.pos_drop(x + self.backbone.pos_embed)

        for blk in self.backbone.blocks:
            x = blk(x)

        gp = x[:, 1:, :].mean(dim=1)
        gp = self.backbone.fc_norm(gp)

        return {
            "x_norm_clstoken": gp,
            "x_norm_patchtokens": x[:, 1:],  # pre-norm patch
            "x_prenorm": x,
            "x_prenorm_clstoken": x[:, 0],
            "x_prenorm_patchtokens": x[:, 1:],
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