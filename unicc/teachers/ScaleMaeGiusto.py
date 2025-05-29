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
        self.mean = CONFIG[self.finetuning_bands]["mean"]
        self.std = CONFIG[self.finetuning_bands]["std"]

        # Class weights
        self.class_weights = torch.ones(self.num_classes)
        if self.use_weight:
            self.class_weights = self._get_class_weights()
        
    
    def _get_class_weights(self):
        weights = torch.ones(self.num_classes)

        if [4, 5, 6] == self.bands: # DA CAMBIARE !!!
            print('vegetations')
            for i in range(self.num_classes):
                if i not in [4, 5, 8, 10, 13, 14, 15]:
                    weights[i] *= 0.5  
        
        if [7, 10, 11] == self.bands: # DA CAMBIARE !!!
            print('rocks')
            for i in range(self.num_classes):
                if i not in [6,9,12,17]:
                    weights[i] *= 0.5 

        return weights


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {"params": self.backbone.parameters(), "lr": self.lr * 0.1},
            {"params": self.classifier.parameters(), "lr": self.lr}
        ], weight_decay=self.wd)

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=3,
                verbose=True
            ),
            "monitor": "val_map",
            "interval": "epoch",
            "frequency": 1
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}



    def forward(self, x):

        features = self.backbone.forward_features(x) # (B, 197, D)
        cls_token = features[:, 0, :]  # (B, D)
        return self.classifier(cls_token)  # (B, num_classes)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=self.class_weights.to(self.device)) #forse da cambiare anche l'average precispon? 
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        preds = torch.sigmoid(logits)
        self.metric.update(preds, y.int())
        loss = F.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=self.class_weights.to(self.device))
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        val_map = self.metric.compute()
        self.log("val_map", val_map, prog_bar=True)
        self.metric.reset()

    def infer(self, batch, threshold=0.5):
        self.eval()
        with torch.no_grad():
            x = batch["image"]
            x = x[:, self.rgb_band_indices, :, :]
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            logits = self(x)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).int()
        return preds

    def teacher(self):
        model = self.backbone
        model = model.cuda()
        model = model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

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