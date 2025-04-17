import argparse
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from Dataset import carica_dati
from dinov2.models.vision_transformer import get_model

logger = logging.getLogger()

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MultilabelAveragePrecision
from dinov2.models.vision_transformer import get_model

class ViTFinetune(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.lr = args.lr
        self.wd = args.wd
        self.num_classes = args.num_classes

        # Encoder ViT
        self.backbone = get_model(
            arch=args.arch,
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,
            img_size=args.image_size,
            in_chans=12
        )

        self.classifier = nn.Linear(self.backbone.embed_dim, self.num_classes)
        self.metric = MultilabelAveragePrecision(num_labels=self.num_classes)




    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x, _ = self.backbone.prepare_tokens_with_masks(x)
        for blk in self.backbone.blocks[0]:
            x = blk(x)
        cls_token = x[:, 0, :]
        return self.classifier(cls_token)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        preds = torch.sigmoid(logits)
        self.metric.update(preds, y.int())
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        val_map = self.metric.compute()
        self.log("val_map", val_map, prog_bar=True)
        self.metric.reset()

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


import argparse
import os
import logging
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from Dataset import carica_dati

logger = logging.getLogger()

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--arch", type=str, default="vit_base")
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--drop_path_rate", type=float, default=0.1)
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--bands", type=str, default="s2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--data_dir", type=str, default="D:/tesi_carlotta/data")
    parser.add_argument("--checkpoint_dir", type=str, default="D:/tesi_carlotta/checkpoints/ViT")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--wd", type=float, default=5e-3)
    parser.add_argument("--output_dir", type=str, default="teachers/output")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--transform", type=bool, default=True)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    return args

def main(args):

    train, train_loader, validation, validation_loader = carica_dati(args)

    print('\n--dati caricati')
    print(train)
    print(validation)

    model = ViTFinetune(args)

    logger = TensorBoardLogger("tb_logs", name="ViTFinetune")

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        monitor="val_map",
        mode="max",
        save_top_k=1,
        filename="best-checkpoint",
        verbose=True
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_map",
        patience=7,
        mode="max",
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)

if __name__ == "__main__":
    args = get_args()
    main(args)
