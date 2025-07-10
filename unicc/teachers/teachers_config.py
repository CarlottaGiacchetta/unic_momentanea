#from .vit_dino import dino_vitbase
#from .vit_deit3 import deit3_vitbase
#from .vit_dbotft import dbotft_vitbase

from teachers.ScaleMae import scalemae_RGB, scalemae_VEG, scalemae_GEO
from teachers.ViT import ViT, ViT_RGB, ViT_VEG, ViT_GEO, ViT_large

TEACHER_CFG = {
    "scalemae_rgb": {
        "loader": scalemae_RGB,
        "ckpt_path": "/raid/home/rsde/cgiacchetta_unic/unic_momentanea/models/scalemae_RGB/best-checkpoint.ckpt", #"/workspace/models/scalemae_RGB/best-checkpoint.ckpt"
        "ckpt_key": "model",
        "num_features": 1024,
        "resolution": 224,
        "finetuning_bands": "rgb"
        
    },
    "scalemae_veg": {
        "loader": scalemae_VEG,
        "ckpt_path": "/raid/home/rsde/cgiacchetta_unic/unic_momentanea/models/scalemae_VEG/best-checkpoint.ckpt",
        "ckpt_key": "model",
        "num_features": 1024,
        "resolution": 224,
        "finetuning_bands": "veg"
        
    },
    "scalemae_geo": {
        "loader": scalemae_GEO,
        "ckpt_path": "/raid/home/rsde/cgiacchetta_unic/unic_momentanea/models/scalemae_GEO/best-checkpoint.ckpt", 
        "ckpt_key": "model",
        "num_features": 1024,
        "resolution": 224,
        "finetuning_bands": "geo"
        
    },
    "vit_tinyRGB": {
        "loader": ViT_RGB,
        "ckpt_path": "/raid/home/rsde/cgiacchetta_unic/unic_momentanea/models/ViT_tinyRGB/best-checkpoint.ckpt", 
        "ckpt_key": "model",
        "num_features": 192,
        "resolution": 224,
        "finetuning_bands": "rgb"
        
    },
    "vit_tinyVEG": {
        "loader": ViT_VEG,
        "ckpt_path": "/raid/home/rsde/cgiacchetta_unic/unic_momentanea/models/ViT_tinyVEG/best-checkpoint.ckpt", 
        "ckpt_key": "model",
        "num_features": 192,
        "resolution": 224,
        "finetuning_bands": "veg"
        
    },
    "vit_tinyGEO": {
        "loader": ViT_GEO,
        "ckpt_path": "/raid/home/rsde/cgiacchetta_unic/unic_momentanea/models/ViT_tinyGEO/best-checkpoint.ckpt", 
        "ckpt_key": "model",
        "num_features": 192,
        "resolution": 224,
        "finetuning_bands": "geo"
        
    },
    "vit_tiny": {
        "loader": lambda ckpt_path: ViT(ckpt_path, in_chans=12, finetuning_bands="all"),
        "ckpt_path": "/raid/home/rsde/cgiacchetta_unic/unic_momentanea/models/ViT_tiny/best-checkpoint.ckpt", 
        "ckpt_key": "model",
        "num_features": 192,
        "resolution": 224,
        "finetuning_bands": "all"        
    },
    "vit_large": {
        "loader": ViT_large,
        "ckpt_path": "/raid/home/rsde/cgiacchetta_unic/unic_momentanea/models/ViT_large/best-checkpoint.ckpt", 
        "ckpt_key": "model",
        "num_features": 1024,
        "resolution": 224,
        "finetuning_bands": "nove"        
    },

}