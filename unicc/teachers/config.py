#from .vit_dino import dino_vitbase
#from .vit_deit3 import deit3_vitbase
#from .vit_dbotft import dbotft_vitbase

from teachers.ScaleMae import scalemae_RGB, scalemae_VEG

TEACHER_CFG = {
    "scalemae_rgb": {
        "loader": scalemae_RGB,
        "ckpt_path": "models/best-checkpoint-v2.ckpt",
        "ckpt_key": "model",
        "num_features": 1024,
        "resolution": 224,
    },
    "scalemae_veg": {
        "loader": scalemae_VEG,
        "ckpt_path": "models/best-checkpoint-v1.ckpt",
        "ckpt_key": "model",
        "num_features": 1024,
        "resolution": 224,
        
    },

}