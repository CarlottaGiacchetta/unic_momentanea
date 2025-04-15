#from .vit_dino import dino_vitbase
#from .vit_deit3 import deit3_vitbase
#from .vit_dbotft import dbotft_vitbase

from teachers.ScaleMae import scalemae_RGB, scalemae_VEG

TEACHER_CFG = {
    "scalemae_rgb": {
        "loader": scalemae_RGB,
        "ckpt_path": "/raid/home/rsde/cgiacchetta_unic/unic_momentanea/models/best-checkpoint_rgb.ckpt",
        "ckpt_key": "model",
        "num_features": 1024,
        "resolution": 224,
    },
    "scalemae_veg": {
        "loader": scalemae_VEG,
        "ckpt_path": "/raid/home/rsde/cgiacchetta_unic/unic_momentanea/models/best-checkpoint_veg.ckpt",
        "ckpt_key": "model",
        "num_features": 1024,
        "resolution": 224,
        
    },

}