import logging
from collections import defaultdict
from typing import List, Dict

import torch

from utils import standard_normalize
from .teachers_config import TEACHER_CFG
from .builder import build_teachers


logger = logging.getLogger()


@torch.no_grad()
def get_teacher_output(
    image: torch.Tensor,
    teachers: List[str],
    teacher_ft_stats: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
    teacher_ft_stat_ema_momentum: float = 0.0,
    concat: bool = False
) -> Dict[str, Dict[str, torch.Tensor]]:

    teacher_output = defaultdict(dict)

    cls_list = []
    patch_list = []

    for tname in teachers.keys():
        image_copy = image
        if tname == 'scalemae_veg':
            image_copy = image_copy[:, [4, 5, 6], :, :] 
            mean = torch.tensor([942.7476806640625, 1769.8486328125, 2049.475830078125], device=image_copy.device).view(1, 3, 1, 1)
            std = torch.tensor([727.5784301757812,   1087.4288330078125, 1261.4302978515625], device=image_copy.device).view(1, 3, 1, 1)
            image_copy = (image_copy - mean) / std

        elif tname == 'scalemae_rgb':
            image_copy = image_copy[:, [4, 3, 2], :, :] 
            mean = torch.tensor([942.7476806640625, 588.4096069335938, 614.0556640625], device=image_copy.device).view(1, 3, 1, 1)
            std = torch.tensor([727.5784301757812,   684.56884765625, 603.2968139648438], device=image_copy.device).view(1, 3, 1, 1)
            image_copy = (image_copy - mean) / std
            
        elif tname == 'scalemae_geo':
            image_copy = image_copy[:, [7, 10, 11], :, :] 
            mean = torch.tensor([2193.2919921875, 1568.2117919921875, 997.715087890625], device=image_copy.device).view(1, 3, 1, 1)
            std = torch.tensor([1369.3717041015625, 1063.9197998046875, 806.8846435546875], device=image_copy.device).view(1, 3, 1, 1)
            image_copy = (image_copy - mean) / std
        
        tout_dict = teachers[tname].forward_features(image_copy) #forward_features(image) per ottenere l'output del ViT (prima della testa lineare)
        

        for ttype in ["cls", "patch"]:
            key = "x_norm_{}{}".format(ttype, "token" if ttype == "cls" else "tokens")
            tout = tout_dict[key]
            tout = standard_normalize(
                tout,
                mean_ema=teacher_ft_stats[tname][ttype]["mean"],
                std_ema=teacher_ft_stats[tname][ttype]["std"],
                ema_momentum=teacher_ft_stat_ema_momentum,
            )
            
            print(concat)
            if concat == True:
                # Salva per il calcolo della media
                if ttype == "cls":
                    cls_list.append(tout)
                elif ttype == "patch":
                    patch_list.append(tout)

            teacher_output[tname][ttype] = tout

    # Calcola media tra tutti i teacher per cls e patch
    if concat == True:
        teacher_output = {"mergedFeatures" : {}}
        teacher_output["mergedFeatures"] = {
            "cls": torch.mean(torch.stack(cls_list, dim=0), dim=0),
            "patch": torch.mean(torch.stack(patch_list, dim=0), dim=0)
        }
    

          
    return teacher_output
