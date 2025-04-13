import logging
from collections import defaultdict
from typing import List, Dict

import torch

from utils import standard_normalize
from .config import TEACHER_CFG
from .builder import build_teachers


logger = logging.getLogger()


@torch.no_grad()
def get_teacher_output(
    image: torch.Tensor,
    teachers: List[str],
    teacher_ft_stats: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
    teacher_ft_stat_ema_momentum: float = 0.0,
) -> Dict[str, Dict[str, torch.Tensor]]:

    teacher_output = defaultdict(dict)

    for tname in teachers.keys():
        if tname == 'scalemae_veg':
            print('sono nel teacher ', tname)
            image = image[:, [4, 5, 6], :, :] #FUNZIONA SOLO PER RGB
            mean = torch.tensor([942.7476806640625, 1769.8486328125, 2049.475830078125], device=image.device).view(1, 3, 1, 1)
            std = torch.tensor([727.5784301757812,   1087.4288330078125, 1261.4302978515625], device=image.device).view(1, 3, 1, 1)
            image = (image - mean) / std

        elif tname == 'scalemae_rgb':
            print('sono nel teacher ', tname)
            image = image[:, [4, 3, 2], :, :] #FUNZIONA SOLO PER RGB
            mean = torch.tensor([942.7476806640625, 588.4096069335938, 614.0556640625], device=image.device).view(1, 3, 1, 1)
            std = torch.tensor([727.5784301757812,   684.56884765625, 603.2968139648438], device=image.device).view(1, 3, 1, 1)
            image = (image - mean) / std
        
        tout_dict = teachers[tname].forward_features(image) #forward_features(image) per ottenere l'output del ViT (prima della testa lineare)
        print('\n\n\ntout_dict')
        print(tout_dict)


        for ttype in ["cls", "patch"]:
            key = "x_norm_{}{}".format(ttype, "token" if ttype == "cls" else "tokens")
            tout = tout_dict[key]
            tout = standard_normalize(
                tout,
                mean_ema=teacher_ft_stats[tname][ttype]["mean"],
                std_ema=teacher_ft_stats[tname][ttype]["std"],
                ema_momentum=teacher_ft_stat_ema_momentum,
            )


            teacher_output[tname][ttype] = tout
          
    return teacher_output
