import os
import logging
from collections import OrderedDict
from typing import List, Dict, Union

import torch

from .teachers_config import TEACHER_CFG


logger = logging.getLogger()


def build_teachers(
    teacher_names: List[str],
) -> Union[Dict[str, torch.nn.Module], Dict[str, Dict[str, Dict[str, torch.Tensor]]]]:
    teachers = OrderedDict()
    teacher_ft_stats = OrderedDict()

    for tname in teacher_names:
        tname = tname.strip()
        logger.info("Loading teacher '{}'".format(tname))
        model = _build_teacher(tname)
        teachers[tname] = model

        # buffers for teacher feature statistics
        ft_dim = TEACHER_CFG[tname]["num_features"]
        teacher_ft_stats[tname] = {
            "cls": {
                "mean": torch.zeros(1, ft_dim).cuda(),
                "std": torch.ones(1, ft_dim).cuda(),
            },
            "patch": {
                "mean": torch.zeros(1, 1, ft_dim).cuda(),
                "std": torch.ones(1, 1, ft_dim).cuda(),
            },
        }
    
    teacher_dims = [TEACHER_CFG[t]["num_features"] for t in teacher_names]

    return teachers, teacher_ft_stats, teacher_dims


def _build_teacher(name):
    # name is expected to be in the following format:
    #  dino_vitbase_16
    #  <model_name>_<arch>_<patch_size>
    if name not in TEACHER_CFG.keys():
        raise ValueError(
            "Unsupported teacher name: {} (supported ones: {})".format(
                name, TEACHER_CFG.keys()
            )
        )

    ckpt_path = TEACHER_CFG[name]["ckpt_path"]
    ckpt_key = TEACHER_CFG[name]["ckpt_key"]

    if not os.path.isfile(ckpt_path):
        raise ValueError("Invalid teacher model path: {}".format(ckpt_path))


    model = TEACHER_CFG[name]["loader"](ckpt_path)

    for param in model.parameters():
        param.requires_grad = False

    return model
