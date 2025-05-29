import torch.distributed as dist
import os
import logging
from collections import OrderedDict
from typing import List, Dict, Union

import torch

from .teachers_config import TEACHER_CFG

import logging


logger = logging.getLogger()


def build_teachers(teacher_names: List[str]) -> Union[Dict[str, torch.nn.Module], Dict[str, Dict[str, Dict[str, torch.Tensor]]]]:
    teachers = OrderedDict()
    teacher_ft_stats = OrderedDict()

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0

    for tname in teacher_names:
        tname = tname.strip()
        logger.info(f"[rank{rank}] Loading teacher '{tname}'")

        # Scarica solo su rank 0
        if rank == 0:
            _ = _build_teacher(tname)

        # Sincronizza tutti i processi prima di procedere (barriera)
        if is_dist:
            dist.barrier()

        # Ora carica da file locale (tutti i rank)
        model = _build_teacher(tname)
        teachers[tname] = model

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

    return teachers, teacher_ft_stats


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