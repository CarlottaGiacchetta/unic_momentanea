import logging
from collections import defaultdict
from typing import List, Dict

import torch

from utils import standard_normalize
from .teachers_config import TEACHER_CFG
from .builder import build_teachers
from .concat import RepresentationAlignmentBlock, AttentionFusionBlock


logger = logging.getLogger()


@torch.no_grad()
def get_teacher_output(
    image: torch.Tensor,
    teachers: Dict[str, torch.nn.Module],
    teacher_ft_stats: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
    teacher_ft_stat_ema_momentum: float = 0.0,
    strategy: List[str] = None,
    aggregation_parameter: Dict[str, float] = None,
    aggregator=None,
) -> Dict[str, Dict[str, torch.Tensor]]:

    teacher_output = defaultdict(dict)
    cls_list, patch_list = [], []

    use_mean = strategy and "mean" in strategy
    use_rab = strategy and "rab" in strategy
    use_abf = strategy and "abf" in strategy

    rab_cls = None
    rab_patch = None
    abf_cls = None
    abf_patch = None

    rab_out_dim = None  # dimensione delle feature dopo RAB (C // 4)

    for tname in teachers.keys():
        image_copy = image
        if tname == 'scalemae_veg':
            image_copy = image_copy[:, [4, 5, 6], :, :]
            mean = torch.tensor([942.7477, 1769.8486, 2049.4758], device=image.device).view(1, 3, 1, 1)
            std = torch.tensor([727.5784, 1087.4288, 1261.4303], device=image.device).view(1, 3, 1, 1)
        elif tname == 'scalemae_rgb':
            image_copy = image_copy[:, [4, 3, 2], :, :]
            mean = torch.tensor([942.7477, 588.4096, 614.0557], device=image.device).view(1, 3, 1, 1)
            std = torch.tensor([727.5784, 684.5688, 603.2968], device=image.device).view(1, 3, 1, 1)
        elif tname == 'scalemae_geo':
            image_copy = image_copy[:, [7, 10, 11], :, :]
            mean = torch.tensor([2193.2920, 1568.2118, 997.7151], device=image.device).view(1, 3, 1, 1)
            std = torch.tensor([1369.3717, 1063.9198, 806.8846], device=image.device).view(1, 3, 1, 1)

        image_copy = (image_copy - mean) / std
        tout_dict = teachers[tname].forward_features(image_copy)

        for ttype in ["cls", "patch"]:
            key = f"x_norm_{ttype}{'token' if ttype == 'cls' else 'tokens'}"
            tout = tout_dict[key]  # (B, L, C)
            tout = standard_normalize(
                tout,
                mean_ema=teacher_ft_stats[tname][ttype]["mean"],
                std_ema=teacher_ft_stats[tname][ttype]["std"],
                ema_momentum=teacher_ft_stat_ema_momentum,
            )
            if tout.ndim == 2:
                tout = tout.unsqueeze(1)  # (B, C) ? (B, 1, C)
            B, L, C = tout.shape

            if use_mean or use_abf:
                if ttype == "cls":
                    cls_list.append(tout)
                else:
                    patch_list.append(tout)

            teacher_output[tname][ttype] = tout

    # Fusione finale
    merged_output = {}

    if use_mean:
        merged_output["mean"] = {
            "cls": torch.mean(torch.stack(cls_list, dim=0), dim=0).squeeze(1),
            "patch": torch.mean(torch.stack(patch_list, dim=0), dim=0)
        }

    if use_abf:
        assert aggregator is not None, "Pass `aggregator` if using ABF strategy"
        abf = aggregator(cls_list, patch_list)
        merged_output["abf"] = {
            "cls": abf["cls"].squeeze(1),
            "patch": abf["patch"]
        }

    if merged_output:
        teacher_output = {"mergedFeatures": {}}
        # Se entrambe presenti, si può decidere se restituirle separate o fonderle ulteriormente
        if "mean" in merged_output and "abf" in merged_output:
            alpha = aggregation_parameter.get("alpha", 0.5)
            beta = aggregation_parameter.get("beta", 0.5)
            assert abs(alpha + beta - 1.0) < 1e-5, f"alpha + beta must be 1.0, got {alpha + beta}"

            teacher_output = {"mergedFeatures": {
                "cls": alpha * merged_output["mean"]["cls"] + beta * merged_output["abf"]["cls"],
                "patch": alpha * merged_output["mean"]["patch"] + beta * merged_output["abf"]["patch"]
            }}
        elif "mean" in merged_output:
            teacher_output = {"mergedFeatures": merged_output["mean"]}
        elif "abf" in merged_output:
            teacher_output = {"mergedFeatures": merged_output["abf"]}

    return teacher_output
