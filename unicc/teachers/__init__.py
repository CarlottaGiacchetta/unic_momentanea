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
) -> Dict[str, Dict[str, torch.Tensor]]:

    teacher_output = defaultdict(dict)
    cls_list, patch_list = [], []

    use_mean = strategy and "mean" in strategy
    use_rab = strategy and "rab" in strategy
    use_abf = strategy and "abf" in strategy

    rab_cls = RepresentationAlignmentBlock(input_dim=768).to(image.device) if use_rab else None
    rab_patch = RepresentationAlignmentBlock(input_dim=768).to(image.device) if use_rab else None
    abf_cls = AttentionFusionBlock(input_dim=3 * 192, output_dim=768).to(image.device) if use_abf else None
    abf_patch = AttentionFusionBlock(input_dim=3 * 192, output_dim=768).to(image.device) if use_abf else None

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

            if use_rab:
                # Converti da (B, L, C) ? (B, C, H, W) per RAB
                B, L, C = tout.shape
                H = W = int(L**0.5)
                tout = tout.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
                tout = rab_cls(tout) if ttype == "cls" else rab_patch(tout)  # (B, C', H, W)
                tout = tout.permute(0, 2, 3, 1).reshape(B, L, -1)  # (B, L, C')

            if use_mean or use_abf:
                if ttype == "cls":
                    cls_list.append(tout)
                else:
                    patch_list.append(tout)

            teacher_output[tname][ttype] = tout

    # Fusione finale
    if use_mean or use_abf:
        teacher_output = {"mergedFeatures": {}}

        if use_abf:
            def fuse(feat_list, block):
                feats = torch.stack(feat_list, dim=1)  # (B, N, L, C)
                B, N, L, C = feats.shape
                H = W = int(L**0.5)
                feats = feats.view(B, N, H, W, C).permute(0, 1, 4, 2, 3)  # (B, N, C, H, W)
                feats = torch.cat([f for f in feats.unbind(dim=1)], dim=1)  # (B, N*C, H, W)
                fused = block(feats)  # (B, C_out, H, W)
                return fused.permute(0, 2, 3, 1).reshape(B, L, -1)  # (B, L, C_out)

            teacher_output["mergedFeatures"] = {
                "cls": fuse(cls_list, abf_cls),
                "patch": fuse(patch_list, abf_patch)
            }

        elif use_mean:
            teacher_output["mergedFeatures"] = {
                "cls": torch.mean(torch.stack(cls_list, dim=0), dim=0),
                "patch": torch.mean(torch.stack(patch_list, dim=0), dim=0)
            }

    return teacher_output

