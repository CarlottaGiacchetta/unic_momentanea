# concat.py
import torch
import torch.nn as nn
from itertools import chain
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger()

# ----------------------------
# BLOCK DEFINITIONS
# ----------------------------

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtStyleBlock(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.act2 = nn.GELU()

        self.conv3 = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        # Permute for LayerNorm: from (B,C,H,W) ? (B,H,W,C)
        x = x.permute(0, 2, 3, 1)
        x = self.conv1(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.act1(self.norm1(x))

        x = self.conv2(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.act2(self.norm2(x))

        x = self.conv3(x.permute(0, 3, 1, 2))  # Back to (B,C,H,W)
        return x



class RepresentationAlignmentBlock(nn.Module):
    """
    Implementa la pipeline:
        (B, C, H, W) -? 1x1 conv ? BN ? ReLU  (C ? C/2)
                     -? ConvNeXtBlock        (C/2 ? C/2)
                     -? 1x1 conv ? BN ? ReLU (C/2 ? C/4)
        Restituisce (B, C/4, H, W)
    """
    def __init__(self, input_dim: int):
        super().__init__()
        assert input_dim % 4 == 0, "`input_dim` must be divisible by 4"
        mid_dim   = input_dim // 2
        out_dim   = input_dim // 4

        # channel compression C/2
        self.stage1 = nn.Sequential(
            nn.Conv2d(input_dim, mid_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
        )

        # ConvNeXt-style block on C/2
        self.stage2 = ConvNeXtStyleBlock(mid_dim)

        # final projection C/4
        self.stage3 = nn.Sequential(
            nn.Conv2d(mid_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor (B, C, H, W)
        Returns:
            Tensor (B, C/4, H, W)
        """
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x




# ------------------------------------------------------------
# 1. CBAM: Convolutional Block Attention Module
# ------------------------------------------------------------

class CBAM(nn.Module):

    def __init__(self, n_channels_in, reduction_ratio, kernel_size):
        super(CBAM, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        self.channel_attention = ChannelAttention(n_channels_in, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, f):
        chan_att = self.channel_attention(f)
        fp = chan_att * f
        spat_att = self.spatial_attention(fp)
        fpp = spat_att * fp
        return fpp


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1, "Odd kernel size required"
        self.conv = nn.Conv2d(in_channels = 2, out_channels = 1, kernel_size = kernel_size, padding= int((kernel_size-1)/2))
        # batchnorm

    def forward(self, x):
        max_pool = self.agg_channel(x, "max")
        avg_pool = self.agg_channel(x, "avg")
        pool = torch.cat([max_pool, avg_pool], dim = 1)
        conv = self.conv(pool)
        # batchnorm ????????????????????????????????????????????
        conv = conv.repeat(1,x.size()[1],1,1)
        att = torch.sigmoid(conv)        
        return att

    def agg_channel(self, x, pool = "max"):
        b,c,h,w = x.size()
        x = x.view(b, c, h*w)
        x = x.permute(0,2,1)
        if pool == "max":
            x = F.max_pool1d(x,c)
        elif pool == "avg":
            x = F.avg_pool1d(x,c)
        x = x.permute(0,2,1)
        x = x.view(b,1,h,w)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = int(self.n_channels_in/ float(self.reduction_ratio))

        self.bottleneck = nn.Sequential(
            nn.Linear(self.n_channels_in, self.middle_layer_size),
            nn.ReLU(),
            nn.Linear(self.middle_layer_size, self.n_channels_in)
        )


    def forward(self, x):
        kernel = (x.size()[2], x.size()[3])
        avg_pool = F.avg_pool2d(x, kernel )
        max_pool = F.max_pool2d(x, kernel)

        
        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        max_pool = max_pool.view(max_pool.size()[0], -1)
        

        avg_pool_bck = self.bottleneck(avg_pool)
        max_pool_bck = self.bottleneck(max_pool)

        pool_sum = avg_pool_bck + max_pool_bck

        sig_pool = torch.sigmoid(pool_sum)
        sig_pool = sig_pool.unsqueeze(2).unsqueeze(3)

        out = sig_pool.repeat(1,1,kernel[0], kernel[1])
        return out


# ------------------------------------------------------------
# 2. Attention-based Fusion Block
# ------------------------------------------------------------
class AttentionFusionBlock(nn.Module):
    """
    Implements:  F_T = Conv1×1( CBAM( Concat(F_T_i_align) ) )

    Args
    ----
    input_dim  : #channels after concatenation (C_concat = N * C_align)
    output_dim : desired #channels for the fused teacher feature (C_T)
    reduction  : channel reduction ratio inside CBAM (default 16)
    """
    def __init__(self, input_dim: int, output_dim: int,
                 reduction: int = 16, act: str = "relu"):
        super().__init__()
        self.cbam = CBAM(input_dim, reduction_ratio=reduction, kernel_size=7)
        self.proj = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: concatenated aligned features, shape (B, C_concat, H, W)
        returns: fused feature, shape (B, output_dim, H, W)
        """
        x = self.cbam(x)
        x = self.proj(x)
        return x






class TeacherAggregator(nn.Module):
    """
    Aggrega le feature normalizzate dei teacher con:
      • Representation Alignment Block (RAB)   – per ogni teacher
      • Attention-Based Fusion (ABF)           – per cls & patch token
    """

    def __init__(
        self,
        teacher_dims,          # list[int] – C di ogni teacher (= embed_dim)
        strategy,              # list[str]  – ["rab","abf","mean", ...]
        reduction=16           # per il CBAM interno all'ABF
    ):
        super().__init__()
        self.fused_dim = max(teacher_dims) # canale finale dopo ABF
        
        logger.info('strategy: ',strategy)
        if not strategy:
            self.use_rab = self.use_abf = self.use_mean = False
            logger.info("No strategy specified: using identity fallback.")
        else:
            self.use_rab  = "rab"  in strategy
            self.use_abf  = "abf"  in strategy
            self.use_mean = "mean" in strategy          # fallback
            logger.info(f"Strategy specified: rab {self.use_rab}, abf {self.use_abf}, mean {self.use_mean}")

        # 1) RAB – uno per teacher
        self.rab_cls   = nn.ModuleList()
        self.rab_patch = nn.ModuleList()
        self.rab_out_dim = None                     # C' = C//4
        
        if self.use_rab:
            for C in teacher_dims:
                self.rab_cls.append(RepresentationAlignmentBlock(C))
                self.rab_patch.append(RepresentationAlignmentBlock(C))
            self.rab_out_dim = teacher_dims[0] // 4
        else:
            self.rab_out_dim = teacher_dims[0]

        # 2) ABF – uno per tipo di token
        if self.use_abf:
            in_dim = len(teacher_dims) * self.rab_out_dim
            self.abf_cls   = AttentionFusionBlock(in_dim, self.fused_dim, reduction)
            self.abf_patch = AttentionFusionBlock(in_dim, self.fused_dim, reduction)

    # ------------------------------------------------------------------
    def _align(self, feats, rab_list):
        """Applica il RAB a una lista di feature (una per teacher)."""
        out = []
        for f, rab in zip(feats, rab_list):
            B, L, C = f.shape
            H = W = int(L ** 0.5)
            f = f.view(B, H, W, C).permute(0, 3, 1, 2)   # (B,C,H,W)
            f = rab(f)                                   # (B,C',H,W)
            f = f.permute(0, 2, 3, 1).reshape(B, L, -1)  # (B,L,C')
            out.append(f)
        return out                                      # list[(B,L,C')]

    # ------------------------------------------------------------------
    def forward(self, cls_list, patch_list):
        """
        Args
        ----
        cls_list   : list[Tensor]  (B, L, C)
        patch_list : list[Tensor]  (B, L, C)

        Returns
        -------
        Dict[str, Tensor] con chiavi "cls" & "patch"
        """

        # 1) Alignment
        if self.use_rab:
            cls_aligned   = self._align(cls_list,   self.rab_cls)
            patch_aligned = self._align(patch_list, self.rab_patch)
        else:
            cls_aligned, patch_aligned = cls_list, patch_list

        # 2) Fusione
        if self.use_abf:
            fused = {}

            def _fuse(feats, abf):
                B, N, L, C = feats.shape
                H = W = int(L ** 0.5)
                feats = feats.view(B, N, H, W, C).permute(0, 1, 4, 2, 3)
                feats = torch.cat([f for f in feats.unbind(dim=1)], dim=1)  # (B,NC,H,W)
                fused = abf(feats)                                          # (B,Cf,H,W)
                return fused.permute(0, 2, 3, 1).reshape(B, L, -1)          # (B,L,Cf)

            cls_stack   = torch.stack(cls_aligned,   dim=1)   # (B,N,L,C')
            patch_stack = torch.stack(patch_aligned, dim=1)

            fused["cls"]   = _fuse(cls_stack,   self.abf_cls)
            fused["patch"] = _fuse(patch_stack, self.abf_patch)

        else:   # media semplice
            fused = {
                "cls":   torch.mean(torch.stack(cls_aligned,   dim=0), dim=0),
                "patch": torch.mean(torch.stack(patch_aligned, dim=0), dim=0),
            }

        return fused
