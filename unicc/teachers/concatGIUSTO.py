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


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block (fedelmente alla figura):
      • Depthwise conv 7x7, groups=C
      • LayerNorm
      • PW-Conv 1x1 -> 4C
      • GELU
      • PW-Conv 1x1 -> C
      • Residual
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dw_conv   = nn.Conv2d(dim, dim, kernel_size=7,
                                   padding=3, groups=dim)
        self.ln        = nn.LayerNorm(dim, eps=eps)          # LN su canali
        self.pw_conv1  = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act       = nn.GELU()
        self.pw_conv2  = nn.Conv2d(4 * dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print('sono nel forward della convnext')
        print(x.shape)
        shortcut = x # (B,C,H,W)
        x = self.dw_conv(x) # depthwise 7×7

        # LayerNorm richiede (B,H,W,C) ? permuta, applica, poi ripermuta
        x = x.permute(0, 2, 3, 1) # (B,H,W,C)
        x = self.ln(x) #LayerNorm
        x = x.permute(0, 3, 1, 2) # (B,C,H,W)

        x = self.pw_conv1(x) # conv 1×1, 4C
        x = self.act(x) #Gelu activation
        x = self.pw_conv2(x) # con 1×1, C

        return x + shortcut # residual


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
        self.stage2 = ConvNeXtBlock(mid_dim)

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
'''class CBAM(nn.Module):
    """
    Standard CBAM implementation:
      • Channel Attention (MLP on global avg & max-pool features)
      • Spatial  Attention (7x7 conv over concatenated avg/max maps)
    """
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        # Channel attention --------------------------------------------------
        reduced = max(in_channels // reduction, 1)
        
        #mlp dovrebbe avere un unico hidden layer
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, reduced, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, in_channels, kernel_size=1, bias=False)
        )
        # Spatial attention --------------------------------------------------
        self.spatial = nn.Conv2d(2, 1, kernel_size=kernel_size,
                                 padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---- Channel attention ----
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)            # (B,C,1,1)
        max_pool = torch.amax(x, dim=(2, 3), keepdim=True)
         # (B,C,1,1)
        ca = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))   # (B,C,1,1)
        print('channel attention shape (C 1 1)', ca.shape)
        x = x * ca
        print('x shape after ca', x.shape)

        # ---- Spatial attention ----
        avg_map = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_map, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        sa = torch.sigmoid(self.spatial(torch.cat([avg_map, max_map], dim=1)))  # (B, 1, H, W)
        print('spatial attention shape (1 H W)', ca.shape)
        x = x * sa
        print('x shape after sa', x.shape)
        return x'''

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
        print('print(chan_att.size())')
        print(chan_att.size())
        fp = chan_att * f
        print('print(fp.size())')
        print(fp.size())
        spat_att = self.spatial_attention(fp)
        print('print(spat_att.size())')
        print(spat_att.size())
        fpp = spat_att * fp
        print('print(fpp.size())')
        print(fpp.size())
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
        '''self.proj = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True) if act == "relu" else nn.GELU()
        )'''
        self.proj = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: concatenated aligned features, shape (B, C_concat, H, W)
        returns: fused feature, shape (B, output_dim, H, W)
        """
        x = self.cbam(x)
        x = self.proj(x)
        exit()
        return x


