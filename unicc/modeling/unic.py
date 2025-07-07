import math
import os
from collections import defaultdict
from typing import Dict, List, Optional


import torch
import torch.nn as nn
import torch.nn.functional as F

from dinov2.models import vision_transformer
from dinov2.models import timesformer 
from teachers.config import CONFIG
from dinov2.logging import setup_logging, ExternalLogger, MetricLogger
import logging
logger = logging.getLogger()


class UNIC(nn.Module):
    def __init__(self, encoder, lp, in_chans, strategy = 'split', num_frames = 3):
        super().__init__()
        self.encoder = encoder
        logger.info(self.encoder.__class__.__name__)
        if self.encoder.__class__.__name__.startswith("TimeSFormerStudent"):
            logger.info("YESSS")
        self.lp = lp
        self.in_chans = in_chans
        self.strategy = strategy
        self.num_frames = num_frames

    def forward(self, image):
        logger.info('uso la strategia ', self.strategy)

        image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)

        if self.num_frames == 3:
            image = image[:, CONFIG['nove']['bands'], :]
            std = CONFIG['nove']['std']
            mean = CONFIG['nove']['mean']
        
        elif self.num_frames == 4:
            image = image[:, CONFIG['all']['bands'], :]
            std = CONFIG['all']['std']
            mean = CONFIG['all']['mean']
        

        image = (image - mean.to(image.device)) / std.to(image.device)
        
        x, num_register_tokens = self.encoder.prepare_tokens_with_masks(image)
        
        output_cls = [x[:, 0, :]]
        output_patch = [x[:, 1 + num_register_tokens :, :]]
        
        
        if isinstance(self.encoder.blocks[0], nn.ModuleList):
            # chunked
            for block_chunk in self.encoder.blocks:
                for blk in block_chunk:
                    x = blk(x)
                    output_cls.append(x[:, 0, :])
                    output_patch.append(x[:, 1 + num_register_tokens :, :])
        else:
            # not chunked
            for blk in self.encoder.blocks:
                x = blk(x)
                output_cls.append(x[:, 0, :])
                output_patch.append(x[:, 1 + num_register_tokens :, :])
        
        if self.encoder.__class__.__name__.startswith("TimeSFormerStudent"):  
            patch_tokens_split = None
            B, N, D = output_patch[-1].shape           # N = T * num_patches
            num_patches = self.encoder.num_patches     # 196 se 224/16
            
            T = N // num_patches                       # numero frame
            
            if self.strategy == "split":
                # [B, T, 196, D]
                patch_tokens_split = output_patch[-1].reshape(B, T, num_patches, D)
            
            elif self.strategy == "mean":
                # Fai la media T?1 per TUTTI i livelli
                for i, p in enumerate(output_patch):
                    B, N, D = p.shape
                    num_patches = self.encoder.num_patches          # 196
                    T = N // num_patches                            # 3
                    output_patch[i] = p.reshape(B, T, num_patches, D).mean(dim=1)
    
            out = self.lp(
                    output_cls,
                    output_patch,
                    patch_tokens_split=patch_tokens_split,
                    strategy=self.strategy,
            )
            
        else:
            out = self.lp(output_cls, output_patch)

        

        return out


class LP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        head_dims: Dict[str, int],
        n_encoder_blocks: int,
        patch_tokens_split: Optional[torch.Tensor] = None,
        strategy: str = None,
        which_blocks: List[int] = None,
        hidden_dim: int = 768,
        last_hidden_dim: int = 3072,
        prenorm: bool = False,
        midnorm: bool = False,
        std: float = 0.02,
    ):
        super().__init__()

        if which_blocks is None:
            which_blocks = list(range(n_encoder_blocks))
        self.which_blocks = which_blocks

        def _make_head(output_dim):
            return nn.ModuleList(
                [
                    (
                        AdaptMLP(
                            hidden_dim=(
                                last_hidden_dim
                                if bix == n_encoder_blocks - 1
                                else hidden_dim
                            ),
                            prenorm=prenorm,
                            midnorm=midnorm,
                            dim=input_dim,
                            output_dim=output_dim,
                        )
                        if bix in which_blocks
                        else None
                    )
                    for bix in range(n_encoder_blocks)
                ]
            )

        self.heads = nn.ModuleDict(
            {
                hname: nn.ModuleDict(
                    {
                        "cls": _make_head(head_dims[hname]),
                        "patch": _make_head(head_dims[hname]),
                    }
                )
                for hname in head_dims.keys()
            }
        )

        for m in self.heads.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
            self,
            x_cls:  List[torch.Tensor],
            x_patch: List[torch.Tensor],
            *,
            patch_tokens_split: Optional[torch.Tensor] = None,
            strategy: str = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        out = defaultdict(dict)
        

        for idx, (hname, head_dict) in enumerate(self.heads.items()):
            xc, xp = 0, 0 
            
            for bix in self.which_blocks:
                xc = xc + head_dict["cls"][bix](x_cls[bix + 1])

                if strategy == "split":
                    if bix == self.which_blocks[-1]:
                        xp = head_dict["patch"][bix](patch_tokens_split[:, idx])
                else:
                    xp = xp + head_dict["patch"][bix](x_patch[bix + 1])

            out[hname]["cls"] = xc
            out[hname]["patch"] = xp

        return out
        
        


class AdaptMLP(nn.Module):

    def __init__(
        self,
        hidden_dim,
        prenorm=False,
        midnorm=False,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
        scale=1.0,
        zinit=False,
        dim=None,
        output_dim=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.prenorm = prenorm
        self.midnorm = midnorm
        self.norm_fn = norm_fn
        self.act_fn = act_fn
        self.scale = nn.Parameter(torch.ones(1).float()) if scale == 0.0 else scale
        self.zinit = zinit
        if dim is not None:
            self.setup(dim, output_dim)

    def extra_repr(self):
        repr = "scale={}, zinit={}".format(self.scale, self.zinit)
        return repr

    def setup(self, dim, output_dim=None):
        layers = []

        if self.prenorm:
            layers.append(self.norm_fn(dim))

        layers.append(nn.Linear(dim, self.hidden_dim))
        if self.zinit:
            nn.init.kaiming_uniform_(layers[-1].weight, a=math.sqrt(5))
            nn.init.zeros_(layers[-1].bias)

        if self.midnorm:
            layers.append(self.norm_fn(self.hidden_dim))

        layers.append(self.act_fn())

        layers.append(
            nn.Linear(self.hidden_dim, dim if output_dim is None else output_dim)
        )
        if self.zinit:
            nn.init.zeros_(layers[-1].weight)
            nn.init.zeros_(layers[-1].bias)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.scale * self.layers(x)


def _build_encoder_from_args(args):

    if args.arch.startswith("vit"):
        logger.info("creato vision transformer")
        return vision_transformer.get_model(
        arch=args.arch,
        patch_size=args.patch_size,
        drop_path_rate=args.drop_path_rate,
        img_size=args.image_size,
        in_chans=args.in_chans
        )
    else:
        logger.info("creato timesformer")
        return timesformer.get_model(
            arch=args.arch,
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,
            img_size=args.image_size,
            in_chans=3,
            num_frames=args.num_frames          
    )
    


def load_student_encoder_from_checkpoint(ckpt_fname, ckpt_key="model"):
    assert os.path.isfile(ckpt_fname), "Student checkpoint ({}) not found!".format(
        ckpt_fname
    )
    ckpt = torch.load(ckpt_fname, "cpu")

    encoder = _build_encoder_from_args(ckpt["args"])

    state_dict = ckpt.get(ckpt_key, ckpt)
    encoder.load_state_dict(
        {
            k.replace("module.", "").replace("encoder.", ""): v
            for k, v in state_dict.items()
            if "encoder." in k
        }
    )

    return encoder, ckpt["epoch"]
    
    

from typing import Dict, List
from collections import defaultdict
import torch
import torch.nn as nn

class IdentityLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        head_dims: Dict[str, int],
        n_encoder_blocks: int,
        which_blocks: List[int] = None,
        hidden_dim: int = 768,
        last_hidden_dim: int = 3072,
        prenorm: bool = False,
        midnorm: bool = False,
        std: float = 0.02,
    ):
        super().__init__()
        if which_blocks is None:
            which_blocks = list(range(n_encoder_blocks))
        self.which_blocks = which_blocks
        self.head_dims = head_dims

        # crea un linear per ogni head_dim
        self.proj = nn.ModuleDict({
            hname: nn.Linear(input_dim, head_dims[hname])
            for hname in head_dims.keys()
        })

    def forward(
        self, x_cls: List[torch.Tensor], x_patch: List[torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        out = defaultdict(dict)
        for hname in self.head_dims.keys():
            out[hname]["cls"] = self.proj[hname](x_cls[-1])
            out[hname]["patch"] = self.proj[hname](x_patch[-1])
        return out



def build_student_from_args(args):

    encoder = _build_encoder_from_args(args)

    from teachers import TEACHER_CFG
    print('check nel file unic buildstudent bla bla', args.Teacher_strategy)
    if "abf" in args.Teacher_strategy or "mean" in args.Teacher_strategy:
        head_dims = {
            "mergedFeatures": max([TEACHER_CFG[tname]["num_features"] for tname in args.teachers])  # o metti a mano il valore corretto
        }
        print(head_dims)
        
    else:
        head_dims={
            tname: TEACHER_CFG[tname.strip()]["num_features"] for tname in args.teachers
        }
    
    print('\n\n\n\n')    
    print(head_dims)
    #print(encoder.embed_dim)
    
    if args.use_lp:
        lp_args = eval(args.lp_args)
        lp = LP(
            input_dim=encoder.embed_dim,
            head_dims=head_dims,
            n_encoder_blocks=encoder.n_blocks,
            **lp_args,
        )
    else:
      print('non uso lp, faccio identita')
      lp_args = eval(args.lp_args)
      lp = IdentityLP(
        input_dim=encoder.embed_dim,
        head_dims=head_dims,
        n_encoder_blocks=encoder.n_blocks,
        **lp_args,
      )
      

    model = UNIC(encoder, lp, args.in_chans, args.Student_strategy, args.num_frames)

    return model


def load_student_from_checkpoint(ckpt_fname, ckpt_key="model"):
    assert os.path.isfile(ckpt_fname), ckpt_fname
    ckpt = torch.load(ckpt_fname, "cpu")

    model = build_student_from_args(ckpt["args"])
    tnorms = ckpt["teacher_ft_stats"] if "teacher_ft_stats" in ckpt else None

    state_dict = ckpt.get(ckpt_key, ckpt)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})

    return model, tnorms, ckpt["epoch"]
