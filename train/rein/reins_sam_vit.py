from functools import partial
from typing import Tuple, Type
import torch.nn as nn
from .reins import Reins
from .utils import set_requires_grad, set_train
from mmseg.models.builder import BACKBONES, MODELS
from .sam_vit import SAMViT
import torch
import torch.nn.functional as F


@BACKBONES.register_module()
class ReinsSAMViT(SAMViT):
    def __init__(
        self,
        reins_config=None,
        img_size: int = 1024,
        out_indices=[3, 5, 7, 11],
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        init_cfg=None,
    ) -> None:
        super().__init__(
            img_size,
            out_indices,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            norm_layer,
            act_layer,
            use_abs_pos,
            use_rel_pos,
            rel_pos_zero_init,
            window_size,
            global_attn_indexes,
            init_cfg,
        )
        self.reins: Reins = MODELS.build(reins_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        x = self.patch_embed(x)
        Hp, Wp = H // self.patch_size, W // self.patch_size
        if self.pos_embed is not None:
            x = x + self.pos_embed
        C = x.shape[-1]
        features = []
        for idx, blk in enumerate(self.blocks):
            x=blk(x)
            x=self.reins.forward(x.view(B,-1,C),idx,batch_first=True,has_cls_token=True)
            x = x.view(B, Hp, Wp, C)
            if idx in self.out_indices:
                features.append(x.permute(0, 3, 1, 2))
        features[0] = F.interpolate(
            features[0], scale_factor=4, mode="bilinear", align_corners=False
        )
        features[1] = F.interpolate(
            features[1], scale_factor=2, mode="bilinear", align_corners=False
        )
        features[3] = F.interpolate(
            features[3], scale_factor=0.5, mode="bilinear", align_corners=False
        )
        return self.reins.return_auto(tuple(features))

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["reins"])
        set_train(self, ["reins"])
