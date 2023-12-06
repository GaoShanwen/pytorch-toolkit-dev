######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: byoanet.py
# function: reduce the last fc layers' dim of halonet for reid.
######################################################
import torch.nn as nn
from typing import Tuple, Optional, Union

from timm.models.byoanet import ByobNet, ByoModelCfg, model_cfgs
from timm.layers.classifier import _create_fc
from timm.models._registry import register_model
from timm.models._builder import build_model_with_cfg
from timm.models._registry import register_model


class ByobRedutionNet(ByobNet):
    def __init__(
        self,
        cfg: ByoModelCfg,
        num_classes: int = 1000,
        in_chans: int = 3,
        global_pool: str = "avg",
        output_stride: int = 32,
        img_size: Optional[Union[int, Tuple[int, int]]] = None,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        zero_init_last: bool = True,
        reduction_dim=128,
        **kwargs,
    ):
        super(ByobRedutionNet, self).__init__(
            cfg,  # type: ignore
            num_classes,
            in_chans,
            global_pool,
            output_stride,
            img_size,
            drop_rate,
            drop_path_rate,
            zero_init_last,
            **kwargs,
        )
        del self.head.fc
        self.reduction_dim = reduction_dim
        num_pooled_features = self.num_features * self.head.global_pool.feat_mult()
        if self.head.use_conv:
            self.head.reduction = nn.Conv2d(num_pooled_features, self.reduction_dim, 1, bias=True)
        else:
            self.head.reduction = nn.Linear(num_pooled_features, self.reduction_dim, bias=True)
        self.head.fc = _create_fc(self.reduction_dim, num_classes, self.head.use_conv)

    def reset_classifier(self, num_classes, global_pool="avg"):
        self.head.reset(num_classes, pool_type=global_pool)

    def forward_head(self, x, pre_logits: bool = False):
        x = self.head.global_pool(x)
        x = self.head.drop(x)
        if pre_logits:
            return self.head.flatten(x)
        x = self.head.reduction(x)  # type: ignore
        x = self.head.fc(x)
        return self.head.flatten(x)


def _create_byoanet(variant, cfg_variant=None, pretrained=False, **kwargs):
    return build_model_with_cfg(
        ByobRedutionNet,
        variant,
        pretrained,
        model_cfg=model_cfgs[variant] if not cfg_variant else model_cfgs[cfg_variant],
        feature_cfg=dict(flatten_sequential=True),
        **kwargs,
    )


@register_model
def haloregnetz_redution_b(pretrained=False, **kwargs) -> ByobNet:
    """Halo + RegNetZ"""
    return _create_byoanet("haloregnetz_b", pretrained=pretrained, **kwargs)  # type: ignore


if __name__ == "__main__":
    import timm

    m = timm.create_model("haloregnetz_redution_b.ra3_in1k", pretrained=True, num_classes=100)
    # import torch
    # o = m(torch.randn(2, 3, 224, 224))
