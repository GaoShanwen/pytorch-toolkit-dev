######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2023.11.09
# filenaem: mobilenetv3.py
# function: reduce the last fc layers' dim of mobilenetv3 for reid.
######################################################
from functools import partial

import torch.nn as nn
import torch.nn.functional as F
from timm.layers import SelectAdaptivePool2d
from timm.models import MobileNetV3

# from timm.models.mobilenetv3 import _gen_mobilenet_v3
from timm.models._builder import build_model_with_cfg, pretrained_cfg_for_features
from timm.models._efficientnet_blocks import SqueezeExcite
from timm.models._efficientnet_builder import decode_arch_def, resolve_act_layer, resolve_bn_args, round_channels
from timm.models._registry import register_model
from timm.models.layers import Linear


class MobileNetV3Redution(MobileNetV3):
    """convert MobiletNet-V3 for reid"""

    def __init__(self, block_args, num_classes=1000, reduction_dim=128, **kwargs):
        super(MobileNetV3Redution, self).__init__(block_args, num_classes, **kwargs)
        del self.classifier
        self.reduction_dim = reduction_dim
        self.reduction = Linear(self.conv_head.out_channels, self.reduction_dim)
        self.classifier = Linear(self.reduction_dim, num_classes) if num_classes > 0 else nn.Identity()

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1]
        layers.extend(self.blocks)
        layers.extend([self.global_pool, self.conv_head, self.act2])
        layers.extend([nn.Flatten(), nn.Dropout(self.drop_rate), self.reduction, self.classifier])
        return nn.Sequential(*layers)

    def reset_classifier(self, num_classes, global_pool="avg"):
        self.num_classes = num_classes
        # cannot meaningfully change pooling of efficient head after creation
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.reduction = Linear(self.conv_head.out_channels, self.reduction_dim)
        self.classifier = Linear(self.reduction_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = self.flatten(x)
        if pre_logits:
            return x
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.reduction(x)
        return self.classifier(x)


def _create_mnv3(variant, pretrained=False, **kwargs):
    model_cls = MobileNetV3Redution
    kwargs_filter = None
    features_mode = "cls"

    model = build_model_with_cfg(
        model_cls,
        variant,
        pretrained,
        features_only=features_mode == "cfg",
        pretrained_strict=features_mode != "cls",
        kwargs_filter=kwargs_filter,
        **kwargs,
    )
    if features_mode == "cls":
        model.default_cfg = pretrained_cfg_for_features(model.default_cfg)
    return model


def _gen_mobilenet_v3(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a MobileNet-V3 model.

    Ref impl: ?
    Paper: https://arxiv.org/abs/1905.02244

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    if "small" in variant:
        num_features = 1024
        if "minimal" in variant:
            act_layer = resolve_act_layer(kwargs, "relu")
            arch_def = [
                # stage 0, 112x112 in
                ["ds_r1_k3_s2_e1_c16"],
                # stage 1, 56x56 in
                ["ir_r1_k3_s2_e4.5_c24", "ir_r1_k3_s1_e3.67_c24"],
                # stage 2, 28x28 in
                ["ir_r1_k3_s2_e4_c40", "ir_r2_k3_s1_e6_c40"],
                # stage 3, 14x14 in
                ["ir_r2_k3_s1_e3_c48"],
                # stage 4, 14x14in
                ["ir_r3_k3_s2_e6_c96"],
                # stage 6, 7x7 in
                ["cn_r1_k1_s1_c576"],
            ]
        else:
            act_layer = resolve_act_layer(kwargs, "hard_swish")
            arch_def = [
                # stage 0, 112x112 in
                ["ds_r1_k3_s2_e1_c16_se0.25_nre"],  # relu
                # stage 1, 56x56 in
                ["ir_r1_k3_s2_e4.5_c24_nre", "ir_r1_k3_s1_e3.67_c24_nre"],  # relu
                # stage 2, 28x28 in
                [
                    "ir_r1_k5_s2_e4_c40_se0.25",
                    "ir_r2_k5_s1_e6_c40_se0.25",
                ],  # hard-swish
                # stage 3, 14x14 in
                ["ir_r2_k5_s1_e3_c48_se0.25"],  # hard-swish
                # stage 4, 14x14in
                ["ir_r3_k5_s2_e6_c96_se0.25"],  # hard-swish
                # stage 6, 7x7 in
                ["cn_r1_k1_s1_c576"],  # hard-swish
            ]
    else:
        num_features = 1280
        if "minimal" in variant:
            act_layer = resolve_act_layer(kwargs, "relu")
            arch_def = [
                # stage 0, 112x112 in
                ["ds_r1_k3_s1_e1_c16"],
                # stage 1, 112x112 in
                ["ir_r1_k3_s2_e4_c24", "ir_r1_k3_s1_e3_c24"],
                # stage 2, 56x56 in
                ["ir_r3_k3_s2_e3_c40"],
                # stage 3, 28x28 in
                ["ir_r1_k3_s2_e6_c80", "ir_r1_k3_s1_e2.5_c80", "ir_r2_k3_s1_e2.3_c80"],
                # stage 4, 14x14in
                ["ir_r2_k3_s1_e6_c112"],
                # stage 5, 14x14in
                ["ir_r3_k3_s2_e6_c160"],
                # stage 6, 7x7 in
                ["cn_r1_k1_s1_c960"],
            ]
        else:
            act_layer = resolve_act_layer(kwargs, "hard_swish")
            arch_def = [
                # stage 0, 112x112 in
                ["ds_r1_k3_s1_e1_c16_nre"],  # relu
                # stage 1, 112x112 in
                ["ir_r1_k3_s2_e4_c24_nre", "ir_r1_k3_s1_e3_c24_nre"],  # relu
                # stage 2, 56x56 in
                ["ir_r3_k5_s2_e3_c40_se0.25_nre"],  # relu
                # stage 3, 28x28 in
                [
                    "ir_r1_k3_s2_e6_c80",
                    "ir_r1_k3_s1_e2.5_c80",
                    "ir_r2_k3_s1_e2.3_c80",
                ],  # hard-swish
                # stage 4, 14x14in
                ["ir_r2_k3_s1_e6_c112_se0.25"],  # hard-swish
                # stage 5, 14x14in
                ["ir_r3_k5_s2_e6_c160_se0.25"],  # hard-swish
                # stage 6, 7x7 in
                ["cn_r1_k1_s1_c960"],  # hard-swish
            ]
    se_layer = partial(
        SqueezeExcite,
        gate_layer="hard_sigmoid",
        force_act_layer=nn.ReLU,
        rd_round_fn=round_channels,
    )
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        num_features=num_features,
        stem_size=16,
        fix_stem=channel_multiplier < 0.75,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=act_layer,
        se_layer=se_layer,
        **kwargs,
    )
    model = _create_mnv3(variant, pretrained, **model_kwargs)
    return model


@register_model
def mobilenetv3_redution_large_100(pretrained=False, **kwargs) -> MobileNetV3:
    """Redution MobileNet V3"""
    model = _gen_mobilenet_v3("mobilenetv3_large_100", 1.0, pretrained=pretrained, **kwargs)
    return model


if __name__ == "__main__":
    import timm
    import torch

    # m = timm.create_model("mobilenetv3_redution_large_100", pretrained=True, num_classes=41)
    m = timm.create_model("mobilenetv3_redution_large_100.miil_in21k_ft_in1k", pretrained=True, num_classes=41)
    o = m(torch.randn(2, 3, 224, 224))
    # load_checkpoint(model, "output/train/20231011-120458-mobilenetv3_large_100-224/model_best.pth.tar", False)
    # model.reset_classifier(0) # 移除分类层
    print(m)
