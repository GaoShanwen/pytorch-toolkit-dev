######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: regnet.py
# function: reduce the last fc layers' dim of regnet for reid.
######################################################
import torch.nn as nn
from timm.models import RegNet, RegNetCfg
from timm.layers.classifier import _create_fc
from timm.models.regnet import model_cfgs, _filter_fn
from timm.models._builder import build_model_with_cfg
from timm.models._registry import register_model


class RegNetRedution(RegNet):
    """convert RegNet for reid"""

    def __init__(
        self,
        cfg: RegNetCfg,
        in_chans=3,
        num_classes=1000,
        output_stride=32,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        zero_init_last=True,
        reduction_dim=128,
        **kwargs,
    ):
        super(RegNetRedution, self).__init__(
            cfg,  # type: ignore
            in_chans,
            num_classes,
            output_stride,
            global_pool,
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


def _create_regnet(variant, pretrained, **kwargs):
    return build_model_with_cfg(
        RegNetRedution,
        variant,
        pretrained,
        model_cfg=model_cfgs[variant],
        pretrained_filter_fn=_filter_fn,
        **kwargs,
    )


@register_model
def regnety_redution_008(pretrained=False, **kwargs) -> RegNet:
    """RegNetY-redution-0.8GF"""
    return _create_regnet("regnety_008", pretrained, **kwargs)  # type: ignore


@register_model
def regnety_redution_016(pretrained=False, **kwargs) -> RegNet:
    """RegNetY-redution-1.6GF"""
    return _create_regnet("regnety_016", pretrained, **kwargs)  # type: ignore


@register_model
def regnety_redution_040(pretrained=False, **kwargs) -> RegNet:
    """RegNetY-redution-4.0GF"""
    return _create_regnet("regnety_040", pretrained, **kwargs)  # type: ignore


@register_model
def regnety_redution_064(pretrained=False, **kwargs) -> RegNet:
    """RegNetY-redution-6.4GF"""
    return _create_regnet("regnety_064", pretrained, **kwargs)  # type: ignore


@register_model
def regnety_redution_120(pretrained=False, **kwargs) -> RegNet:
    """RegNetY-redution-12.0GF"""
    return _create_regnet("regnety_120", pretrained, **kwargs)  # type: ignore


@register_model
def regnetv_redution_040(pretrained=False, **kwargs) -> RegNet:
    """RegNetV-redution 6.35GFlops"""
    return _create_regnet("regnetv_040", pretrained, **kwargs)  # type: ignore


@register_model
def regnetz_redution_040_h(pretrained=False, **kwargs) -> RegNet:
    """RegNetZ-redution 6.4GFlops"""
    return _create_regnet("regnetz_040_h", pretrained, **kwargs)  # type: ignore


@register_model
def regnetz_redution_040(pretrained=False, **kwargs) -> RegNet:
    """RegNetZ-redution 6.35GFlops"""
    return _create_regnet("regnetz_040", pretrained, **kwargs)  # type: ignore


if __name__ == "__main__":
    import timm
    import torch
    from timm import utils

    # from timm.optim import create_optimizer_v2, optimizer_kwargs
    import torch.optim as optim

    # m = timm.create_model('regnety_redution_016.tv2_in1k', pretrained=True, num_classes=100)
    # m = timm.create_model('regnetz_redution_040_h.ra3_in1k', pretrained=True, num_classes=100)
    # m = timm.create_model("regnety_redution_040.ra3_in1k", pretrained=True, num_classes=4281)
    m = timm.create_model("regnety_320.swag_ft_in1k", pretrained=True, num_classes=4281)
    o = m(torch.randn(2, 3, 224, 224))
    # parameters = m.parameters()
    # optimizer = optim.Adam(parameters)
    # saver = utils.CheckpointSaver(
    #     model=m,
    #     optimizer=optimizer,
    #     args=None,
    #     model_ema=None,
    #     amp_scaler=None,
    #     checkpoint_dir=None,
    #     recovery_dir=None,
    #     decreasing=None,
    #     max_history=1
    # )
    # best_metric, best_epoch = saver.save_checkpoint(2, metric=0.5)
    # optimizer = create_optimizer_v2(m, **optimizer_kwargs(cfg={"lr":0.1}),)
    # save_path = "output/converted_model/regnety_redution_040.ra3_in1k-test.pth.tar"
    # saver._save(save_path, 2, metric=0.01)
    import pdb

    pdb.set_trace()
