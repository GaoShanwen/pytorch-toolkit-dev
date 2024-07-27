######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2024.04.25
# filenaem: feat_extract_model.py
# function: REbuild Model for Feature Extract.
######################################################
import torch
import torch.nn as nn

from timm.layers.classifier import _create_fc


class FeatExtractModel(nn.Module):
    def __init__(self, model, model_name, feat_dim=128):
        super().__init__()
        model_name = model_name.split("_")[0]
        assert model_name in ["mobilenetv3", "mobilenetv4", "regnety", "haloregnetz"], f"{model_name} is not support yet!"
        # num_pooled_features = model.num_features
        self.num_classes = model.num_classes
        self.num_features = feat_dim
        if model_name.startswith("mobilenet"):
            cls_in_features = model.classifier.in_features
            model.classifier = nn.Identity()
            use_conv = False
        else:
            # num_pooled_features *= model.head.global_pool.feat_mult()
            cls_in_features = model.head.fc.in_features
            model.head.fc = nn.Identity()
            model.head.flatten = nn.Identity()
            use_conv = model.head.use_conv
        self.out_layer = nn.Flatten(1)
        self.reduction = _create_fc(cls_in_features, feat_dim, use_conv)
        self.classifier = _create_fc(feat_dim, self.num_classes, use_conv)
        self.base_model = model

    @torch.jit.ignore
    def get_classifier(self):
        return self.classifier
    
    def remove_head(self):
        self.classifier = nn.Identity()
        return self
    
    def forward(self, x):
        x = self.base_model(x)
        x = self.reduction(x)
        x = self.classifier(x)
        return self.out_layer(x)

    
if __name__ == "__main__":
    import timm
    model_name = "mobilenetv4_hybrid_medium.e500_r224_in1k" 
    # model_name = "mobilenetv4_conv_large.e500_r256_in1k" 
    # model_name = "regnety_160.swag_ft_in1k" # "mobilenetv3_large_100.miil_in21k_ft_in1k" #
    m = timm.create_model(model_name, pretrained=True, num_classes=20)
    m = FeatExtractModel(m, model_name)
    m.classifier = nn.Identity()
    o = m(torch.randn(2, 3, 224, 224))
    print(m)
    print(o.shape)