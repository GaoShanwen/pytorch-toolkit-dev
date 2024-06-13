######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2024.05.03
# filenaem: reid_model.py
# function: Rebuild Model for Re-ID task.
######################################################
import torch
import torch.nn as nn

from timm.layers.classifier import _create_fc

from local_lib.models.reid_loss import pairwise_circleloss, triplet_loss


def create_fc(num_features, num_classes, use_conv=False, bias=True):
    if num_classes <= 0:
        fc = nn.Identity()  # pass-through (no classifier)
    elif use_conv:
        fc = nn.Conv2d(num_features, num_classes, 1, bias=bias)
    else:
        fc = nn.Linear(num_features, num_classes, bias=bias)
    return fc


class ReidModel(nn.Module):
    def __init__(self, model, model_name, config=None):
        super().__init__()
        model_name = model_name.split("_")[0]
        # assert model_name in ["mobilenetv3", "regnety", "haloregnetz"], f"{model_name} is not support yet!"
        assert isinstance(config, dict), "config must be a dict!"

        weights = config.get("weights", {})
        assert weights.get("cls", None) and weights.get("feats", None), "weights must contain 'cls' and 'feats'!"
        feat_dim, metric_loss = config.get("feat_dim", None), config.get("metric_loss", None)
        assert feat_dim and metric_loss, "feat_dim and metric_lossmust be set!"
        self.num_classes = model.num_classes
        self.weights = {k: torch.tensor(v) for k, v in weights.items()}
        self.num_features = feat_dim
        self.metric_loss = eval(metric_loss)
        
        if getattr(model, 'classifier', None): #for mobilenetv3
            num_pooled_features = model.classifier.in_features
            model.classifier = nn.Identity()
            use_conv = False
        elif getattr(model, 'fc', None):
            num_pooled_features = model.fc.in_features
            model.fc = nn.Identity()
            model.global_pool = nn.Identity()
            use_conv = True #False
        elif getattr(model, 'head', None):
            num_pooled_features = model.head.fc.in_features
            model.head.fc = nn.Identity()
            model.head.flatten = nn.Identity()
            use_conv = model.head.use_conv
        else:
            raise ValueError("Invalid model!")

        # reduction = create_fc(num_pooled_features, feat_dim, use_conv, False)
        reduction = nn.Conv2d(num_pooled_features, feat_dim, 1, 1, bias=False)
        global_pool = nn.AdaptiveAvgPool2d(1)
        last_layer = nn.Identity()
        self.base_model = nn.Sequential(
            model, 
            reduction,
            global_pool,
            last_layer,
        )

        class OutLayer(nn.Module):
            def __init__(self, features, classifier):
                super().__init__()
                self.features = features
                self.classifier = classifier
        features = nn.Flatten(1) # nn.Identity() #

        classifier = nn.Sequential(
            nn.BatchNorm2d(feat_dim, eps=1e-05, momentum=0.1),
            # nn.ReLU(inplace=True),
            _create_fc(feat_dim, self.num_classes, use_conv),
            nn.Flatten(1),
        )
        self.out_layer = OutLayer(features, classifier)

    @torch.jit.ignore
    def get_classifier(self):
        return self.out_layer.classifier
    
    def remove_head(self):
        return self.base_model
    
    def forward(self, x):
        x = self.base_model(x)
        out = {
            "feats": self.out_layer.features(x),
            "cls": self.out_layer.classifier(x)
        }
        return out
    
    def get_loss(self, cls_loss_fn, output, target):
        cls_loss = cls_loss_fn(output["cls"], target)
        feats_loss = self.metric_loss(output["feats"], target)
        total_loss = cls_loss * self.weights["cls"] + feats_loss * self.weights["feats"]
        return {"cls": cls_loss, "feats": feats_loss, "total": total_loss}


if __name__ == "__main__":
    import timm
    model_name = "resnet50.tv2_in1k" #"regnety_160.swag_ft_in1k" # "mobilenetv3_large_100.miil_in21k_ft_in1k" #
    m = timm.create_model(model_name, pretrained=True, num_classes=20)
    m = ReidModel(m, model_name, {"weights": {"cls": 1, "feats": 1}, "feat_dim": 128, "metric_loss": "triplet_loss"})
    # m.classifier = nn.Identity()
    o = m(torch.randn(2, 3, 224, 224))
    print(m)
    print("feats", o["feats"].shape)
    print("cls", o["cls"].shape)
    
    m = m.remove_head()
    m.last_layer = nn.Flatten(1)
    o = m(torch.randn(2, 3, 224, 224))
    print(o.shape)