######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2024.04.25
# filenaem: multi_label_model.py
# function: Rebuild Model for Multi-Label Classification.
# reference: https://github.com/yang-ruixin/PyTorch-Image-Models-Multi-Label-Classification
######################################################

import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score


class MultiLabelModel(nn.Module):
    def __init__(self, model, config, drop_p: float=0.2):
        super().__init__()
        self.base_model = model
        self.attributes = config["attributes"]
        self.label_nums = config["label_nums"]
        self.weights = config["weights"]
        assert len(self.attributes) >= 2, f"attributes' number be request >= 2!" 
        assert len(self.attributes) == len(self.weights), "attributes' number must equal to weights!"
        self.label_pairs_num = len(self.weights)
        last_channel = model.num_features
        self.channels, moddle_fc = [last_channel] * len(self.attributes), nn.Identity()
        if config["channels"]:
            self.channels = config["channels"]
            moddle_fc = nn.Linear(in_features=last_channel, out_features=last_channel)
        
        model = model.remove_head()
        self.base_model = nn.Sequential(model, moddle_fc)
        
        assert sum(self.channels) % last_channel == 0, "channels' number must be divisible by last_channel!"
        for attr, label_num, channel_in in zip(self.attributes, self.label_nums, self.channels):
            attr_fc = nn.Sequential(
                nn.Dropout(p=drop_p),
                nn.Linear(in_features=channel_in, out_features=label_num)
            )
            exec(f"self.{attr} = attr_fc")

    def forward(self, x):
        x = self.base_model(x)

        output, start_channel = {}, 0
        for attr, channel in zip(self.attributes, self.channels):
            attr_x = x[:, start_channel:start_channel+channel]
            start_channel = (start_channel + channel) % x.shape[1]
            attr_x = eval(f'self.{attr}(attr_x)')
            output.update({attr: attr_x})
        return output

    def remove_head(self):
        return self.base_model
    
    def get_loss(self, loss_fn, output, target):
        device = torch.cuda.current_device()
        for idx, (attr, weight) in enumerate(zip(self.attributes, self.weights)):
            this_loss = loss_fn(output[attr], target[attr].to(device=device))
            loss = this_loss * weight if not idx else loss + this_loss * weight
        return loss

    def get_accuracy(self, accuracy, output, target, topk=(1,)):
        device = torch.cuda.current_device()
        attrs_acc1 = {}
        for idx, (attr, weight) in enumerate(zip(self.attributes, self.weights)):
            this_acc1, this_acc5 = accuracy(output[attr], target[attr].to(device=device), topk=topk)
            acc1 = this_acc1 * weight if not idx else acc1 + this_acc1 * weight
            acc5 = this_acc5 * weight if not idx else acc5 + this_acc5 * weight
            attrs_acc1.update({attr: this_acc1})
        return acc1, acc5, attrs_acc1

    def calculate_metrics(self, output, target):
        calculat_result = []
        for attr in self.attributes:
            calculat_result += balanced_accuracy_score(
                y_true=target[attr].cpu().numpy(), 
                y_pred=output[attr].cpu().argmax(1).numpy()
            )
        return calculat_result


if __name__ == "__main__":
    import timm
    import yaml
    from local_lib.models.feat_extract_model import FeatExtractModel
    with open("cfgs/multilabel/multilabel-regnety_040.ra3_in1k.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    model_name = "mobilenetv3_large_100.miil_in21k_ft_in1k" #cfg["model"] # "regnety_160.swag_ft_in1k" # 
    m = timm.create_model(model_name, pretrained=True)#, num_classes=20
    m = FeatExtractModel(m, model_name)
    # m = MultiLabelModel(m, cfg["multilabel"])
    # m.classifier = nn.Identity()
    print(m)
    with torch.no_grad():
        o = m(torch.randn(2, 3, 224, 224))
    print(o)
    m = m.remove_head()
    o = m(torch.randn(2, 3, 224, 224))
    print(o.shape)