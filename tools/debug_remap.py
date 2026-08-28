# Smoke test: instantiate MixedRFDETRDataset and print a few samples to verify class_mapping applied
import json
import sys
sys.path.insert(0, ".")
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

# Direct test of remap_target_labels to avoid constructing the full dataset
from local_lib.models.train import remap_target_labels
import torch

mapping = {8: 9, 10: 11}
print('Using mapping:', mapping)

# Build a fake target dict with labels containing old ids
labels = torch.tensor([3, 8, 12, 10], dtype=torch.int64)
orig_target = {
    'labels': labels,
    'boxes': torch.randn((4,4), dtype=torch.float32),
    'area': torch.rand((4,), dtype=torch.float32),
    'iscrowd': torch.zeros((4,), dtype=torch.int64)
}
print('orig labels:', orig_target['labels'].tolist())
new_t = remap_target_labels(orig_target, mapping)
print('remapped labels:', new_t['labels'].tolist())
# Also test case where no label matches mapping
labels2 = torch.tensor([1,2,3], dtype=torch.int64)
orig2 = {'labels': labels2}
print('orig2 labels:', orig2['labels'].tolist())
new2 = remap_target_labels(orig2, mapping)
print('remapped2 labels (should be empty):', new2['labels'].tolist())
