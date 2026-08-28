# Test MixedRFDETRDataModule.setup wrapping of validation dataset
import sys
import torch
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
from local_lib.models.train import MixedRFDETRDataset, MixedRFDETRDataModule


class SimpleBaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        # 3 samples with labels 0,8,10 to test mapping
        self.data = [torch.zeros((3,64,64)), torch.zeros((3,64,64)), torch.zeros((3,64,64))]
        self.targets = [
            {'labels': torch.tensor([0])},
            {'labels': torch.tensor([8])},
            {'labels': torch.tensor([10])},
        ]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class FakeBaseDM:
    def __init__(self):
        self._dataset_train = SimpleBaseDataset()
        self._dataset_val = SimpleBaseDataset()
    def setup(self, stage):
        # noop
        pass

base_dm = FakeBaseDM()
# mixed train wrap with mapping
mixed = MixedRFDETRDataset(base_dataset=base_dm._dataset_train, class_mapping={8:9,10:11})
print('Before setup, val labels:', [base_dm._dataset_val[i][1]['labels'].tolist() for i in range(len(base_dm._dataset_val))])
mdm = MixedRFDETRDataModule(base_dm, mixed)
mdm.setup('fit')
# After setup, base_dm._dataset_val should be a MixedRFDETRDataset if mapping applied
wrapped_val = base_dm._dataset_val
print('Wrapped val type:', type(wrapped_val))
print('After setup, val labels remapped:')
for i in range(len(wrapped_val)):
    img, tgt = wrapped_val[i]
    print(tgt.get('labels').tolist())
