import numpy as np
import torch
import sys
sys.path.append('.')
from local_lib.data.utils import box_ioa


if __name__ == "__main__":
    preds = torch.from_numpy(np.array([
        [0.366046, 0.513751, 0.0117025, 0.0214876],
        [0.471373, 0.504401, 0.0121832, 0.0217872],
        [0.474911, 0.504995, 0.0176712, 0.0258731],
        [0.362888, 0.511631, 0.0164649, 0.0305923]
    ]))
    gts = torch.from_numpy(np.array([
        [0.470083, 0.502094, 0.015000, 0.030000],
        [0.364831, 0.516424, 0.015000, 0.030000]
    ]))
    preds *= torch.tensor([640, 360, 640, 360])
    preds[:, :2] -= preds[:, 2:]/2. 
    preds[:, 2:] += preds[:, :2] 
    gts *= torch.tensor([640, 360, 640, 360])
    gts[:, :2] -= gts[:, 2:]/2. 
    gts[:, 2:] += gts[:, :2] 
    pred_keeps = box_ioa(preds.T, gts) <= 0.25
    print("before: ",preds)
    print("after: ", preds[pred_keeps.all(dim=0)])
    # p_keeps = (box_ioa(torch.from_numpy(boxes[p_idx, 1:5]).T, torch.from_numpy(boxes[v_idx, 1:5])) <= ioav).all(dim=0).numpy()