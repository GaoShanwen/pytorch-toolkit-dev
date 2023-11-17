######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: hooks.py
# function: add tensorboard to train.py. 
######################################################
from torch.utils.tensorboard import SummaryWriter
import time


class TensorBoardWriter(object):
    def __init__(self, log_dir) -> None:
        starttime = time.strftime("%Y-%m-%d_%H:%M:%S")[:16]
        print("Start experiment:", starttime)
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{starttime}", comment=starttime, flush_secs=60)
    
    def write_content(self, train_loss, train_acc, test_loss, test_acc, epoch) -> None:
        self.writer.add_scalar('loss/train_loss', train_loss, epoch)
        self.writer.add_scalar('acc/train_acc', train_acc, epoch)
        self.writer.add_scalar('loss/test_loss', test_loss, epoch)
        self.writer.add_scalar('acc/val_acc', test_acc, epoch)
