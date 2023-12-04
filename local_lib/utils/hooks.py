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
    def __init__(self, log_dir, exp_name="") -> None:
        if not exp_name:
            exp_name = time.strftime("%Y-%m-%d_%H-%M-%S")[:16]
            print("Start experiment:", exp_name)
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{exp_name}", comment=exp_name)

    def update(self, epoch, train_metrics, eval_metrics, lr) -> None:
        train_loss, eval_loss = train_metrics.pop("loss"), eval_metrics.pop("loss")
        self.writer.add_scalar("loss_train", train_loss, epoch)
        self.writer.add_scalar("loss_val", eval_loss, epoch)
        self.writer.add_scalar("lr", lr, epoch)
        self.writer.add_scalars("val_acc", eval_metrics, epoch)

    def close(self):
        self.writer.close()


if __name__ == "__main__":
    # 使用add_image方法
    # 构建一个100*100，3通道的img数据
    from collections import OrderedDict

    tb_writer = TensorBoardWriter("./logs")
    write_contents = [
        [0.94, 0.01, 0.91, 0.18, 0.1],
        [0.34, 0.1, 0.21, 0.38, 0.01],
        [0.09, 0.81, 0.11, 0.95, 0.001],
    ]
    for i, content in enumerate(write_contents):
        train_loss, top1, test_loss, top5, lr = content
        train_metrics = OrderedDict([("loss", train_loss)])
        eval_metrics = OrderedDict([("loss", test_loss), ("top1", top1), ("top5", top5)])
        tb_writer.update(i, train_metrics, eval_metrics, lr)
    tb_writer.close()
