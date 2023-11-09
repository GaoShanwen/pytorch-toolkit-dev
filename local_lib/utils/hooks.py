from torch.utils.tensorboard import SummaryWriter 
import time

starttime = time.strftime("%Y-%m-%d_%H:%M:%S")#时间格式可以自定义，如果需要定义到分钟记得改下冒号，否则输入logdir时候会出问题
print("Start experiment:", starttime)#定义实验时间

writer = SummaryWriter(log_dir="./log/"+starttime[:13],comment=starttime[:13],flush_secs=60)#以实验时间命名，[:13]可以自定义，我是定义到小时基本能确定是哪个实验了

writer.add_scalar('loss/train_loss', train_loss, epoch)
writer.add_scalar('acc/train_acc', train_acc, epoch)
writer.add_scalar('loss/test_loss', test_loss, epoch)
writer.add_scalar('acc/val_acc', acc, epoch)
