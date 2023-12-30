import collections
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split


class Accumulator:  # @save
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):  # @save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制


def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])


if __name__ == "__main__":
    # # 生成随机数据
    # x = torch.randn((3000, 5, 30))  # 输入数据（3000，5，30）
    # y = torch.randint(0, 5, (3000, 1))  # 输出数据（3000，30，1）
    data = np.load("output/temp/weighted_knn.npz")
    scores, plabels, gts = data["scores"], data["plabels"], data["gts"]
    x, y = [], []
    for score, plabel, gt in zip(scores, plabels, gts):
        final_l = collections.Counter(plabel.tolist()).most_common()[:5]
        final_l = [l for l, _ in final_l] + [-1] * (5 - len(final_l))
        random.shuffle(final_l)
        if gt not in final_l:
            continue
        input = np.zeros((5, 30))
        for i, (l, s) in enumerate(zip(plabel, score)):
            if l not in final_l:
                continue
            input[final_l.index(l), i] = s
        y.append(final_l.index(gt))
        x.append(input)
    tensors = [torch.from_numpy(arr).type(torch.FloatTensor) for arr in x]
    x = torch.cat(tensors, dim=0).view(-1, 5, 30)
    y = torch.from_numpy(np.array(y)).unsqueeze(1)

    # 将数据划分为训练集和验证集
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    # 定义超参数
    num_epochs = 2000  # 训练轮数
    batch_size = 10000  # 批次大小
    learning_rate = 0.01  # 学习率
    W = torch.normal(0, 0.01, size=(30, 1), requires_grad=True)
    b = torch.zeros((5, 1), requires_grad=True)
    optimizer = torch.optim.Adam([W, b], lr=learning_rate)

    def net(X):
        scores = torch.matmul(X, W) + b
        return softmax(scores.reshape(-1, 5))

    # 训练模型
    for epoch in range(num_epochs):
        for i in range(0, x_train.shape[0], batch_size):
            # 获取批次数据（x_batch，y_batch）
            x_batch = x_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size]
            # # 点乘运算
            y_pred = net(x_batch)  # softmax(torch.matmul(x_batch, W.t())+b)  # （batch_size，1）

            # 计算损失函数
            loss = F.cross_entropy(y_pred, y_batch.reshape(-1))

            # 反向传播和优化
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

    def evaluate_accuracy(net, data_iter):  # @save
        """计算在指定数据集上模型的精度"""
        if isinstance(net, torch.nn.Module):
            net.eval()  # 将模型设置为评估模式
        metric = Accumulator(2)  # 正确预测数、预测总数
        with torch.no_grad():
            for X, y in data_iter:
                metric.add(accuracy(net(X), y), y.numel())

        # import pdb; pdb.set_trace()
        return metric[0] / metric[1]

    test_iter = zip(x_val, y_val)
    test_acc = evaluate_accuracy(net, test_iter)
    print("weights: ", W.detach().view(-1).numpy().round(decimals=3))  # .tolist()
    print("biais: ", b.detach().view(-1).numpy().round(decimals=3))  # .tolist()
    print("test_acc: ", test_acc)
