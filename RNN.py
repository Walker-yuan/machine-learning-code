import numpy as np
import torch
import torch.utils.data as Data
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt

## 超参数
DOWNLOAD_MNIST = False
batch_size = 64
Input_size = 28
Hidden_size = 64
Output_size = 10
epoches = 10
lr = 0.0001


## 对数据进行处理， 将数据改变为tensor的形式, 同时对数据进行标准化处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,),  # 3 for RGB channels
                         std=(0.3087,))])

## 如果自己定义数据集的话, 重写__len__和__getitem__

## train data
## 其中root是数据集的下载路径, train是下载训练数据集和测试数据集
## transform是对数据进行处理，前面定义了transfrom的形式, download是否从datasets中下载mnist数据，如果有就不下载（False）
train_mnist = torchvision.datasets.MNIST(root='./data/',
                                   train=True,
                                   transform=transform,
                                   download=DOWNLOAD_MNIST)

## train data loader
## 将train_mnist数据集进行封装, dataset为需要封装的数据集, batch_size为批训练的数据个数
## shuffle, 洗牌, 是否打乱数据集
## 还有参数num_workers 线程数，默认为0; drop_last如果数据总数除以batch_size不为整数, 此时若为True,
## 则删除最后一个不完整的batch, 若为False，则保留

train_loader = torch.utils.data.DataLoader(dataset=train_mnist,
                                          batch_size=batch_size,
                                          shuffle=True)


## test data
test_mnist = torchvision.datasets.MNIST(root='./data/',
                                   train=False,
                                   transform=transform,
                                   download=DOWNLOAD_MNIST)


## test data loader
test_loader = torch.utils.data.DataLoader(dataset=test_mnist,
                                          batch_size=batch_size,
                                          shuffle=True)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=Input_size,   ## 数据的输入维数
            hidden_size=Hidden_size, ## 隐藏层的神经元个数
            num_layers=1,            ## 隐藏层的层数
            batch_first=True         ## 是否以batch_size作为第一个维度[batch_size, time_step, input_size]
        )
        self.fc = nn.Linear(Hidden_size, Output_size)

    def forward(self, input):
        r_out, _ = self.rnn(input, None)
        out = self.fc(r_out[:, -1, :])
        return out


rnn = RNN()
# print(rnn)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr=lr)



## 训练
Acc_list = []
Loss_list= []
for epoch in range(epoches):
    len_loader = len(train_loader)
    Loss, Acc = 0, 0
    for step, (batch_x, batch_y) in enumerate(train_loader):
        # print(batch_x.shape)##数据是1*28*28的size
        batch_x = batch_x.view(-1, 28, 28)                      ## rnn的输入为[batch_size, 28, 28]
        batch_x, batch_y = Variable(batch_x), Variable(batch_y) # 将tensor变成变量的形式， 保留梯度
        output = rnn(batch_x)                                   # 前向传播
        # print(output)

        _, y_predict = torch.max(output, 1)                     #取output每行的最大值, 并返回最大值所对的索引
        # print(y_predict)
        acc = np.sum(np.array(batch_y)==np.array(y_predict))/len(y_predict)
        # print(acc)
        loss = criterion(output, batch_y)                       # 计算损失

        optimizer.zero_grad()                                   # 梯度归0, 防止逆向传播过程中梯度的累积导致梯度爆炸
        loss.backward()                                         # 梯度逆向传播
        optimizer.step()                                        # 更新参数

        Acc += acc                                              # 累积精度
        Loss += loss.item()
        if (step + 1) % 100 == 0:                               #每100步打印一次结果
            print('Epoch [{}/{}] Step:{}, Loss: {:.2f}, ACC:{:.2f}'
                  .format(epoch + 1, epoches, step+1, loss.item(), acc))
    print('Epoch [{}/{}], 平均损失: {:.2f}, 平均精度:{:.2f}'
                .format(epoch + 1, epoches, Loss/len_loader, Acc/len_loader))
    Acc_list.append(Acc/len_loader)
    Loss_list.append(Loss/len_loader)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121)
ax1.plot(Loss_list)
ax1.set_xlabel('epoch')
ax1.set_ylabel('Loss')

ax2 = fig.add_subplot(122)
ax2.plot(Acc_list)
ax2.set_xlabel('epoch')
ax2.set_ylabel('Acc')

fig.show()


## 测试
with torch.no_grad():
    correct = 0
    total = 0
    for step, (batch_x, batch_y) in enumerate(test_loader):
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        batch_x = batch_x.view(-1, 28, 28)
        output = rnn(batch_x)
        _, predicted = torch.max(output, 1)
        total += batch_y.size(0)
        correct += np.sum(np.array(batch_y)==np.array(predicted))

    print("The accuracy of total {} images: {}%".format(total, 100 * correct / total))

