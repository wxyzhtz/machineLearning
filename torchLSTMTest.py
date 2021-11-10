# Setup
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
batch_size=64
torch.manual_seed(1)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Mnist手写数字
train_data = dsets.MNIST(root='./mnist/',    # 保存或者提取位置
                         train=True,  # this is tra`ining data
                         transform=transforms.ToTensor(),    # 转换 PIL.Image or numpy.ndarray 成
                                                    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
                         download=True,          # 没下载就下载, 下载了就不用再下了改成False
)

test_data = dsets.MNIST(root='./mnist/',
                        train=False,
                        transform=transforms.ToTensor())

# Dataloader
# PyTorch中数据读取的一个重要接口，该接口定义在dataloader.py中，只要是用PyTorch来训练模型基本都会用到该接口（除非用户重写…），
# 该接口的目的：将自定义的Dataset根据batch size大小、是否shuffle等封装成一个Batch Size大小的Tensor，用于后面的训练。
train_loader = DataLoader(dataset=train_data,
                          batch_size=batch_size,
                          shuffle=True) # 在每个epoch开始的时候，对数据重新打乱进行训练。在这里其实没啥用，因为只训练了一次

test_loader = DataLoader(dataset=test_data,
                         batch_size=batch_size,
                         shuffle=False)

# LSTM
# __init__ is basically a function which will "initialize"/"activate" the properties of the class for a specific object
# self represents that object which will inherit those properties
class simpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(simpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        # 初始化hidden和memory cell参数
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # forward propagate lstm
        out, (h_n, h_c) = self.lstm(x, (h0, c0))

        # 选取最后一个时刻的输出
        out = self.fc(out[:, -1, :])
        return out

# train the model
# 关于reshape(-1)的解释 https://www.zhihu.com/question/52684594
# view()和reshape()区别的解释 https://stackoverflow.com/questions/49643225/whats-the-difference-between-reshape-and-view-in-pytorch

# Hyper Parameters
epochs = 1           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
batch_size = 64
time_step = 28      # rnn 时间步数 / 图片高度
input_size = 28     # rnn 每步输入值 / 图片每行像素
hidden_size = 64
num_layers = 1
num_classes = 10
lr = 0.01           # learning rate

model = simpleLSTM(input_size, hidden_size, num_layers, num_classes)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr)

total_step = len(train_loader)
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, time_step, input_size).to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, epochs, i+1, total_step, loss.item()))


# Test the model
# https://stackoverflow.com/questions/55627780/evaluating-pytorch-models-with-torch-no-grad-vs-model-eval
# torch.max()用法。https://blog.csdn.net/weixin_43255962/article/details/84402586
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, time_step, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))