import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader



# 自定义Dataset的子类
class MyDataset(Dataset):
    # 构造器初始化方法
    def __init__(self, data_X, data_y):
        self.data_X = data_X
        self.data_y = data_y

    # 重写getitem方法用于通过idx获取数据内容
    def __getitem__(self, idx):
        return self.data_X[idx], self.data_y[idx]

    # 重写len方法获取数据集大小
    def __len__(self):
        return self.data_X.shape[0]

if __name__ == '__main__':
    data_X = np.random.randn(100, 5)
    data_y = 3 * data_X + 5

    # 构造Dataset对象
    dataset = MyDataset(data_X, data_y)
    # 构造DataLoader对象
    dataloader = DataLoader(dataset, batch_size=16, num_workers=0, shuffle=True, drop_last=False)

    for batch_X, batch_y in dataloader:
        print(batch_X.shape, batch_y.shape)
