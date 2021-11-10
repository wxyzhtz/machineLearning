import torch
import numpy as np
import matplotlib.pylab as plt
import sys
sys.path.append("..")
import main
from tools import toolkits
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from selfDataLoader import MyDataset
from sklearn.model_selection import train_test_split
import pandas as pd


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=5, out_features=4)
        self.fc2 = nn.Linear(in_features=4, out_features=4)
        self.output = nn.Linear(in_features=4, out_features=2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)

        return x

if __name__ == '__main__':

    factor = pd.read_csv('database\\fctValue\\2018-01-18.csv', index_col=0)
    factorList = factor.columns[6:].tolist()
    # factorList = ['net_buy_pct_PeriodsRet_ewma_2H']

    tradingDates = toolkits().getTradingPeriod('2018-04-01', '2021-08-20', )
    filePath = 'database\\fctValue'
    epochs = 200
    for tradingDate in tradingDates:
        print(tradingDate)
        trainingStart = toolkits().getXTradedate(tradingDate, 20)
        trainingEnd = toolkits().getXTradedate(tradingDate, 1)

        date_in_sample = toolkits().getTradingPeriod(trainingStart, trainingEnd)
        date_test = toolkits().getTradingPeriod(tradingDate, tradingDate)
        sampleDataParams = {
            'filePath': filePath,
            'date_in_sample': date_in_sample,
            'percent_select':[0.3, 0.3],
            'factorList': factorList
        }
        testDataParams = {
            'filePath': filePath,
            'date_in_test': date_test,
            'factorList': factorList
        }
        sampleData = toolkits().getSampleData(**sampleDataParams)
        csData = toolkits().getTestData(**testDataParams)
        csX = csData.loc[:, factorList].values
        csX_Tensor = torch.FloatTensor(csX)

        X = sampleData.loc[:, factorList].values
        y = sampleData.loc[:, 'return_bin'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)

        model = ANN()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_arr = []
        for i in range(epochs):
            y_hat = model.forward(X_train)
            loss = criterion(y_hat, y_train)
            loss_arr.append(loss)

            if i % 10 == 0:
                print(f'Epoch: {i} Loss: {loss}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        preds_train = []

        with torch.no_grad():
            for val in X_test:
                y_hat = model.forward(val)
                # print(y_hat)
                y_hat = F.softmax(y_hat)
                preds_train.append((y_hat.argmax().item()))
        accuracyList = [1 if i==j else 0 for i, j in zip(preds_train, y_test.tolist()) ]
        accuracyRate = np.sum(accuracyList)/(len(accuracyList))
        print('train accuracy: ', accuracyRate)

        preds = []
        preds_test = []

        with torch.no_grad():
            for val in csX_Tensor:
                y_hat = model.forward(val)
                # print(y_hat)
                y_hat = F.softmax(y_hat)
                preds.append(y_hat.max().item()*((y_hat.argmax().item() - 1)*2+1))
                preds_test.append((y_hat.argmax().item()))
        accuracyList = [1 if i == j else 0 for i, j in zip(preds_test, y_test.tolist())]
        accuracyRate = np.sum(accuracyList) / (len(accuracyList))
        # print('test accuracy: ', accuracyRate)

        df = pd.DataFrame({'code': csData.index.tolist(), 'YHat': preds, '5dayFutureRetTWAP':csData.loc[:, '5dayFutureRetTWAP'].tolist(), 'acc':accuracyList})
        df.sort_values(by='YHat')
        df.to_csv('results\\nn\\nn_' + tradingDate + '.csv')