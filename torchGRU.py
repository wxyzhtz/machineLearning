import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tools import toolkits
from torch.utils.data import Dataset  # Dataset是个抽象类，只能用于继承
from torch.utils.data import DataLoader # DataLoader需实例化，用于加载数据
import ML_API
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split

class GetData(Dataset):  # 继承Dataset类
    def __init__(self, X, y):
        # 把数据和标签拿出来
        self.data = X
        self.label = y
        # 数据集的长度
        self.length = self.data.shape[0]

    # 下面两个魔术方法比较好写，直接照着这个格式写就行了
    def __getitem__(self, index):  # 参数index必写
        return self.data[index], self.label[index]

    def __len__(self):
        return self.length  # 只需返回一个数据集长度即可

class GRURegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRURegressor, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.8)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        # self.sigmoid = nn.Sigmoid()
        # self.act = nn.Softmax()

    def forward(self, x):
        output, h = self.gru(x, None)
        output = self.fc(output)
        output = output[:, -1, :]
        return output

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_dim)).zero_()
        return hidden

def getLabelRegr(rawDF, upLimit=0.7, downLimit=0.3):
    rawDF['return_bin'] = np.NaN
    upStocks = rawDF[rawDF.ret>0]
    downStocks = rawDF[rawDF.ret<0]

    commonLength = min(len(upStocks), len(downStocks))
    upStocks = upStocks[upStocks.ret.rank(ascending=False)<=commonLength]
    downStocks = downStocks[downStocks.ret.rank()<=commonLength]

    topUpStocks = upStocks[upStocks.ret.rank(pct=True)>upLimit]
    topDownStocks = downStocks[downStocks.ret.rank(pct=True)<downLimit]
    rawDF = pd.concat([topUpStocks, topDownStocks])
    rawDF['return_bin'] = rawDF.ret.tolist()
    rawDF.replace(np.inf, np.NaN, inplace=True)
    rawDF.dropna(inplace=True)
    return rawDF

def readAllStockTS(fctDates, fctNames):
    '''
    :param fctDates: 因子计算日
    :param fctNames: 因子名称
    :return: index: 日期 column: 股票代码 cell: 当日全部因子值, 数据结构为list
    '''
    stockCodes = pd.read_csv('database\\stockCode\\stock_code.csv', index_col=0)['代码'].tolist()
    stockTickers = list(map(lambda x: x[-6:] + '.' + x[:2], stockCodes))
    fctDict = {}
    for fctDate in fctDates:
        print('Read {} data'.format(fctDate))
        fctValues = pd.read_pickle('database\\fctValue\\' + fctDate + '.pkl', compression='gzip')
        fctValues = fctValues.loc[fctValues.status!=0, :]
        fctValues.replace(np.inf, np.NaN, inplace=True)
        fctValues.fillna(fctValues.mean(), inplace=True)
        fctDict[fctDate] = {}
        for stockTicker in stockTickers:
            if stockTicker in fctValues.index:
                fctDict[fctDate][stockTicker] = fctValues.loc[stockTicker, fctNames].tolist()
    stockFactorTSDF = pd.DataFrame(fctDict)
    return stockFactorTSDF.T

def evaluate_loss(x,y,net, lossFn):
    with torch.no_grad():
        out = net(x)

        loss = lossFn(out, y).item()
    return loss

if __name__ == '__main__':
    fctType = '技术指标'
    fctInfo = pd.read_excel('F:\\research\\多因子\\机器学习\\结果统计.xlsx', sheet_name='筛选后优矿因子')
    factorList = fctInfo.loc[fctInfo['大类']==fctType, '具体因子'].tolist()
    # factorList = fctInfo.loc[:, '具体因子'].tolist()
    closePrice = toolkits().readRawData('closePrice', 'database\\tradingData')
    lowPrice = toolkits().readRawData('lowPrice', 'database\\tradingData')
    highPrice = toolkits().readRawData('highPrice', 'database\\tradingData')
    dailyHL = abs(highPrice - lowPrice) / (highPrice + lowPrice)
    epochs = 100
    dailySTD = toolkits().readRawData('std_1m_daily', 'database\\tradingData')

    benchmark = pd.read_csv('database\\tradingData\\SH000905.csv', index_col=0)
    tradingDates = toolkits().getTradingPeriod('2017-01-01', '2021-09-24')
    filePath = 'database\\fctValue'
    target = closePrice
    factorDict = {}
    factorLinearDict = {}
    factorNonLinearDict = {}
    stat = {}
    tradingDates = pd.Series(toolkits().cutTimeSeries(tradingDates, freq='week'),
                             index=toolkits().cutTimeSeries(tradingDates, freq='week'))[-50:]
    historyFactorTSInfo = readAllStockTS(tradingDates, factorList)
    historyFactorTSInfo.dropna(axis=0, how='all', inplace=True)
    historyXPeriods = 10
    crossSectionNum = 4
    for tradingDate in tradingDates[historyXPeriods*crossSectionNum+1:]:
        a = tradingDates[:tradingDate].iloc[-(historyXPeriods*crossSectionNum+1):]
        trainingDates = tradingDates[:tradingDate].iloc[-(historyXPeriods*crossSectionNum + 1):-1].iloc[historyXPeriods-1::historyXPeriods]
        # trainingDates_1 = tradingDates[:tradingDate].iloc[-(historyXPeriods*crossSectionNum + 1 ):-1]
        targetDates = tradingDates.shift(-1)[trainingDates]
        dataBagTrainX = []
        dataBagTrainY = []
        model = GRURegressor(input_dim=len(factorList), hidden_dim=50, num_layers=2, output_dim=1, )
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
        # 载入损失函数
        loss_fn = torch.nn.MSELoss()
        for trainingDate in trainingDates:
            sampleTrainingPeriod = tradingDates[:trainingDate][-historyXPeriods:]
            X_train_tmp = historyFactorTSInfo.loc[sampleTrainingPeriod, :]
            X_train_tmp.dropna(axis=1, inplace=True)
            X_train_tmp.columns = list(map(lambda x: x[:6], X_train_tmp.columns.tolist()))
            targetDate = targetDates[trainingDate]
            ret = closePrice.loc[targetDate, X_train_tmp.columns.tolist()] / closePrice.loc[
                trainingDate, X_train_tmp.columns.tolist()] - \
                  benchmark.loc[targetDate, 'closePrice'] / benchmark.loc[trainingDate, 'closePrice']
            ret = (ret - ret.mean()) / ret.std()
            ret[ret.rank(pct=True) > 0.99] = ret.quantile(0.99)
            ret[ret.rank(pct=True) < 0.01] = ret.quantile(0.01)
            for col in X_train_tmp.columns.tolist():
                dataBagTrainX.append(X_train_tmp.loc[:, col].tolist())
                dataBagTrainY.append(ret.loc[col])
        tmpDict = {}
        tmpDict['X'] = dataBagTrainX
        tmpDict['ret'] = dataBagTrainY


        tmpDataset = pd.DataFrame(tmpDict)
        tmpDataset.dropna(inplace=True)
        tmpDatasetLabeled = getLabelRegr(tmpDataset)
        print(tmpDatasetLabeled)

        X_sample = np.array([v for i, v in tmpDatasetLabeled.loc[:, 'X'].items()])

        y_sample = np.array(tmpDatasetLabeled.loc[:, ['ret']])
        X_train, X_cv, y_train, y_cv = train_test_split(X_sample,
                                                        y_sample,
                                                        test_size=0.2,
                                                        random_state=42)
        print(X_train.shape)
        # X_train = torch.FloatTensor(X_train)
        X_cv = torch.FloatTensor(X_cv)
        # y_train = torch.FloatTensor(y_train)
        y_cv = torch.FloatTensor(y_cv)

        train_dataset = GetData(X=X_train, y=y_train)

        train_loader = DataLoader(dataset=train_dataset,  # 要传递的数据集
                                  batch_size=2048,  # 一个小批量数据的大小是多少
                                  shuffle=True,  # 数据集顺序是否要打乱，一般是要的。测试数据集一般没必要
                                  num_workers=1)  # 需要几个进程来一次性读取这个小批量数据
        lossTmp = 100000
        for t in range(epochs):
            # h = model.init_hidden(32)
            for i, data in enumerate(train_loader, 0):
                # Initialise hidden state
                # Don't do this if you want your LSTM to be stateful
                # model.hidden = model.init_hidden()

                # Forward pass
                optimiser.zero_grad()
                x_train, y_train = data
                x_train = torch.tensor(x_train, dtype=torch.float32)
                y_train = torch.tensor(y_train, dtype=torch.float32)

                y_train_pred = model(x_train)
                # print(y_train_pred,y_train )
                # print(y_train_pred.size(), y_train.size(), y_train_pred.squeeze().size(), y_train.squeeze().size())
                # print(y_train_pred.squeeze(), y_train.squeeze())
                loss = loss_fn(y_train_pred, y_train)
                # Backward pass
                loss.backward()
                # Update parameters
                optimiser.step()
                # 模型评价
                cvLoss = evaluate_loss(X_cv, y_cv, model, loss_fn)

                # accuracyRateTrain = evaluate_accuracy(x_train, y_train, model)

                print('Epoch {} train loss: {} cv loss: {}'.
                      format(t, loss.item(), cvLoss))

                if cvLoss < lossTmp:
                    lossTmp = cvLoss
                    print("save model, the cv_loss is {}".format(cvLoss))
                    torch.save(model.state_dict(), 'modelGRU.pth')
            if t % 1 == 0 and t != 0:
                pass
                # print('Epoch {} loss: {} Accuracy: {}% cv_Accuracy: {}%'.
                #       format(t, loss.item(), accuracyRateTrain, accuracyRateCV) )
                # print('Epoch {} train loss: {} cv loss: {}'.
                #       format(t, loss.item(), cvLoss))