import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from matplotlib import pyplot as plt
import pandas as pd
import random
from pylab import mpl, plt
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'
from tools import toolkits
import os
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from torch.utils.data import Dataset  # Dataset是个抽象类，只能用于继承
from torch.utils.data import DataLoader # DataLoader需实例化，用于加载数据

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

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, taskType='classification'):
        super(LSTM, self).__init__()
        self.taskType = taskType
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        # self.sigmoid = nn.Sigmoid()
        # self.act = nn.Softmax()

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        # out = self.act(out)
        # predictions = self.sigmoid(out)
        if self.taskType == 'classification':
            predictions = F.softmax(out, dim=1)
        elif self.taskType == 'regression':
            predictions = out.squeeze(-1)

        return predictions


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

def evaluate_accuracy(x,y,net):
    with torch.no_grad():
        out = net(x)
        correct= (out.argmax(1) == y).sum().item()
        n = y.shape[0]
    return round(correct / n * 100, 2)

def evaluate_loss(x,y,net, lossFn):
    with torch.no_grad():
        out = net(x)
        loss = lossFn(out, y).item()
    return loss

def getLabel(rawDF):
    rawDF['return_bin'] = np.NaN
    upStocks = rawDF.ret[rawDF.ret>0]
    downStocks = rawDF.ret[rawDF.ret<0]

    commonLength = min(len(upStocks), len(downStocks))
    upStocks = upStocks[upStocks.rank(ascending=False)<=commonLength]
    downStocks = downStocks[downStocks.rank()<=commonLength]

    topUpStocks = upStocks[upStocks.rank(pct=True)>0.4]
    topDownStocks = downStocks[downStocks.rank(pct=True)<0.6]
    rawDF.loc[topUpStocks.index.tolist(), 'return_bin'] = 1
    rawDF.loc[topDownStocks.index.tolist(), 'return_bin'] = 0
    rawDF.dropna(inplace=True)

    return rawDF

def getLabel10(rawDF):
    rawDF['return_bin'] = np.NaN
    upStocks = rawDF.ret[rawDF.ret>0]
    downStocks = rawDF.ret[rawDF.ret<0]

    commonLength = min(len(upStocks), len(downStocks))
    upStocks = upStocks[upStocks.rank(ascending=False)<=commonLength]
    downStocks = downStocks[downStocks.rank()<=commonLength]

    topUpStocks = upStocks[upStocks.rank(pct=True)>0.4]
    topDownStocks = downStocks[downStocks.rank(pct=True)<0.6]
    rawDF.loc[topUpStocks.index.tolist(), 'return_bin'] = topUpStocks
    rawDF.loc[topDownStocks.index.tolist(), 'return_bin'] = topDownStocks
    rawDF['return_bin'] = (np.floor(abs((rawDF['return_bin'].rank(pct=True)-0.0000000001))*10)).astype(int)
    rawDF.dropna(inplace=True)
    return rawDF

def getLabelR(rawDF):
    rawDF['return_bin'] = np.NaN
    upStocks = rawDF.ret[rawDF.ret>0]
    downStocks = rawDF.ret[rawDF.ret<0]

    commonLength = min(len(upStocks), len(downStocks))
    upStocks = upStocks[upStocks.rank(ascending=False)<=commonLength]
    downStocks = downStocks[downStocks.rank()<=commonLength]

    topUpStocks = upStocks[upStocks.rank(pct=True)>0.4]
    topDownStocks = downStocks[downStocks.rank(pct=True)<0.6]
    rawDF.loc[topUpStocks.index.tolist(), 'return_bin'] = topUpStocks
    rawDF.loc[topDownStocks.index.tolist(), 'return_bin'] = topDownStocks
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

def TSStack(DF, rolling_num):
    newDict = {}

    for index, row in DF.iterrows():
        tmpDF = DF.loc[:index, :]
        if len(tmpDF) >= rolling_num:
            totalRollingDF = tmpDF.iloc[-rolling_num:, :]
            totalRollingDF.dropna(inplace=True, axis=1)
            date = totalRollingDF.index[-1]
            crossSectionDict = {}
            for col in totalRollingDF.columns:
                crossSectionDict[col] = totalRollingDF.loc[:, col].tolist()
            crossSectionSeries = pd.Series(crossSectionDict)
            newDict[date] = crossSectionSeries
    newDF = pd.DataFrame(newDict)
    newDF_NPArray = newDF.values
    print()

def getSampleDataTimeSeries(filePath, date_in_sample, targetDF, targetDates, factorList, benchmark, fileType='csv'):
    data_in_sample = []
    for date in date_in_sample:
        if fileType=='csv':
            fileName = filePath + '\\' + date + '.csv'
            dataDaily = pd.read_csv(fileName, index_col=0)
        elif fileType=='pkl':
            fileName = filePath + '\\' + date + '.pkl'
            dataDaily = pd.read_pickle(fileName, compression='gzip')
        else:
            print('No file type detected')
        dataDaily = dataDaily.loc[:,
                    ['status'] + factorList]

        dataDaily = dataDaily.loc[dataDaily.status != 0, :]
        if date == date_in_sample[0]:
            stockList = dataDaily.index.tolist()
        dataDaily = dataDaily.loc[stockList, :]

        dataDaily.replace(np.inf, np.NaN, inplace=True)
        dataDaily.fillna(dataDaily.mean(), inplace=True)
        ret = targetDF.loc[targetDates[date], :]/targetDF.loc[date, :] - 1
        benchmark_ret = benchmark.loc[targetDates[date], 'closePrice']/benchmark.loc[date, 'closePrice'] - 1
        dataDaily['ret'] = ret.loc[list(map(lambda x:x[:6], dataDaily.index.tolist()))].tolist() - benchmark_ret
        dataDaily.dropna(inplace=True)
        dataDaily = getLabel10(dataDaily)
        dataDaily['date'] = date
        data_in_sample.append(dataDaily)
    data_in_sampleDF = pd.concat(data_in_sample)
    data_in_sampleDF.dropna(inplace=True)
    return data_in_sampleDF

def getTestDataV1(filePath, date_in_test, factorList, fileType='csv'):
    if fileType == 'csv':
        fileName = filePath + '\\' + date_in_test[0] + '.csv'
        dataDaily = pd.read_csv(fileName, index_col=0)
    elif fileType == 'pkl':
        fileName = filePath + '\\' + date_in_test[0] + '.pkl'
        dataDaily = pd.read_pickle(fileName, compression='gzip')
    else:
        print('No file type detected')

    dataDaily = dataDaily.loc[:,
                ['status']+factorList]
    dataDaily = dataDaily.loc[dataDaily.status != 0, :]
    dataDaily.replace(np.inf, np.NaN, inplace=True)
    dataDaily.loc[dataDaily.count(axis=1) < 2] = np.NaN
    dataDaily.dropna(how='all', inplace=True)
    dataDaily.fillna(dataDaily.mean(), inplace=True)
    dataDaily.dropna(inplace=True)
    return dataDaily

if __name__ == '__main__':
    fctType = '技术指标'
    fctInfo = pd.read_excel('F:\\research\\多因子\\机器学习\\结果统计.xlsx', sheet_name='筛选后优矿因子')
    factorList = fctInfo.loc[fctInfo['大类']==fctType, '具体因子'].tolist()
    factorList = fctInfo.loc[:, '具体因子'].tolist()

    # factorList = ['CallAuction_Buy1_2017_2020_pctMean',
    #               'Amt_Rank_Factor_DIF_120',
    #               'net_buy_pct_Share_N_Mean_Neu2',
    #               'Updated_Q_netprofit_Q_trend',
    #               'rating_instnum',
    #               'OpenAuctionVolFactor',
    #               'bk_AmtPerTrade_20172020_Netflow_monthly_10',
    #               'UID_Factor',
    #               'val_petohist60_bk',
    #               'report_appointDate_Factor_1',
    #               'west_netprofit_FTM_ROC',
    #               'Amt_Rank_Factor_n_ev_UID_PriceBasis',
    #               'RetSkew',
    #               'Pricebasis4_rolling_neu_Swing',
    #               'BET_Factor']

    closePrice=toolkits().readRawData('closePrice', 'database\\tradingData')
    benchmark = pd.read_csv('database\\tradingData\\SH000905.csv', index_col=0)
    method = 'SVM'
    filePath = 'database\\fctValue'
    percent_select = [0.3, 0.3]
    percent_cv = 0.2
    target = closePrice
    factorDict = {}
    factorLinearDict = {}
    factorNonLinearDict = {}
    stat = {}
    stockCodes = pd.read_csv('database\\stockCode\\stock_code.csv', index_col=0)['代码'].tolist()
    stockTickers = list(map(lambda x: x[-6:] + '.' + x[:2], stockCodes))
    # fctDates = list(map(lambda x:x[:-4], os.listdir('database\\fctValue')))
    tradingDates = toolkits().getTradingPeriod('2018-01-01', '2021-09-24')
    tradingDates = pd.Series(toolkits().cutTimeSeries(tradingDates), index=toolkits().cutTimeSeries(tradingDates))[26:]

    input_size = len(factorList)
    hidden_size = 200
    num_layers = 2
    output_dim = 10
    epochs = 5
    batch_size = 128
    tim_step = 20


    historyFactorTSInfo = readAllStockTS(tradingDates, factorList)
    #　TSStack(historyFactorTSInfo, 15)
    fctValueDict = {}
    scaler = MinMaxScaler(feature_range=(-1, 1))
    cvDict = {}

    trainingPeriod = 10
    for tradingDate in tradingDates[trainingPeriod+25:]:
        # 载入模型
        model = LSTM(input_dim=input_size, hidden_dim=hidden_size, output_dim=output_dim, num_layers=num_layers, taskType='classification')
        # 载入优化器
        optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
        # 载入损失函数
        loss_fn = torch.nn.CrossEntropyLoss() # 分类
        # loss_fn = torch.nn.MSELoss()  # 回归
        # 初始最佳准确率为0
        best_acc = 0
        print('{} starts.'.format(tradingDate))
        # 训练日(当前交易日前 1-X个因子计算日)
        trainingDates = tradingDates[:tradingDate].iloc[-(trainingPeriod+1):-1]
        # 预测目标日(因子计算日的收益结算日)
        targetDates = tradingDates[:tradingDate].iloc[-trainingPeriod:]
        # 测试日(当前交易日)
        testDate = tradingDate
        # 载入训练日因子数据
        #
        dataBagTrainX = []
        dataBagTrainY = []
        for trainingDate in trainingDates:
            lastPeriods = tradingDates[:tradingDate].iloc[-10:]
            historyFactorTSInfoTrainSelected = historyFactorTSInfo.loc[lastPeriods, :].dropna(axis=1)
            historyFactorTSInfoTrainSelected.columns = list(
                map(lambda x: x[:6], historyFactorTSInfoTrainSelected.columns.tolist()))
            historyFactorTSInfoTrainSelected_col = historyFactorTSInfoTrainSelected.columns.tolist()
            targetDate = tradingDates.shift(-1).loc[trainingDate]
            ret = closePrice.loc[targetDate, historyFactorTSInfoTrainSelected_col] / closePrice.loc[trainingDate, historyFactorTSInfoTrainSelected_col] - \
              benchmark.loc[targetDate, 'closePrice'] / benchmark.loc[trainingDate, 'closePrice']
            ret_rank = ret.rank(pct=True)
            ret[ret_rank >= 0.7] = ret[ret_rank >= 0.7]
            ret[ret_rank <= 0.3] = ret[ret_rank <= 0.3]
            ret[(ret_rank > 0.3) & (ret_rank < 0.7)] = np.NaN
            ret.dropna(inplace=True)
            ret = (np.floor(abs((ret.rank(pct=True) - 0.0000000001)) * 10)).astype(int)

            historyFactorTSInfoTrainSelected = historyFactorTSInfoTrainSelected.loc[:, ret.index.tolist()]
            for col in historyFactorTSInfoTrainSelected.columns.tolist():
                dataBagTrainX.append(historyFactorTSInfoTrainSelected.loc[:, col].tolist())
                dataBagTrainY.append(ret.loc[col])

        X_Train = np.array(dataBagTrainX)
        y_Train = np.array(dataBagTrainY)
        print(X_Train.shape)
        # 载入测试日因子数据
        historyFactorTSInfoTestSelected = historyFactorTSInfo.loc[targetDates, :].dropna(axis=1)

        dataBagTest = []
        for stockTicker in historyFactorTSInfoTestSelected.columns:
            dataBagTest.append(historyFactorTSInfoTestSelected.loc[:, stockTicker].tolist())

        X_Test = np.array(dataBagTest)

        X_train, X_cv, y_train, y_cv = train_test_split(X_Train, y_Train, test_size=percent_cv,
                                                        random_state=42)

        x_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)
        x_cv = torch.FloatTensor(X_cv)
        y_cv = torch.LongTensor(y_cv)
        x_test = torch.FloatTensor(X_Test)
        testStockList = historyFactorTSInfoTestSelected.columns.tolist()

        train_dataset = GetData(X=X_train, y=y_train)
        train_loader = DataLoader(dataset=train_dataset,  # 要传递的数据集
                                  batch_size=batch_size,  # 一个小批量数据的大小是多少
                                  shuffle=True,  # 数据集顺序是否要打乱，一般是要的。测试数据集一般没必要
                                  num_workers=1)  # 需要几个进程来一次性读取这个小批量数据

        hist = np.zeros(epochs)
        hist_cv = np.zeros(epochs)
        lossTmp = 100000
        for t in range(epochs):
            for i, data in enumerate(train_loader, 0):
                # Initialise hidden state
                # Don't do this if you want your LSTM to be stateful
                # model.hidden = model.init_hidden()

                # Forward pass
                optimiser.zero_grad()
                x_train, y_train = data
                x_train = torch.tensor(x_train, dtype=torch.float32)
                y_train_pred = model(x_train)
                # print(y_train_pred.size(), y_train.size(), y_train_pred.squeeze().size(), y_train.squeeze().size())
                # print(y_train_pred.squeeze(), y_train.squeeze())
                loss = loss_fn(y_train_pred, y_train)
                # Backward pass
                loss.backward()
                # Update parameters
                optimiser.step()
                # 模型评价
                cvLoss = evaluate_loss(x_cv, y_cv, model, loss_fn)

                #accuracyRateTrain = evaluate_accuracy(x_train, y_train, model)
                accuracyRateCV = evaluate_accuracy(x_cv, y_cv, model)
                # print('Epoch {} train loss: {} cv loss: {}　cv acc: {}%'.
                #       format(t, loss.item(), cvLoss, accuracyRateCV))
                hist[t] = loss.item()
                if cvLoss<lossTmp:
                    lossTmp = cvLoss
                    print("save model, the cv_acc is {}%".format(accuracyRateCV))
                    torch.save(model.state_dict(), 'modelLSTM.pth')
            if t % 1 == 0 and t != 0:
                # print('Epoch {} loss: {} Accuracy: {}% cv_Accuracy: {}%'.
                #       format(t, loss.item(), accuracyRateTrain, accuracyRateCV) )
                print('Epoch {} train loss: {} cv loss: {}　cv acc: {}%'.
                      format(t, loss.item(), cvLoss, accuracyRateCV))


        with torch.no_grad():
            model = LSTM(input_dim=input_size, hidden_dim=hidden_size, output_dim=output_dim, num_layers=num_layers, taskType='classification')
            model.load_state_dict(torch.load('modelLSTM.pth'))
            y_test_pred = model(x_test)
            cvDict[tradingDate] = lossTmp
            # testLoss = evaluate_loss(x_test, y_test, model, loss_fn)

            # y_cv_pred = model(x_cv)
            cvLoss = evaluate_loss(x_cv, y_cv, model, loss_fn)

            # randomAcc = np.sum(y_test.detach().numpy())/len(y_test.detach().numpy())
            # print(cvAcc)
            print("The selected model cv_loss is {}".format(cvLoss))
        fctValue = pd.Series(dict(zip(testStockList, (np.dot(y_test_pred.detach().numpy(), list(np.array([[1], [2], [3],
                                                                                                    [4], [5], [6],
                                                                                                    [7], [8], [9],
                                                                                                    [10]]))).flatten()) )))
        fctValueDict[tradingDate] = fctValue
        plt.figure()
        plt.plot(hist, label="Training loss")
        plt.plot(hist_cv, label="CV loss")
        plt.legend()
        plt.savefig('results\\lstm\\pic\\' + tradingDate + '.png')

    fct = pd.DataFrame(fctValueDict).T
    fct.to_csv('results\\lstm\\' + fctType  + 'lstm5.csv')
    trainingStat = pd.Series(cvDict).T
    # trainingStat.to_excel('results\\lstm\\stat\\' + fctType  + 'lstm25.csv')
    print(fct)








