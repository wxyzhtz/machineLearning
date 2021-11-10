from torch.utils.data import Dataset  # Dataset是个抽象类，只能用于继承
from torch.utils.data import DataLoader # DataLoader需实例化，用于加载数据
import pandas as pd
import numpy as np
from tools import toolkits
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, auc
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgbm
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.model_selection import GridSearchCV
import time
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectPercentile, chi2, mutual_info_regression, SelectKBest
from xgboost import plot_importance

class XGBCWorkflowAPI:
    def __init__(self, X_sample, y_sample, randomSeed):
        self.X_sample = X_sample
        self.y_sample= y_sample
        self.randomSeed = randomSeed
        self.dataSplit()
        self.trainEngine()

    def dataSplit(self):
        self.X_train, self.X_cv, self.y_train, self.y_cv = train_test_split(self.X_sample, self.y_sample, test_size=0.2, random_state=self.randomSeed)

    def trainEngine(self):
        self.Regr = xgb.XGBClassifier(use_label_encoder=False, n_estimators=1000, max_depth=8, min_child_weight=1,
                                 subsample=0.95,
                                 learning_rate=0.05, objective='binary:logistic', eval_metric='auc',
                                 colsample_bytree=0.8)
        eval_set = [(self.X_cv, self.y_cv)]
        self.Regr.fit(self.X_train, self.y_train, early_stopping_rounds=10, eval_metric='auc', eval_set=eval_set, verbose=False)
        y_pred = self.Regr.predict(self.X_cv)
        CV_accuracy = np.sum([i == j for i, j in zip(y_pred, self.y_cv)]) / len(self.y_cv)
        print('XGBC cv accuracy {}'.format(CV_accuracy))

    def predictResult(self, X):
        groups = np.array(range(2))
        groupsReshape = groups.reshape(2, -1)
        preds = self.Regr.predict_proba(X)  # [:, 1]
        groupPreds = (np.dot(preds, groupsReshape)).flatten()

        return groupPreds

class LGBMCWorkflowAPI:
    def __init__(self, X_sample, y_sample, randomSeed):
        self.X_sample = X_sample
        self.y_sample = y_sample
        self.randomSeed = randomSeed
        self.dataSplit()
        self.trainEngine()

    def dataSplit(self):
        self.X_train, self.X_cv, self.y_train, self.y_cv = train_test_split(self.X_sample, self.y_sample,
                                                                            test_size=0.2,
                                                                            random_state=self.randomSeed)

    def trainEngine(self):
        self.Regr = lgbm.LGBMClassifier(learning_rate=0.01, n_estimators=1000, max_depth=20, random_state=42,
                                       num_leaves=50,
                                       subsample=0.8, )
        eval_set = [(self.X_cv, self.y_cv)]
        self.Regr.fit(self.X_train, self.y_train, eval_set=eval_set, early_stopping_rounds=10, verbose=False)
        y_pred = self.Regr.predict(self.X_cv)
        CV_accuracy = np.sum([i == j for i, j in zip(y_pred, self.y_cv)]) / len(self.y_cv)
        print('LGBMC cv accuracy {}'.format(CV_accuracy))

    def predictResult(self, X):
        groups = np.array(range(2))
        groupsReshape = groups.reshape(2, -1)
        preds = self.Regr.predict_proba(X)  # [:, 1]
        groupPreds = (np.dot(preds, groupsReshape)).flatten()

        return groupPreds


class LOGWorkflowAPI:
    def __init__(self, X_sample, y_sample, randomSeed):
        self.X_sample = X_sample
        self.y_sample = y_sample
        self.randomSeed = randomSeed
        self.dataSplit()
        self.trainEngine()

    def dataSplit(self):
        self.X_train, self.X_cv, self.y_train, self.y_cv = train_test_split(self.X_sample, self.y_sample,
                                                                            test_size=0.2,
                                                                            random_state=self.randomSeed)

    def trainEngine(self):
        self.Regr = LogisticRegression(fit_intercept=False, max_iter=1000)
        self.Regr.fit(self.X_train, self.y_train)
        y_pred = self.Regr.predict(self.X_cv)
        CV_accuracy = np.sum([i == j for i, j in zip(y_pred, self.y_cv)]) / len(self.y_cv)
        print('Logistic cv accuracy {}'.format(CV_accuracy))

    def predictResult(self, X):
        groups = np.array(range(2))
        groupsReshape = groups.reshape(2, -1)
        preds = self.Regr.predict_proba(X)  # [:, 1]
        groupPreds = (np.dot(preds, groupsReshape)).flatten()

        return groupPreds

class RFWorkflowAPI:
    def __init__(self, X_sample, y_sample, randomSeed):
        self.X_sample = X_sample
        self.y_sample = y_sample
        self.randomSeed = randomSeed
        self.dataSplit()
        self.trainEngine()

    def dataSplit(self):
        self.X_train, self.X_cv, self.y_train, self.y_cv = train_test_split(self.X_sample, self.y_sample,
                                                                            test_size=0.2,
                                                                            random_state=self.randomSeed)

    def trainEngine(self):
        self.Regr = RandomForestClassifier(n_estimators=1000, max_depth=10, min_samples_split=4)
        self.Regr.fit(self.X_train, self.y_train)
        y_pred = self.Regr.predict(self.X_cv)
        CV_accuracy = np.sum([i == j for i, j in zip(y_pred, self.y_cv)]) / len(self.y_cv)
        print('RF cv accuracy {}'.format(CV_accuracy))

    def predictResult(self, X):
        groups = np.array(range(2))
        groupsReshape = groups.reshape(2, -1)
        preds = self.Regr.predict_proba(X)  # [:, 1]
        groupPreds = (np.dot(preds, groupsReshape)).flatten()

        return groupPreds

class GBDTWorkflowAPI:
    def __init__(self, X_sample, y_sample, randomSeed):
        self.X_sample = X_sample
        self.y_sample = y_sample
        self.randomSeed = randomSeed
        self.dataSplit()
        self.trainEngine()

    def dataSplit(self):
        self.X_train, self.X_cv, self.y_train, self.y_cv = train_test_split(self.X_sample, self.y_sample,
                                                                            test_size=0.2,
                                                                            random_state=self.randomSeed)

    def trainEngine(self):
        self.Regr = GradientBoostingClassifier(learning_rate=0.01, n_estimators=1000, subsample=0.85)
        self.Regr.fit(self.X_train, self.y_train)
        y_pred = self.Regr.predict(self.X_cv)
        CV_accuracy = np.sum([i == j for i, j in zip(y_pred, self.y_cv)]) / len(self.y_cv)
        print('GBDT cv accuracy {}'.format(CV_accuracy))

    def predictResult(self, X):
        groups = np.array(range(2))
        groupsReshape = groups.reshape(2, -1)
        preds = self.Regr.predict_proba(X)  # [:, 1]
        groupPreds = (np.dot(preds, groupsReshape)).flatten()

        return groupPreds

def getLabel(rawDF, upLimit=0.4, downLimit=0.6):
    rawDF['return_bin'] = np.NaN
    upStocks = rawDF.ret[rawDF.ret>0]
    downStocks = rawDF.ret[rawDF.ret<0]

    commonLength = min(len(upStocks), len(downStocks))

    upStocks = upStocks[upStocks.rank(ascending=False)<=commonLength]
    downStocks = downStocks[downStocks.rank()<=commonLength]

    topUpStocks = upStocks[upStocks.rank(pct=True)>upLimit]
    topDownStocks = downStocks[downStocks.rank(pct=True)<downLimit]
    rawDF.loc[topUpStocks.index.tolist(), 'return_bin'] = 1
    rawDF.loc[topDownStocks.index.tolist(), 'return_bin'] = 0
    # rawDF.loc[topUpStocks.index.tolist(), 'return_bin'] = topUpStocks
    # rawDF.loc[topDownStocks.index.tolist(), 'return_bin'] = topDownStocks
    rawDF.dropna(inplace=True)

    # rawDF['return_bin'] = (np.floor(abs((rawDF['return_bin'].rank(pct=True)-0.0000000001))*10)).astype(int)
    return rawDF

def getLabel10(rawDF, upLimit=0.7, downLimit=0.3):
    rawDF['return_bin'] = np.NaN
    upStocks = rawDF.ret[rawDF.ret>0]
    downStocks = rawDF.ret[rawDF.ret<0]

    commonLength = min(len(upStocks), len(downStocks))
    upStocks = upStocks[upStocks.rank(ascending=False)<=commonLength]
    downStocks = downStocks[downStocks.rank()<=commonLength]

    topUpStocks = upStocks[upStocks.rank(pct=True) > upLimit]
    topDownStocks = downStocks[downStocks.rank(pct=True) < downLimit]
    rawDF.loc[topUpStocks.index.tolist(), 'return_bin'] = topUpStocks
    rawDF.loc[topDownStocks.index.tolist(), 'return_bin'] = topDownStocks
    rawDF.dropna(inplace=True)

    rawDF['return_bin'] = (np.floor(abs((rawDF['return_bin'].rank(pct=True)-0.0000000001))*50)).astype(int)
    return rawDF

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
    rawDF['return_bin'] = ((rawDF.ret - rawDF.ret.mean())/rawDF.ret.std()).tolist()
    rawDF.replace(np.inf, np.NaN, inplace=True)
    rawDF.dropna(inplace=True)
    return rawDF

def getSampleData(filePath, date_in_sample, targetDF, targetDates, factorList, benchmark, dailySTD, fileType='csv'):
    data_in_sample = []
    for date in date_in_sample:
        if fileType=='csv':
            fileName = filePath + '\\' + date + '.csv'
            dataDaily = pd.read_csv(fileName, index_col=0)
        elif fileType=='pkl':
            fileName = filePath + '\\' + date + '.pkl'
            dataDaily = pd.read_pickle(fileName, compression='gzip')

        dataDaily.index = list(map(toolkits().toTicker, dataDaily.index.tolist()))
        dataDaily = dataDaily.loc[:,
                    ['status'] + factorList]
        dataDaily = dataDaily.loc[dataDaily.status != 0, :]
        dataDaily.replace(np.inf, np.NaN, inplace=True)
        dataDaily.replace(0, np.NaN, inplace=True)
        factorNaN = dataDaily.loc[:,factorList].count(axis=1)

        dataDaily.loc[factorNaN<(len(factorList)*0.6), :] = np.NaN

        noAllNaN = dataDaily.loc[:, factorList].dropna(how='all')
        dataDaily = dataDaily.loc[noAllNaN.index.tolist()]
        # dataDaily.loc[:, factorList] = (dataDaily.loc[:, factorList].sub(dataDaily.loc[:, factorList].mean())).div(dataDaily.loc[:, factorList].std())
        dataDaily.loc[:, factorList] = (toolkits().getZScore(dataDaily.loc[:, factorList].T, 1)).T
        dataDaily.fillna(dataDaily.mean(), inplace=True)
        ret = targetDF.loc[targetDates[date], :] / targetDF.loc[date, :] - 1

        periodSTD = dailySTD.loc[toolkits().getXTradedate(date, -1): targetDates[date], :]
        periodSTD_Mean = periodSTD.mean()

        benchmark_ret = benchmark.loc[targetDates[date], 'closePrice'] / benchmark.loc[date, 'closePrice'] - 1
        sameTicker = list(set(dataDaily.index.tolist()).intersection(set(ret.index.tolist())))
        dataDaily = dataDaily.loc[np.in1d(dataDaily.index.tolist(), sameTicker)]
        dataDaily['ret'] = ret.loc[dataDaily.index.tolist()].tolist() - benchmark_ret

        dataDaily = pd.concat([dataDaily, periodSTD_Mean], axis=1)
        # dataDaily.dropna(inplace=True)
        res = dataDaily['ret'] #/dataDaily.iloc[:, -1]
        dataDaily['ret'] = res
        dataDaily = getLabel10(dataDaily)
        data_in_sample.append(dataDaily)
    data_in_sampleDF = pd.concat(data_in_sample)
    data_in_sampleDF.dropna(inplace=True)
    return data_in_sampleDF

def getSampleData_V1(filePath, date_in_sample, targetDF, targetDates, factorList, benchmark, dailySTD, fileType='csv', upLimit=0.7, downLimit=0.3):
    '''
    :param filePath: 文件路径
    :param date_in_sample: 样本内日期
    :param targetDF: 目标文件
    :param targetDates: 目标日期
    :param factorList: 因子列表
    :param benchmark: 基准
    :param dailySTD: 每日波动
    :param fileType: 文件类型
    :return:
    '''
    data_in_sample = []
    for date in date_in_sample:
        if fileType=='csv':
            fileName = filePath + '\\' + date + '.csv'
            dataDaily = pd.read_csv(fileName, index_col=0)
        elif fileType=='pkl':
            fileName = filePath + '\\' + date + '.pkl'
            dataDaily = pd.read_pickle(fileName, compression='gzip')

        dataDaily.index = list(map(toolkits().toTicker, dataDaily.index.tolist()))
        # print(np.array(factorList)[~np.in1d(factorList, dataDaily.columns.tolist())])
        dataDaily = dataDaily.loc[:,
                    ['status'] + factorList]

        dataDaily = dataDaily.loc[dataDaily.status != 0, :]
        dataDaily.replace(np.inf, np.NaN, inplace=True)
        dataDaily.replace(0, np.NaN, inplace=True)
        factorNaN = dataDaily.loc[:,factorList].count(axis=1)

        dataDaily.loc[factorNaN<(len(factorList)*0.6), :] = np.NaN

        noAllNaN = dataDaily.loc[:, factorList].dropna(how='all')
        dataDaily = dataDaily.loc[noAllNaN.index.tolist()]
        # dataDaily.loc[:, factorList] = (dataDaily.loc[:, factorList].sub(dataDaily.loc[:, factorList].mean())).div(dataDaily.loc[:, factorList].std())
        dataDaily.loc[:, factorList] = (toolkits().getZScore(dataDaily.loc[:, factorList].T, 1)).T
        dataDaily.fillna(dataDaily.mean(), inplace=True)
        ret = targetDF.loc[targetDates[date], :] / targetDF.loc[date, :] - 1

        # periodSTD = dailySTD.loc[toolkits().getXTradedate(date, -1): targetDates[date], :]
        # periodSTD.dropna(how='all', inplace=True, axis=1)
        # periodSTD_Mean = periodSTD.mean()

        benchmark_ret = benchmark.loc[targetDates[date], 'closePrice'] / benchmark.loc[date, 'closePrice'] - 1
        sameTicker = list(set(dataDaily.index.tolist()).intersection(set(ret.index.tolist())))
        dataDaily = dataDaily.loc[np.in1d(dataDaily.index.tolist(), sameTicker)]
        dataDaily['ret'] = ret.loc[dataDaily.index.tolist()].tolist() - benchmark_ret

        # dataDaily = pd.concat([dataDaily, periodSTD_Mean], axis=1)
        dataDaily.dropna(inplace=True)
        res = dataDaily['ret'] # /dataDaily.iloc[:, -1]
        res = (res - res.mean()) / res.std()
        res[res.rank(pct=True)>0.99]=res.quantile(0.99)
        res[res.rank(pct=True)<0.01]=res.quantile(0.01)
        dataDaily['ret'] = res
        dataDaily.index = list(map(lambda x: x + '_' + date, dataDaily.index.tolist()))
        data_in_sample.append(dataDaily)
    data_in_sampleDF = pd.concat(data_in_sample)
    data_in_sampleDF = getLabelRegr(data_in_sampleDF, upLimit, downLimit)
    data_in_sampleDF.dropna(inplace=True)
    return data_in_sampleDF

def getTestDataV1(filePath, date_in_test, factorList, fileType='csv'):
    if fileType == 'csv':
        fileName = filePath + '\\' + date_in_test[0] + '.csv'
        dataDaily = pd.read_csv(fileName, index_col=0)
    elif fileType == 'pkl':
        fileName = filePath + '\\' + date_in_test[0] + '.pkl'
        dataDaily = pd.read_pickle(fileName, compression='gzip')
    # fileName = filePath + '\\' + date_in_test[0] + '.csv'
    # dataDaily = pd.read_csv(fileName, index_col=0)
    dataDaily = dataDaily.loc[:,
                ['status']+factorList]
    dataDaily = dataDaily.loc[dataDaily.status != 0, :]
    dataDaily.replace(np.inf, np.NaN, inplace=True)
    dataDaily.replace(0, np.NaN, inplace=True)

    noAllNaN = dataDaily.loc[:, factorList].dropna(how='all')
    dataDaily = dataDaily.loc[noAllNaN.index.tolist()]
    dataDaily.loc[dataDaily.count(axis=1) < 2] = np.NaN
    dataDaily.dropna(how='all', inplace=True)
    dataDaily.loc[:, factorList] = (toolkits().getZScore(dataDaily.loc[:, factorList].T, 1)).T
    dataDaily.fillna(dataDaily.mean(), inplace=True)
    dataDaily.dropna(inplace=True)
    return dataDaily

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

class ANNC(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.fc1 = nn.Linear(in_features=feature_size, out_features=int(feature_size))
        nn.init.xavier_normal_(self.fc1.weight)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=int(feature_size), out_features=int(feature_size/2))

        nn.init.xavier_uniform_(self.fc2.weight)
        self.drop2 = nn.Dropout(0.5)
        self.output = nn.Linear(in_features=int(feature_size/2), out_features=2)

    def forward(self, x):
        x = torch.tanh(self.drop1(self.fc1(x)))
        x = torch.tanh(self.drop2(self.fc2(x)))
        x = self.output(x)
        dout = F.softmax(x, dim=1)
        return dout

class ANNR(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.fc1 = nn.Linear(in_features=feature_size, out_features=int(feature_size*0.75))
        nn.init.xavier_normal_(self.fc1.weight)
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_features=int(feature_size*0.75), out_features=int(feature_size*0.75*0.75))
        nn.init.xavier_uniform_(self.fc2.weight)
        self.drop2 = nn.Dropout(0.2)
        self.output = nn.Linear(in_features=int(feature_size*0.75*0.75), out_features=1)

    def forward(self, x):
        x = torch.tanh(self.drop1(self.fc1(x)))
        x = torch.tanh(self.drop2(self.fc2(x)))
        x = self.output(x)
        return x

def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

def feval_rmspe(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    return 'RMSPE', rmspe(y_true, y_pred), False

def evaluate_accuracy(x,y,net):
    with torch.no_grad():
        out = net(x)
        correct= (out.argmax(1) == y).sum().item()
        n = y.shape[0]
    return correct / n

def evaluate_loss(x,y, net, lossFn):
    with torch.no_grad():
        out = net(x).squeeze(-1)
        loss = lossFn(out, y).item()
    return loss

def evaluate_mae(x,y,net):
    with torch.no_grad():
        out = net(x)
        mae = mean_squared_error(out.argmax(1).detach().numpy(), y.detach().numpy())
    return mae

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

def ANNWorkflow(data_in_sampleDF, data_in_testDF, factorList, targetName):
    epochs = 50

    train = data_in_sampleDF.loc[:, factorList + [targetName]]
    test = data_in_testDF.loc[:, factorList]
    X_test = test.values
    trainPredictors = [x for x in train.columns if x not in [targetName]]

    X_train, X_cv, y_train, y_cv = train_test_split(data_in_sampleDF[trainPredictors].values, data_in_sampleDF[targetName].values,
                                                    test_size=0.2,
                                                    random_state=455)
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # X_train = scaler.fit_transform(X_train)
    # X_cv = scaler.fit_transform(X_cv)
    # X_test = scaler.fit_transform(test.values)

    print(X_train.shape)
    X_train = torch.FloatTensor(X_train)
    X_cv = torch.FloatTensor(X_cv)
    y_train = torch.LongTensor(y_train)
    y_cv = torch.LongTensor(y_cv)
    X_test = torch.FloatTensor(X_test)
    dataset = GetData(X=X_train, y=y_train)
    train_loader = DataLoader(dataset=dataset,  # 要传递的数据集
                              batch_size=2048,
                              shuffle=True, # 数据集顺序是否要打乱，一般是要的。测试数据集一般没必要
                              num_workers=1)  # 需要几个进程来一次性读取这个小批量数据

    model = ANNC(feature_size=len(factorList))
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_arr = []
    best_acc = 0
    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            y_hat = model.forward(inputs)
            loss = criterion(y_hat, labels)
            # print(epoch, i, loss.item())
            loss.backward()
            optimizer.step()
            acc = evaluate_accuracy(X_train, y_train, model)
            acc_cv = evaluate_accuracy(X_cv, y_cv, model)
            # print(f'Epoch: {epoch} batch{i} Loss: {loss}')
            # print('training acc: {}, cv acc: {}'.format(acc, acc_cv))
            if (best_acc < acc_cv) and (epoch>=0):
                best_acc = acc_cv
                print("save model, the cv_acc is {} loss is {}".format(best_acc, loss))
                torch.save(model.state_dict(), 'model_ANNC.pth')

        loss_arr.append(loss)
        #　acc = evaluate_accuracy(X_train, y_train, model)
        #　acc_cv = evaluate_accuracy(X_cv, y_cv, model)
        acc_random_cv = np.sum(y_cv.detach().numpy())/len(y_cv.detach().numpy())
        if epoch % 2 == 0:
            # pass
            cvLoss = evaluate_loss(X_cv, y_cv, model, criterion)
            print(f'Epoch: {epoch} Training Loss: {loss} CV Loss: {cvLoss}')
            print('training acc: {}, cv acc: {} cv random acc:{}'.format(acc, acc_cv, 0.1))


    with torch.no_grad():
        model = ANNC(feature_size=len(factorList))
        model.load_state_dict(torch.load('model_ANNC.pth'))
        y_test_pred = model(X_test)
        # print("The selected model cv_acc is {}%".format(best_acc*100))
        Training_accuracy = evaluate_accuracy(X_train, y_train, model)
        Training_mae = evaluate_mae(X_train, y_train, model)
        cv_accuracy = evaluate_accuracy(X_cv, y_cv, model)
        cv_mae = evaluate_mae(X_cv, y_cv, model)
        # print('Training Accuracy: {} MAE: {}'.format(Training_accuracy, Training_mae))
        # print('Cross-validation Accuracy: {} MAE: {}'.format(cv_accuracy, cv_mae))
        rankArray = np.arange(0, 2).reshape(-1, 1)
        expectY = (np.dot(y_test_pred.detach().numpy(), rankArray)).flatten()
    predSeries = pd.Series(expectY, test.index.tolist())

    return predSeries, Training_accuracy, Training_mae, cv_accuracy, cv_mae

def ANNRWorkflow(data_in_sampleDF, data_in_testDF, factorList, targetName):
    epochs = 100
    scaler = MinMaxScaler(feature_range=(-1, 1))

    train = data_in_sampleDF.loc[:, factorList + [targetName]]

    SP = SelectKBest(mutual_info_regression, k=100)
    SP.fit(data_in_sampleDF[factorList], data_in_sampleDF[targetName])
    selectedFcts = SP.get_feature_names_out(factorList)
    test = data_in_testDF.loc[:, selectedFcts]
    X_test = test.values
    trainPredictors = [x for x in train.columns if x not in [targetName]]


    data_in_sampleDF[selectedFcts] =  scaler.fit_transform(data_in_sampleDF[selectedFcts].values)
    X_train, X_cv, y_train, y_cv = train_test_split(data_in_sampleDF[selectedFcts].values, data_in_sampleDF[targetName].values,
                                                    test_size=0.15,
                                                    random_state=42)
    #　X_train = scaler.fit_transform(X_train)
    #　X_cv = scaler.fit_transform(X_cv)
    #　X_test = scaler.fit_transform(X_test)


    print(X_train.shape)
    X_train = torch.FloatTensor(X_train)
    X_cv = torch.FloatTensor(X_cv)
    y_train = torch.FloatTensor(y_train)
    y_cv = torch.FloatTensor(y_cv)
    X_test = torch.FloatTensor(X_test)
    dataset = GetData(X=X_train, y=y_train)
    train_loader = DataLoader(dataset=dataset,  # 要传递的数据集
                              batch_size=100000,
                              shuffle=True, # 数据集顺序是否要打乱，一般是要的。测试数据集一般没必要
                              num_workers=1)  # 需要几个进程来一次性读取这个小批量数据

    model = ANNR(feature_size=len(selectedFcts))
    # 损失函数
    criterion = RMSELoss
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_arr = []
    best_loss = 100000
    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            # optimizer.zero_grad()
            y_hat = model.forward(inputs).squeeze(-1)
            loss = criterion(y_hat, labels)
            # print(epoch, i, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Epoch: {epoch} batch{i} Loss: {loss}')
            # loss_cv = evaluate_loss(X_cv, y_cv, model, criterion)
            # print(f'Epoch: {epoch} batch{i} Loss: {loss} Loss_cv: {loss_cv}')
            # # print('training acc: {}, cv acc: {}'.format(acc, acc_cv))
            # if (best_loss > loss_cv) and (epoch>=0):
            #     best_loss = loss_cv
            #     # print("save model, the cv_loss is {} loss is {}".format(best_loss, loss_cv))
            #     torch.save(model.state_dict(), 'model_ANNR.pth')

        loss_arr.append(loss)

        if epoch % 2 == 0:
            # pass
            cvLoss = evaluate_loss(X_cv, y_cv, model, criterion)
            print(f'Epoch: {epoch} Training Loss: {loss} CV Loss: {cvLoss}')
            # print('training loss: {}, cv loss: {}'.format(loss, loss_cv))


    with torch.no_grad():
        model = ANNR(feature_size=len(factorList))
        model.load_state_dict(torch.load('model_ANNR.pth'))
        y_test_pred = model(X_test)
        # print("The selected model cv_acc is {}%".format(best_acc*100))
        Training_mae = evaluate_mae(X_train, y_train, model)
        cv_mae = evaluate_mae(X_cv, y_cv, model)
        print('Training MAE: {}'.format(Training_mae))
        print('Cross-validation MAE: {}'.format(cv_mae))
        expectY = y_test_pred.detach().numpy().flatten()
    predSeries = pd.Series(expectY, test.index.tolist())

    return predSeries, Training_mae, cv_mae

def logRegrWorkflow(data_in_sampleDF, data_in_testDF, factorList, targetName):
    train = data_in_sampleDF.loc[:, factorList + [targetName]]
    test = data_in_testDF.loc[:, factorList]
    trainPredictors = [x for x in train.columns if x not in [targetName]]
    scaler = MinMaxScaler(feature_range=(-1, 1))

    X_train, X_cv, y_train, y_cv = train_test_split(data_in_sampleDF[trainPredictors], data_in_sampleDF[targetName],
                                                    test_size=0.15,
                                                    )

    print(X_train.shape)
    Regr = LogisticRegression(fit_intercept=False, max_iter=10000)
    Regr.fit(X_train, y_train)
    predsTrain = Regr.predict(X_train)
    Training_accuracy = np.sum([i == j for i, j in zip(predsTrain, y_train)])/len(y_train)
    Training_mae = mean_absolute_error(predsTrain, y_train)
    print('Training Accuracy: {} MAE: {}'.format(Training_accuracy, Training_mae))

    predsCV = Regr.predict(X_cv)
    accuracy = np.sum([i == j for i, j in zip(predsCV, y_cv)])/len(y_cv)
    mae = mean_absolute_error(predsCV, y_cv)
    print('Cross-validation Accuracy: {} MAE: {}'.format(accuracy, mae))
    preds = Regr.predict_proba(test)[:, 1]
    predSeries = pd.Series(preds, test.index.tolist())
    return predSeries, Training_accuracy, Training_mae, accuracy, mae

def SVMWorkflow(data_in_sampleDF, data_in_testDF, factorList, targetName, kernel='rbf'):
    train = data_in_sampleDF.loc[:, factorList + [targetName]]
    test = data_in_testDF.loc[:, factorList]
    trainPredictors = [x for x in train.columns if x not in [targetName]]

    X_train, X_cv, y_train, y_cv = train_test_split(data_in_sampleDF[trainPredictors], data_in_sampleDF[targetName],
                                                    test_size=0.15,
                                                    random_state=42)
    print(X_train.shape)
    Regr = SVC(kernel=kernel, probability=True)
    Regr.fit(X_train, y_train)
    predsTrain = Regr.predict(X_train)
    Training_accuracy = np.sum([i == j for i, j in zip(predsTrain, y_train)])/len(y_train)
    Training_mae = mean_absolute_error(predsTrain, y_train)
    print('Training Accuracy: {} MAE: {}'.format(Training_accuracy, Training_mae))

    predsCV = Regr.predict(X_cv)
    accuracy = np.sum([i == j for i, j in zip(predsCV, y_cv)])/len(y_cv)
    mae = mean_absolute_error(predsCV, y_cv)
    print('Cross-validation Accuracy: {} MAE: {}'.format(accuracy, mae))
    preds = Regr.predict_proba(test.loc[:, factorList])[:, 1]
    predSeries = pd.Series(preds, test.index.tolist())
    return predSeries, Training_accuracy, Training_mae, accuracy, mae

def XGBCWorkflow(data_in_sampleDF, data_in_testDF, factorList, targetName):
    train = data_in_sampleDF.loc[:, factorList + [targetName]]
    test = data_in_testDF.loc[:, factorList]
    trainPredictors = [x for x in train.columns if x not in [targetName]]

    X_train, X_cv, y_train, y_cv = train_test_split(data_in_sampleDF[trainPredictors], data_in_sampleDF[targetName],
                                                    test_size=0.15,
                                                    random_state=42)
    print(X_train.shape)
    Regr = xgb.XGBClassifier(use_label_encoder=False, n_estimators=200, max_depth=10, subsample=0.80,
                             learning_rate=0.01, objective='binary:logistic', eval_metric='auc', colsample_bytree=0.8, n_jobs=4)
    eval_set = [(X_cv, y_cv)]
    Regr.fit(X_train, y_train, early_stopping_rounds=20, eval_metric='auc', eval_set=eval_set, verbose=False)
    predsTrain = Regr.predict(X_train)
    Training_accuracy = np.sum([i == j for i, j in zip(predsTrain, y_train)])/len(y_train)
    Training_mae = mean_absolute_error(predsTrain, y_train)
    print('Training Accuracy: {} MAE: {}'.format(Training_accuracy, Training_mae))

    predsCV = Regr.predict(X_cv)
    accuracy = np.sum([i == j for i, j in zip(predsCV, y_cv)])/len(y_cv)
    mae = mean_absolute_error(predsCV, y_cv)
    print('Cross-validation Accuracy: {} MAE: {}'.format(accuracy, mae))
    groups = np.array(range(2))
    groupsReshape = groups.reshape(2, -1)
    preds = Regr.predict_proba(test.loc[:, factorList]) # [:, 1]
    groupPreds = (np.dot(preds, groupsReshape)).flatten()
    # preds = preds.sum(axis=1)
    predSeries = pd.Series(groupPreds, test.index.tolist())
    return predSeries, Training_accuracy, Training_mae, accuracy, mae

def XGBRWorkflow(data_in_sampleDF, data_in_testDF, factorList, targetName, ):
    train = data_in_sampleDF.loc[:, factorList + [targetName]]
    test = data_in_testDF.loc[:, factorList]
    trainPredictors = [x for x in train.columns if x not in [targetName]]

    # SP = SelectPercentile(mutual_info_regression, percentile=20)
    SP = SelectKBest(mutual_info_regression, k=50)
    SP.fit(data_in_sampleDF[factorList], data_in_sampleDF[targetName])
    selectedFcts = SP.get_feature_names_out(factorList)
    test = data_in_testDF.loc[:, selectedFcts]
    # print(selectedFcts)
    data_in_sampleDF[selectedFcts].corr().to_csv('results\\pic\\corr_' + testDate + '.csv')
    X_train, X_cv, y_train, y_cv = train_test_split(data_in_sampleDF[selectedFcts], data_in_sampleDF[targetName],
                                                    test_size=0.15)

    Regr = xgb.XGBRegressor(use_label_encoder=False, n_estimators=1000, max_depth=10, min_child_weight=1, subsample=0.95,
                             learning_rate=0.05)
    Regr.fit(X_train, y_train)
    eval_set = [(X_cv, y_cv)]
    Regr.fit(X_train, y_train, early_stopping_rounds=50, eval_set=eval_set, verbose=False)
    predsTrain = Regr.predict(X_train)
    # Training_accuracy = np.sum([i == j for i, j in zip(predsTrain, y_train)])/len(y_train)
    Training_mae = mean_absolute_error(predsTrain, y_train)
    # print('Training Accuracy: {} MAE: {}'.format(Training_accuracy, Training_mae))
    print('MAE: {}'.format(Training_mae))
    fig, ax = plt.subplots(figsize=(20, 12))
    plot_importance(Regr, ax=ax)
    plt.title("Featurertances")
    plt.savefig('results\\pic\\' + testDate + '.png')

    predsCV = Regr.predict(X_cv)
    # accuracy = np.sum([i == j for i, j in zip(predsCV, y_cv)])/len(y_cv)
    mae = mean_absolute_error(predsCV, y_cv)
    # print('Cross-validation Accuracy: {} MAE: {}'.format(accuracy, mae))
    print('Cross-validation MAE: {}'.format(mae))
    preds = Regr.predict(test.loc[:, selectedFcts]) # [:, 1]
    predSeries = pd.Series(preds, test.index.tolist())
    return predSeries, Training_mae, mae

def DTCWorkflow(data_in_sampleDF, data_in_testDF, factorList, targetName):
    train = data_in_sampleDF.loc[:, factorList + [targetName]]
    test = data_in_testDF.loc[:, factorList]
    trainPredictors = [x for x in train.columns if x not in [targetName]]

    X_train, X_cv, y_train, y_cv = train_test_split(data_in_sampleDF[trainPredictors], data_in_sampleDF[targetName],
                                                    test_size=0.15,
                                                    random_state=42)
    print(X_train.shape)
    Regr = tree.DecisionTreeClassifier(min_samples_split=3, max_depth=5)
    Regr.fit(X_train, y_train)
    predsTrain = Regr.predict(X_train)
    Training_accuracy = np.sum([i == j for i, j in zip(predsTrain, y_train)])/len(y_train)
    Training_mae = mean_absolute_error(predsTrain, y_train)
    print('Training Accuracy: {} MAE: {}'.format(Training_accuracy, Training_mae))

    predsCV = Regr.predict(X_cv)
    accuracy = np.sum([i == j for i, j in zip(predsCV, y_cv)])/len(y_cv)
    mae = mean_absolute_error(predsCV, y_cv)
    print('Cross-validation Accuracy: {} MAE: {}'.format(accuracy, mae))
    preds = Regr.predict_proba(test.loc[:, factorList])[:, 1]
    predSeries = pd.Series(preds, test.index.tolist())
    return predSeries, Training_accuracy, Training_mae, accuracy, mae

def RFWorkflow(data_in_sampleDF, data_in_testDF, factorList, targetName, kernel='rbf'):
    train = data_in_sampleDF.loc[:, factorList + [targetName]]
    test = data_in_testDF.loc[:, factorList]
    trainPredictors = [x for x in train.columns if x not in [targetName]]

    X_train, X_cv, y_train, y_cv = train_test_split(data_in_sampleDF[trainPredictors], data_in_sampleDF[targetName],
                                                    test_size=0.15,
                                                    random_state=42)
    print(X_train.shape)
    Regr = RandomForestClassifier(n_estimators=1000, max_features=10, min_samples_split=4, min_samples_leaf=10)
    Regr.fit(X_train, y_train)

    predsTrain = Regr.predict(X_train)
    Training_accuracy = np.sum([i == j for i, j in zip(predsTrain, y_train)])/len(y_train)
    Training_mae = mean_absolute_error(predsTrain, y_train)
    print('Training Accuracy: {} MAE: {}'.format(Training_accuracy, Training_mae))

    predsCV = Regr.predict(X_cv)
    accuracy = np.sum([i == j for i, j in zip(predsCV, y_cv)])/len(y_cv)
    mae = mean_absolute_error(predsCV, y_cv)
    print('Cross-validation Accuracy: {} MAE: {}'.format(accuracy, mae))
    preds = Regr.predict_proba(test.loc[:, factorList])[:, 1]
    predSeries = pd.Series(preds, test.index.tolist())
    return predSeries, Training_accuracy, Training_mae, accuracy, mae

def LGBMWorkflow(data_in_sampleDF, data_in_testDF, factorList, targetName, kernel='rbf'):
    train = data_in_sampleDF.loc[:, factorList + [targetName]]
    test = data_in_testDF.loc[:, factorList]
    trainPredictors = [x for x in train.columns if x not in [targetName]]
    scaler = MinMaxScaler(feature_range=(-1, 1))

    X_train, X_cv, y_train, y_cv = train_test_split(data_in_sampleDF[trainPredictors], data_in_sampleDF[targetName],
                                                    test_size=0.15,
                                                    random_state=42)
    #X_train = scaler.fit_transform(X_train)
    #X_cv = scaler.fit_transform(X_cv)
    #X_test = scaler.fit_transform(test)
    params_test1 = {
        'max_depth': range(3, 20, 2),
        'num_leaves': range(20, 200, 20)
    }
    params_test1 = {
        'learning_rate': [0.001, 0.01, 0.1, 1],
        # 'num_leaves': range(20, 200, 20)
    }
    print(X_train.shape)
    lgbmregr = lgbm.LGBMClassifier(learning_rate=0.01, n_estimators=500, max_depth=10, random_state=42, num_leaves=50,
    subsample=0.8, objective='binary', num_threads=4)
    # gsearch1 = GridSearchCV(estimator=lgbmregr, param_grid=params_test1, scoring='roc_auc', cv=3,
    #                         verbose=1, n_jobs=4)
    # gsearch1.fit(X_train, y_train)
    # print(f'Best params: {gsearch1.best_params_}')
    # print(f'Best validation score = {gsearch1.best_score_}')

    eval_set = [(X_cv, y_cv)]
    lgbmregr.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=50, verbose=False)

    predsTrain = lgbmregr.predict(X_train)

    plt.figure(figsize=(20, 12))
    lgbm.plot_importance(lgbmregr, max_num_features=30)
    plt.title("Featurertances")
    plt.savefig('results\\pic\\' + testDate + '.png')
    Training_accuracy = np.sum([i == j for i, j in zip(predsTrain, y_train)])/len(y_train)
    Training_mae = mean_absolute_error(predsTrain, y_train)
    print('Training Accuracy: {} MAE: {}'.format(Training_accuracy, Training_mae))

    predsCV = lgbmregr.predict(X_cv)
    accuracy = np.sum([i == j for i, j in zip(predsCV, y_cv)])/len(y_cv)
    mae = mean_absolute_error(predsCV, y_cv)
    print('Cross-validation Accuracy: {} MAE: {}'.format(accuracy, mae))
    preds = lgbmregr.predict_proba(test)[:, 1]
    predSeries = pd.Series(preds, test.index.tolist())
    return predSeries, Training_accuracy, Training_mae, accuracy, mae

def LGBMRWorkflow(data_in_sampleDF, data_in_testDF, factorList, targetName, kernel='rbf'):
    train = data_in_sampleDF.loc[:, factorList + [targetName]]
    test = data_in_testDF.loc[:, factorList]
    trainPredictors = [x for x in train.columns if x not in [targetName]]

    # SP = SelectKBest(mutual_info_regression, k=100)
    # SP.fit(data_in_sampleDF[factorList], data_in_sampleDF[targetName])
    # selectedFcts = SP.get_feature_names_out(factorList)
    # test = data_in_testDF.loc[:, selectedFcts]
    # print(selectedFcts)
    X_train, X_cv, y_train, y_cv = train_test_split(data_in_sampleDF[trainPredictors], data_in_sampleDF[targetName],
                                                    test_size=0.15)

    # data_train = lgbm.Dataset(X_train, y_train, silent=True)
    # model_lgb = lgbm.LGBMRegressor(objective='regression', num_leaves=50,
    #                               learning_rate=0.05, n_estimators=500, max_depth=6,
    #                               metric='rmse', bagging_fraction=0.8, feature_fraction=0.8)
    #
    # params_test1 = {
    #     'max_depth': range(15, 25, 2),
    #     'num_leaves': range(150, 270, 30)
    # }
    # params_test4 = {
    #     'feature_fraction': [0.5, 0.6, 0.7, 0.8, 0.9],
    #     'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0]
    # }
    # gsearch1 = GridSearchCV(estimator=model_lgb, param_grid=params_test4, scoring='neg_mean_squared_error', cv=5,
    #                         verbose=1, n_jobs=4)
    # gsearch1.fit(X_train, y_train)
    # print(gsearch1.cv_results_['mean_test_score'],
    #  gsearch1.cv_results_['params'])


    lgbmregr = lgbm.LGBMRegressor(learning_rate=0.05, n_estimators=1000, max_depth=10, num_leaves=500, colsample_bytree=0.95,
    subsample=0.95, objective='regression',  metric='rmse', reg_alpha=0.8)
    print(X_train.shape)
    eval_set = [(X_cv, y_cv)]
    lgbmregr.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=50, verbose=False)

    predsTrain = lgbmregr.predict(X_train)

    Training_mae = mean_absolute_error(predsTrain, y_train)
    print('Training MAE: {}'.format(Training_mae))

    predsCV = lgbmregr.predict(X_cv)
    mae = mean_absolute_error(predsCV, y_cv)
    print('Cross-validation MAE: {}'.format(mae))
    preds = lgbmregr.predict(test)
    predSeries = pd.Series(preds, test.index.tolist())
    plt.figure(figsize=(20, 12))
    lgbm.plot_importance(lgbmregr, max_num_features=50, figsize=(20, 12))
    plt.title("Featurertances")
    plt.savefig('results\\pic\\' + testDate + '.png')
    return predSeries, Training_mae, mae

def GBCWorkflow(data_in_sampleDF, data_in_testDF, factorList, targetName, kernel='rbf'):
    train = data_in_sampleDF.loc[:, factorList + [targetName]]
    test = data_in_testDF.loc[:, factorList]
    trainPredictors = [x for x in train.columns if x not in [targetName]]

    scaler = MinMaxScaler(feature_range=(-1, 1))

    X_train, X_cv, y_train, y_cv = train_test_split(data_in_sampleDF[trainPredictors], data_in_sampleDF[targetName],
                                                    test_size=0.15,
                                                    random_state=42)
    print(X_train.shape)
    regr = GradientBoostingClassifier(learning_rate=0.005)
    regr.fit(X_train, y_train)

    predsTrain = regr.predict(X_train)
    Training_accuracy = np.sum([i == j for i, j in zip(predsTrain, y_train)])/len(y_train)
    Training_mae = mean_absolute_error(predsTrain, y_train)
    print('Training Accuracy: {} MAE: {}'.format(Training_accuracy, Training_mae))

    predsCV = regr.predict(X_cv)
    accuracy = np.sum([i == j for i, j in zip(predsCV, y_cv)])/len(y_cv)
    mae = mean_absolute_error(predsCV, y_cv)
    print('Cross-validation Accuracy: {} MAE: {}'.format(accuracy, mae))
    preds = regr.predict_proba(test.loc[:, factorList])[:, 1]
    predSeries = pd.Series(preds, test.index.tolist())
    return predSeries, Training_accuracy, Training_mae, accuracy, mae

if __name__ == '__main__':
    fctTypes = ['alpha191rolling10']
    for fctType in fctTypes:
        fctInfo = pd.read_excel('F:\\research\\多因子\\机器学习\\结果统计.xlsx', sheet_name='筛选后优矿因子')
        factorList = fctInfo.loc[fctInfo['大类']==fctType, '具体因子'].tolist()
        # factorList = fctInfo.loc[(fctInfo['大类']=='天软高频') | (fctInfo['大类']=='收益与风险类')
        # |(fctInfo['大类']=='技术指标')|(fctInfo['大类']=='情绪')|(fctInfo['大类']=='动量')|(fctInfo['大类']=='alpha101'), '具体因子'].tolist()
        # fctInfo = pd.read_excel('F:\\research\\多因子\\机器学习\\结果统计.xlsx', sheet_name='筛选后优矿因子')
        factorList = fctInfo.loc[(fctInfo['大类']!='行业') , '具体因子'].tolist()
        # factorList = fctInfo.loc[:, '具体因子'].tolist()


        if 'Skewness' in factorList:
            factorList.remove('Skewness')
        if 'Volatility' in factorList:
            factorList.remove('Volatility')

        print(factorList)
        # factorList = ['CallAuction_Buy1_2017_2020_pctMean',we
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
        # factorList = ['Amt_Rank_1', 'Davis_Momentum', 'net_buy_pct_Share_N', 'ReversalFactor']
        closePrice = toolkits().readRawData('closePrice', 'database\\tradingData')
        lowPrice = toolkits().readRawData('lowPrice', 'database\\tradingData')
        highPrice = toolkits().readRawData('highPrice', 'database\\tradingData')
        dailyHL = abs(highPrice-lowPrice)/(highPrice+lowPrice)

        dailySTD = toolkits().readRawData('std_1m_daily', 'database\\tradingData')

        benchmark = pd.read_csv('database\\tradingData\\SH000905.csv', index_col=0)
        tradingDates = toolkits().getTradingPeriod('2020-01-01', '2021-10-15')
        filePath = 'database\\fctValue'
        target = closePrice
        factorDict = {}
        factorLinearDict = {}
        factorNonLinearDict = {}
        stat = {}
        tradingDates = pd.Series(toolkits().cutTimeSeries(tradingDates, freq='week'),
                                 index=toolkits().cutTimeSeries(tradingDates, freq='week'))
        historyXPeriods = 50
        for tradingDate in tradingDates[historyXPeriods+1:]:
            print('{} starts.'.format(tradingDate))
            trainingDates = tradingDates[:tradingDate].iloc[-(historyXPeriods+1):-1]
            targetDates = tradingDates.shift(-1)[:tradingDate].iloc[-(historyXPeriods+1):-1]
            testDate = tradingDate
            sampleData = getSampleData_V1(filePath, trainingDates, target, targetDates, factorList, benchmark, dailySTD, fileType='pkl', upLimit=0.4, downLimit=0.6)
            testData = getTestDataV1(filePath, [tradingDate], factorList, fileType='pkl')
            # 神经网络: ANNWorkflow,
            # logistic回归: logRegrWorkflow,no_indu_
            # 支持向量机: SVMWorkflow,
            # Xgboost: XGBCWorkflow,
            # 决策树: DTCWorkflow,
            # 随机森林: RFWorkflow,
            # LGBM: LGBMWorkflow,
            # GradientBoosting: GBCWorkflow
            # pred, Training_accuracy, Training_mae, CV_accuracy, CV_mae = XGBRWorkflow(sampleData, testData, factorList, 'return_bin')
            pred, Training_mae, cv_mae = LGBMRWorkflow(sampleData, testData, factorList, 'return_bin')

            factorDict[tradingDate] = pred
        fctDF = pd.DataFrame(factorDict).T
        fctDF.to_csv('results\\fct\\' + fctType + '_lightgbm_100_selected.csv')