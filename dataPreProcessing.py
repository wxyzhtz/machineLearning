import pandas as pd
import numpy as np
from tools import toolkits
import random
import lightgbm as lgbm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

# 数据处理流程
def getSampleData(filePath, training_target_Series, targetDF, factorList, benchmark, dailySTD, fileType='csv', upLimit=0.7, downLimit=0.3):
    data_in_sample = []
    for date in training_target_Series.index.tolist():
        if fileType == 'csv':
            fileName = filePath + '\\' + date + '.csv'
            dataDaily = pd.read_csv(fileName, index_col=0)
        elif fileType == 'pkl':
            fileName = filePath + '\\' + date + '.pkl'
            dataDaily = pd.read_pickle(fileName, compression='gzip')

        dataDaily.index = list(map(toolkits().toTicker, dataDaily.index.tolist()))
        # print(np.array(factorList)[~np.in1d(factorList, dataDaily.columns.tolist())])
        dataDaily = dataDaily.loc[:,
                    ['status'] + factorList]

        dataDaily = dataDaily.loc[dataDaily.status != 0, :]
        dataDaily.replace(np.inf, np.NaN, inplace=True)
        dataDaily.replace(0, np.NaN, inplace=True)
        factorNaN = dataDaily.loc[:, factorList].count(axis=1)

        dataDaily.loc[factorNaN < (len(factorList) * 0.8), :] = np.NaN

        noAllNaN = dataDaily.loc[:, factorList].dropna(how='all')
        dataDaily = dataDaily.loc[noAllNaN.index.tolist()]
        # dataDaily.loc[:, factorList] = (dataDaily.loc[:, factorList].sub(dataDaily.loc[:, factorList].mean())).div(dataDaily.loc[:, factorList].std())
        dataDaily.loc[:, factorList] = dataDaily.loc[:, factorList].rank(pct=True)
        # dataDaily.loc[:, factorList] = (toolkits().getZScore(dataDaily.loc[:, factorList].T, 1)).T
        # for fct in factorList:
        #     plt.hist(dataDaily[fct].tolist(), bins=100, density=True)
        #     plt.title(fct)
        #     plt.show()
        dataDaily.fillna(dataDaily.mean(), inplace=True)
        ret = targetDF.loc[training_target_Series[date], :] / targetDF.loc[date, :] - 1

        # periodSTD = dailySTD.loc[toolkits().getXTradedate(date, -1): targetDates[date], :]
        # periodSTD.dropna(how='all', inplace=True, axis=1)
        # periodSTD_Mean = periodSTD.mean()

        benchmark_ret = benchmark.loc[training_target_Series[date], 'closePrice'] / benchmark.loc[date, 'closePrice'] - 1
        sameTicker = list(set(dataDaily.index.tolist()).intersection(set(ret.index.tolist())))
        dataDaily = dataDaily.loc[np.in1d(dataDaily.index.tolist(), sameTicker)]
        dataDaily['ret'] = ret.loc[dataDaily.index.tolist()].tolist() - benchmark_ret
        # dataDaily = pd.concat([dataDaily, periodSTD_Mean], axis=1)
        dataDaily.dropna(inplace=True)
        res = dataDaily['ret']  # /dataDaily.iloc[:, -1]
        res = (res - res.mean()) / res.std()
        res[res.rank(pct=True) > 0.99] = res.quantile(0.99)
        res[res.rank(pct=True) < 0.01] = res.quantile(0.01)
        dataDaily['ret'] = res
        dataDaily.index = list(map(lambda x: x + '_' + date, dataDaily.index.tolist()))

        data_in_sample.append(dataDaily)
    data_in_sampleDF = pd.concat(data_in_sample)
    data_in_sampleDF = getLabelRegr(data_in_sampleDF, upLimit, downLimit)
    data_in_sampleDF.dropna(inplace=True)
    return data_in_sampleDF

def getTestData(filePath, date_in_test, factorList, fileType='csv'):
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
    # dataDaily.loc[:, factorList] = (toolkits().getZScore(dataDaily.loc[:, factorList].T, 1)).T
    dataDaily.loc[:, factorList] = dataDaily.loc[:, factorList].rank(pct=True)
    dataDaily.fillna(dataDaily.mean(), inplace=True)
    dataDaily.dropna(inplace=True)
    return dataDaily

def LGBMRWorkflow(data_in_sampleDF, data_in_testDF, factorList, targetName, kernel='rbf'):
    train = data_in_sampleDF.loc[:, factorList + [targetName]]
    test = data_in_testDF.loc[:, factorList]

    trainPredictors = [x for x in train.columns if x not in [targetName]]

    # SP = SelectKBest(mutual_info_regression, k=100)
    # SP.fit(data_in_sampleDF[factorList], data_in_sampleDF[targetName])
    # selectedFcts = SP.get_feature_names_out(factorList)
    # test = data_in_testDF.loc[:, selectedFcts]
    # print(selectedFcts)


    X_train, X_cv, y_train, y_cv = train_test_split(data_in_sampleDF[factorList], data_in_sampleDF[targetName],
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


    lgbmregr = lgbm.LGBMRegressor(learning_rate=0.1, n_estimators=1000, max_depth=10, num_leaves=800, colsample_bytree=0.9,
    subsample=0.9, objective='regression',  metric='rmse', reg_alpha=8, reg_lambda=8)
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
    plt.savefig('results\\pic\\' + fctDate + '_5day.png')
    return predSeries, Training_mae, mae

if __name__ == '__main__':
    fctInfo = pd.read_excel('F:\\research\\多因子\\机器学习\\结果统计.xlsx', sheet_name='筛选后优矿因子')
    factorList = fctInfo.loc[(fctInfo['大类'] == '技术指标'), '具体因子'].tolist()
    # 个别数据缺失
    if 'Skewness' in factorList:
        factorList.remove('Skewness')
    if 'Volatility' in factorList:
        factorList.remove('Volatility')
    # 读取收盘价
    closePrice = toolkits().readRawData('closePrice', 'database\\tradingData')
    # 读取基准
    benchmark = pd.read_csv('database\\tradingData\\SH000905.csv', index_col=0)
    # 读取全部时间序列
    tradingDates = toolkits().getTradingPeriod('2017-02-01', '2021-10-15')

    dailySTD = toolkits().readRawData('std_1m_daily', 'database\\tradingData')

    target = closePrice
    factorDict = {}
    factorLinearDict = {}
    factorNonLinearDict = {}
    stat = {}
    fctDates = pd.Series(toolkits().cutTimeSeries(tradingDates, freq='week'),
                             index=toolkits().cutTimeSeries(tradingDates, freq='week'))
    historyXPeriods = 50
    length_predict = 5
    filePath = 'database\\fctValue'

    for fctDate in fctDates[historyXPeriods + 1:]:
        print(fctDate)
        fctDate_index = tradingDates.index(fctDate)
        trainingDates = tradingDates[(fctDate_index - length_predict - 250) if (fctDate_index - length_predict - 250)>0 else 0:(fctDate_index - length_predict + 1)]
        targetDates = [toolkits().getXTradedate(x, -length_predict) for x in trainingDates]

        training_target_Series = pd.Series(dict(zip(trainingDates, targetDates)))

        res = random.sample(range(0, len(training_target_Series)-1), int(0.8*len(training_target_Series)))
        res.sort()
        training_target_Series = training_target_Series.iloc[res]
        sampleData = getSampleData(filePath, training_target_Series, target, factorList, benchmark, dailySTD,
                                      fileType='pkl', upLimit=0.7, downLimit=0.3)


        testData = getTestData(filePath, [fctDate], factorList, fileType='pkl')
        pred, Training_mae, cv_mae = LGBMRWorkflow(sampleData, testData, factorList, 'return_bin')

        factorDict[fctDate] = pred
    fctDF = pd.DataFrame(factorDict).T
    fctDF.to_csv('results\\fct\\' + '技术指标' + '_lightgbm_5day.csv')