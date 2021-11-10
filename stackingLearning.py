import pandas as pd
import ML_API
from tools import toolkits
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    fctType = '技术指标'
    fctInfo = pd.read_excel('F:\\research\\多因子\\机器学习\\结果统计.xlsx', sheet_name='筛选后优矿因子')
    factorList = fctInfo.loc[fctInfo['大类']==fctType, '具体因子'].tolist()

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
    # factorList = ['Amt_Rank_1', 'Davis_Momentum', 'net_buy_pct_Share_N', 'ReversalFactor']
    closePrice = toolkits().readRawData('closePrice', 'database\\tradingData')
    lowPrice = toolkits().readRawData('lowPrice', 'database\\tradingData')
    highPrice = toolkits().readRawData('highPrice', 'database\\tradingData')
    dailyHL = abs(highPrice - lowPrice) / (highPrice + lowPrice)
    benchmark = pd.read_csv('database\\tradingData\\SH000905.csv', index_col=0)
    tradingDates = toolkits().getTradingPeriod('2017-01-01', '2021-08-20')
    filePath = 'database\\fctValueBackup'
    target = closePrice
    factorDict = {}
    factorLinearDict = {}
    factorNonLinearDict = {}
    stat = {}
    tradingDates = pd.Series(toolkits().cutTimeSeries(tradingDates, freq='week'),
                             index=toolkits().cutTimeSeries(tradingDates, freq='week'))
    historyXPeriods = 50
    for tradingDate in tradingDates[historyXPeriods + 1:]:
        print('{} starts.'.format(tradingDate))
        trainingDates = tradingDates[:tradingDate].iloc[-(historyXPeriods + 1):-1]
        targetDates = tradingDates.shift(-1)[:tradingDate].iloc[-(historyXPeriods + 1):-1]
        testDate = tradingDate
        sampleData = ML_API.getSampleData(filePath, trainingDates, target, targetDates, factorList, benchmark, dailyHL)
        testData = ML_API.getTestDataV1(filePath, [tradingDate], factorList)
        # 神经网络: ANNWorkflow,
        # logistic回归: logRegrWorkflow,
        # 支持向量机: SVMWorkflow,
        # Xgboost: XGBCWorkflow,
        # 决策树: DTCWorkflow,
        # 随机森林: RFWorkflow,
        # LGBM: LGBMWorkflow,
        # GradientBoosting: GBCWorkflow
        trainX = sampleData.loc[:, factorList].values
        trainy = sampleData.loc[:, ['return_bin']].values
        X_test = testData.loc[:, factorList].values

        X_train, X_cv, y_train, y_cv = train_test_split(trainX, trainy, test_size=0.2, random_state=0)

        XGBCWorkflowCase = ML_API.XGBCWorkflowAPI(X_train, y_train, randomSeed=1)
        # XGBCWorkflowCase_traingSet = XGBCWorkflowCase.X_train
        XGBCWorkflowCase_traingPred = XGBCWorkflowCase.predictResult(X_train)
        XGBCWorkflowCase_cvPred = XGBCWorkflowCase.predictResult(X_cv)
        XGBCWorkflowCase_testPred = XGBCWorkflowCase.predictResult(X_test)

        # # XGBCWorkflowCase_traingReal = trainy
        #
        # LGBMCWorkflowCase = ML_API.LGBMCWorkflowAPI(X_train, y_train, randomSeed=2)
        # # LGBMCWorkflowCase_traingSet = LGBMCWorkflowCase.X_train
        # LGBMCWorkflowCase_traingPred = LGBMCWorkflowCase.predictResult(X_train)
        # LGBMCWorkflowCase_cvPred = LGBMCWorkflowCase.predictResult(X_cv)
        # LGBMCWorkflowCase_testPred = LGBMCWorkflowCase.predictResult(X_test)

        # LGBMCWorkflowCase_traingReal = trainy

        LOGWorkflowCase = ML_API.LOGWorkflowAPI(X_train, y_train, randomSeed=3)
        # LOGWorkflowCase_traingSet = LOGWorkflowCase.X_train
        LOGWorkflowCase_traingPred = LOGWorkflowCase.predictResult(X_train)
        LOGWorkflowCase_cvPred = LOGWorkflowCase.predictResult(X_cv)
        LOGWorkflowCase_testPred = LOGWorkflowCase.predictResult(X_test)

        # LOGWorkflowCase_traingReal = LOGWorkflowCase.y_train

        # RFWorkflowCase = ML_API.RFWorkflowAPI(X_train, y_train, randomSeed=4)
        # # RFWorkflowCase_traingSet = RFWorkflowCase.X_train
        # RFWorkflowCase_traingPred = RFWorkflowCase.predictResult(X_train)
        # RFWorkflowCase_cvPred = RFWorkflowCase.predictResult(X_cv)
        # RFWorkflowCase_testPred = RFWorkflowCase.predictResult(X_test)
        #
        # # RFWorkflowCase_traingReal = RFWorkflowCase.y_train
        #
        # GBDTWorkflowCase = ML_API.GBDTWorkflowAPI(X_train, y_train, randomSeed=5)
        # # RFWorkflowCase_traingSet = RFWorkflowCase.X_train
        # GBDTWorkflowCase_traingPred = GBDTWorkflowCase.predictResult(X_train)
        # GBDTWorkflowCase_cvPred = GBDTWorkflowCase.predictResult(X_cv)
        # GBDTWorkflowCase_testPred = GBDTWorkflowCase.predictResult(X_test)

        X_train_stacking = np.transpose(np.vstack((
                                                XGBCWorkflowCase_traingPred,
                                                # LGBMCWorkflowCase_traingPred,
                                                LOGWorkflowCase_traingPred,
                                                # RFWorkflowCase_traingPred,
                                                # GBDTWorkflowCase_traingPred
                                                   )))
        X_cv_stacking = np.transpose(np.vstack((
                                                XGBCWorkflowCase_cvPred,
                                                # LGBMCWorkflowCase_cvPred,
                                                LOGWorkflowCase_cvPred,
                                                # RFWorkflowCase_cvPred,
                                                # GBDTWorkflowCase_cvPred
                                                )))
        X_test_stacking = np.transpose(np.vstack((
                                                XGBCWorkflowCase_testPred,
                                                # LGBMCWorkflowCase_testPred,
                                                LOGWorkflowCase_testPred,
                                                # RFWorkflowCase_testPred,
                                                # GBDTWorkflowCase_testPred
                                                )))

        # modelRealOutput = np.hstack((XGBCWorkflowCase_traingReal, LGBMCWorkflowCase_traingReal, LOGWorkflowCase_traingReal, RFWorkflowCase_traingReal))


        modelTrainingSet = LogisticRegression(fit_intercept=False, max_iter=1000)
        modelTrainingSet.fit(X=X_train_stacking, y=y_train)

        # df = pd.DataFrame(X_train_stacking, columns = ['XGB', 'LGBM', 'Log', 'RF', 'GBDT'])
        df = pd.DataFrame(X_train_stacking, columns = ['XGB', 'Log',])

        corr = df.corr()
        modelTrainPred = modelTrainingSet.predict(X_train_stacking)
        modelCVPred = modelTrainingSet.predict(X_cv_stacking)

        # modelCVPred = modelTrainingSet.predict_proba(X_train_stacking)

        Training_accuracy = np.sum([i == j for i, j in zip(modelTrainPred, y_train)]) / len(y_train)
        CV_accuracy = np.sum([i == j for i, j in zip(modelCVPred, y_cv)]) / len(y_cv)
        randomChoice = np.sum(y_cv) / len(y_cv)
        print(CV_accuracy, randomChoice)

        modeltestPred = modelTrainingSet.predict_proba(X_test_stacking)
        groups = np.array(range(2))
        groupsReshape = groups.reshape(2, -1)
        groupPreds = (np.dot(modeltestPred, groupsReshape)).flatten()
        predSeries = pd.Series(groupPreds, testData.index.tolist())
        factorDict[tradingDate] = predSeries
    fctDF = pd.DataFrame(factorDict).T
    fctDF.to_csv('results\\fct\\' + fctType + '_stacking_50.csv')
        # pred, Training_accuracy, Training_mae, CV_accuracy, CV_mae = ML_API.XGBCWorkflow()

        # factorDict[tradingDate] = pred
    # fctDF = pd.DataFrame(factorDict).T
    # fctDF.to_csv('results\\fct\\' + '_XGBC_12_15fct.csv')