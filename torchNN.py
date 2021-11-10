import pandas as pd
from tools import toolkits
import ML_API

if __name__ == '__main__':
    fctTypes = ['alpha191rolling10']
    for fctType in fctTypes:
        fctInfo = pd.read_excel('F:\\research\\多因子\\机器学习\\结果统计.xlsx', sheet_name='筛选后优矿因子')
        factorList = fctInfo.loc[fctInfo['大类']==fctType, '具体因子'].tolist()
        # factorList = fctInfo.loc[(fctInfo['大类'] != '行业'), '具体因子'].tolist()
        # factorList = fctInfo.loc[(fctInfo['大类']=='天软高频') | (fctInfo['大类']=='收益与风险类')
        # |(fctInfo['大类']=='技术指标')|(fctInfo['大类']=='情绪')|(fctInfo['大类']=='动量')|(fctInfo['大类']=='alpha101'), '具体因子'].tolist()
        # factorList = fctInfo.loc[(fctInfo['大类']!='行业') , '具体因子'].tolist()
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
        tradingDates = toolkits().getTradingPeriod('2017-02-01', '2021-10-15')
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
            sampleData = ML_API.getSampleData_V1(filePath, trainingDates, target, targetDates, factorList, benchmark, dailySTD, fileType='pkl', upLimit=0.4, downLimit=0.6)
            testData = ML_API.getTestDataV1(filePath, [tradingDate], factorList, fileType='pkl')
            # 神经网络: ANNWorkflow,
            # logistic回归: logRegrWorkflow,no_indu_
            # 支持向量机: SVMWorkflow,
            # Xgboost: XGBCWorkflow,
            # 决策树: DTCWorkflow,
            # 随机森林: RFWorkflow,
            # LGBM: LGBMWorkflow,
            # GradientBoosting: GBCWorkflow
            # pred, Training_accuracy, Training_mae, CV_accuracy, CV_mae = XGBRWorkflow(sampleData, testData, factorList, 'return_bin')
            pred, Training_mae, cv_mae = ML_API.ANNRWorkflow(sampleData, testData, factorList, 'return_bin')

            factorDict[tradingDate] = pred
        fctDF = pd.DataFrame(factorDict).T
        fctDF.to_csv('results\\fct\\' + fctType + '_ANNR_50.csv')