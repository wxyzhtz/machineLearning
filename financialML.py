import pandas as pd
from tools import toolkits
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


if __name__ == '__main__':
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

    mom = closePrice/closePrice.shift(20) - 1
    fctType = '动量'
    fctInfo = pd.read_excel('F:\\research\\多因子\\机器学习\\结果统计.xlsx', sheet_name='筛选后优矿因子')
    factorList = fctInfo.loc[(fctInfo['大类']=='天软高频') | (fctInfo['大类']=='收益与风险类')
    |(fctInfo['大类']=='技术指标')|(fctInfo['大类']=='情绪')|(fctInfo['大类']=='动量'), '具体因子'].tolist()
    if 'Skewness' in factorList:
        factorList.remove('Skewness')
    if 'Volatility' in factorList:
        factorList.remove('Volatility')
    # financialFactorList = ['ROA', 'ROE', 'NPToTOR', 'EPS', 'CurrentRatio', 'QuickRatio', 'TotalAssetsTRate', 'InventoryTRate',
    #                        'ARTRate', 'CurrentAssetsTRate', 'OperatingRevenueGrowRate', 'OperatingProfitGrowRate']
    fctDict = {}
    for financialFactor in factorList:
        print(financialFactor)
        df = toolkits().readRawData(financialFactor, 'database\\rawFct')
        fctDict[financialFactor] = toolkits().getRank(df, 1)

    combinedFct = {}
    for tradingDate in tradingDates:
        print(tradingDate)
        crossSectionValueDict = {}
        for key in fctDict.keys():
            crossSectionValueDict[key] = fctDict[key].loc[tradingDate, :]
        crossSectionValue = pd.DataFrame(crossSectionValueDict)
        crossSectionValue.dropna(inplace=True)
        cosSimilarityValues = cosine_similarity(crossSectionValue.values)
        np.fill_diagonal(cosSimilarityValues, 0, wrap=False)

        cosSimilarityDF = pd.DataFrame(cosSimilarityValues, index=crossSectionValue.index.tolist(), columns=crossSectionValue.index.tolist())
        cosSimilarityDFPct = cosSimilarityDF.div(cosSimilarityDF.sum(axis=1) , axis=1)
        mom_cs = mom.loc[tradingDate, crossSectionValue.index.tolist()].tolist()
        similarMom = cosSimilarityDFPct.mul(mom_cs, axis=1).sum()
        combinedFct[tradingDate] = similarMom
    combinedFct = pd.DataFrame(combinedFct).T
    combinedFct.to_csv('results\\fct\\xy_factor_' + 'no_fundamental' + '.csv')
