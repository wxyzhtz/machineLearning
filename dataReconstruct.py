import pandas as pd
import numpy as np
import os
from tools import toolkits
import time
import multiprocessing

class factorCluster():
    def __init__(self, factorList, startDate, endDate, saveFolder):
        self.factorList = factorList
        self.startDate = startDate
        self.endDate = endDate
        self.saveFolder = saveFolder
        self.goMarketDays = toolkits().readRawData('daily_GoMarketDays', 'database\\stockStatus')
        self.ST = toolkits().readRawData('daily_st', 'database\\stockStatus')
        self.SUSP = toolkits().readRawData('daily_susp', 'database\\stockStatus')
        self.DT = toolkits().readRawData('daily_dt', 'database\\stockStatus')
        self.ZT = toolkits().readRawData('daily_zt', 'database\\stockStatus')

        self.newOut = self.goMarketDays > 120

        self.filterDF = ((~(self.ST == 1)) & (~(self.SUSP == 1)) & (~(self.DT == 1)) & (~(self.ZT == 1)) & self.newOut) * 1

    def workFlow(self):
        fctDict = {}

        for fctName in self.factorList:
            time1 = time.time()

            df = toolkits().readRawData(fctName, 'database\\rawFct', fileType='csv')
            df = df.loc['2016-01-01':, ]

            print(fctName, df.count(axis=1).min())
            if (df.count(axis=1).min() > 500) and (df.index[-1] > '2021-07-31'):
                # print('{} in.'.format(fctName))
                fctDict[fctName] = df # .loc['2019-11-25':'2019-11-29', ]
            else:
                print('{} not in.'.format(fctName))
            time2 = time.time()
            print(time2-time1)

        for key in fctDict.keys():
            fctDict[key] = toolkits().getZScore(fctDict[key], 1)
            print(fctDict[key])

        tradingDates = toolkits().getTradingPeriod(self.startDate, self.endDate)
        allStockCode = toolkits().getCodeWithStockExchange()
        for tradingDate in tradingDates:
            print(tradingDate)
            tmpDF = pd.DataFrame([tradingDate] * len(allStockCode), index=allStockCode, columns=['tradingDate'])
            stockCode = list(map(toolkits().toTicker, allStockCode))
            tmpDF['status'] = self.filterDF.loc[tradingDate, stockCode].tolist()
            for key in fctDict.keys():
                if tradingDate in fctDict[key].index:
                    tmpDF[key] = fctDict[key].loc[tradingDate, stockCode].tolist()
                else:
                    print('{} has no data on {}'.format(key, tradingDate))
                    tmpDF[key] = np.NaN
            if not os.path.exists('database\\' + self.saveFolder):
                os.mkdir('database\\' + self.saveFolder)
            tmpDF.to_pickle('database\\' + self.saveFolder + '\\' + tradingDate + '.pkl', compression='gzip')

def dumpNew(tradingDate, currentDatabaseDate, newFctName, newFct):
    if tradingDate in currentDatabaseDate:
        print(tradingDate, newFctName)
        currentData = pd.read_pickle('database\\fctValue\\' + tradingDate + '.pkl', compression='gzip')

        if not newFctName in currentData.columns.tolist():
            print(tradingDate, newFctName + ' added.')
            newData = newFct.loc[tradingDate, :]
            newData.index = toolkits().getStockFullTicker(newData.index.tolist())
            currentData = pd.concat([currentData, newData], axis=1)
            currentData.columns = currentData.columns[:-1].tolist() + [newFctName]
            currentData.to_pickle('database\\' + 'fctValue' + '\\' + tradingDate + '.pkl', compression='gzip')

def addNewFct(newFctList):
    currentDatabaseDate = list(map(lambda x: x[:-4], os.listdir('database\\fctValue')))

    for newFctName in newFctList:
        newFct = toolkits().readRawData(newFctName, 'database\\rawFctPickle', fileType='pkl')
        if not 'SW' in newFctName:
            newFct = toolkits().getZScore(newFct, num=1)
        newFct.dropna(how='all', inplace=True)
        tradingDates = newFct.index.tolist()
        pool = multiprocessing.Pool(processes=6)
        for tradingDate in tradingDates:
            if (tradingDate>='2019-11-22') and (tradingDate<='2030-12-03'):
                pool.apply_async(dumpNew, (tradingDate, currentDatabaseDate, newFctName, newFct,))
        pool.close()
        pool.join()
        # dumpNew(tradingDate, currentDatabaseDate, newFctName, newFct)

if __name__ == '__main__':
    # files = os.listdir('F:\\machineLearning\\database\\rawFct')
    # factorList = list(map(lambda x: x[:-4], files))
    # startDate = '-11-25'
    # endDate = '2019-11-29'
    # saveFolder = 'fctValue'
    # a = factorCluster(factorList, startDate, endDate, saveFolder)
    # a.workFlow()

    FctList = os.listdir('F:\\machineLearning\\database\\rawFctPickle')
    FctList = list(map(lambda x: x[:-4], FctList))
    newFctList = []
    for fct in FctList:
        if ('rolling10' in fct) and ('Alpha101_alpha4_rolling10' in fct):
            newFctList.append(fct)
            print(fct)
    addNewFct(newFctList)
    print(newFctList)
    # alpha191_list = []
    # df = pd.read_pickle('database\\fctValue\\2021-10-19.pkl', compression='gzip')
    # for fct in df.columns:
    #     if ('rolling10' in fct) and ('Alpha101' in fct):
    #         alpha191_list.append(fct)
    #         print(fct)
    # print()
