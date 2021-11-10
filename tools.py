import pandas as pd
import numpy as np
import math
import datetime
from sklearn.linear_model import LinearRegression
import ast
import os
from scipy.stats.mstats import winsorize

class toolkits():
    def __init__(self):
        self.stockCode = pd.read_csv('database\\stockCode\\stock_code.csv', index_col=0, dtype=str)
        self.calender = pd.read_csv('database\\calender\\trading_calender.csv', index_col=0, dtype=str)

    def getCodeWithStockExchange(self):
        return list(map(lambda x:x[2:] + '.' + x[:2], self.stockCode['代码'].tolist()))

    def readRawData(self, dataName, dataAddress, startDate='2010-01-01', endDate='2050-01-01', fileType='csv'):
        if fileType=='csv':
            rawData = pd.read_csv(dataAddress + '\\' + dataName + '.csv', index_col=0)
        elif fileType=='pkl':
            rawData = pd.read_pickle(dataAddress + '\\' + dataName + '.pkl', compression='gzip')

        rawData.columns = list(map(self.toTicker, rawData.columns.tolist()))
        rawData.index = list(map(self.toStrDate, rawData.index.tolist()))
        rawData = rawData.loc[startDate:endDate, :]
        allStockCode = self.stockCode['A股代码(末值)'].tolist()
        rawData = rawData.reindex(columns=allStockCode)
        rawData.sort_index(axis=1, inplace=True)
        rawData.sort_index(axis=0, inplace=True)
        return rawData

    def toTicker(self, x):
        if type(x) == str and ('.X' in x or 'S' in x):
            return '0' * (6 - len(x)) + ''.join(list(filter(str.isdigit, x)))
        elif type(x) == str and not '.' in x:
            return '0' * (6 - len(str(int(x)))) + ''.join(list(filter(str.isdigit, str(int(x)))))
        elif type(x) == str:
            return '0' * (6 - len(str(int(float(x))))) + ''.join(list(filter(str.isdigit, str(int(float(x))))))
        elif type(x) == int:
            return '0' * (6 - len(str(x))) + str(x)
        elif math.isnan(x):
            return np.NaN
        elif type(x) == float:
            return '0' * (6 - len(str(int(x)))) + str(int(x))

    def toStrDate(self, x):
        if '/' in x:
            x = datetime.datetime.strptime(x, '%Y/%m/%d')
        elif '-' in x:
            x = datetime.datetime.strptime(x, '%Y-%m-%d')
        x = datetime.datetime.strftime(x, '%Y-%m-%d')
        return x

    def getTradingPeriod(self, startDate, endDate):
        return self.calender.loc[startDate:endDate, 'trading_dates'].tolist()

    def getZScore(self, df, num=1):
        for _ in range(num):
            df = self.getWinsorize(df)
            df = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)
        return df

    def getRank(self, df, num=1):
        for _ in range(num):
            df = self.getWinsorize(df)
            df = df.rank(axis=1, pct=True)
        return df

    def getWinsorize(self, df):
        quantile_max = 0.99
        quantile_min = 0.01
        quantile_max_series = df.quantile(q=quantile_max, axis=1)
        quantile_min_series = df.quantile(q=quantile_min, axis=1)
        quantile_max_mask = df.ge(quantile_max_series, axis='index')
        quantile_min_mask = df.le(quantile_min_series, axis='index')
        df[quantile_max_mask] = 0
        df[quantile_min_mask] = 0
        df[quantile_max_mask] = df[quantile_max_mask].add(quantile_max_series, axis=0)
        df[quantile_min_mask] = df[quantile_min_mask].add(quantile_min_series, axis=0)
        return df

    def getWinsorizeStat(self, df):
        df_winsorized = winsorize(df.values, limits=[0.2, 0.2], axis=1, nan_policy='omit')
        df = pd.DataFrame(df_winsorized, index=df.index.tolist(), columns=df.columns.tolist())
        return df

    def getXTradedate(self, current_date='2021-07-28', numDate=10):
        getXTradedate_shift = self.calender.shift(numDate)
        return getXTradedate_shift.loc[current_date].values[0]

    def getSampleData(self, filePath, date_in_sample, percent_select, factorList):
        data_in_sample = []
        for date in date_in_sample:
            fileName = filePath + '\\' + date + '.csv'
            dataDaily = pd.read_csv(fileName, index_col=0)
            dataDaily = dataDaily.loc[:,
                        ['status', '5dayFutureRet', '5dayFutureRetTWAP'] + factorList]
            dataDaily = dataDaily.loc[dataDaily.status != 0, :]
            dataDaily.replace(np.inf, np.NaN, inplace=True)
            dataDaily.loc[dataDaily.count(axis=1) < 4] = np.NaN
            dataDaily.dropna(how='all', inplace=True)
            dataDaily.fillna(dataDaily.mean(), inplace=True)
            dataDaily.dropna(inplace=True)
            dataDaily = self.label_data(dataDaily, percent_select)
            data_in_sample.append(dataDaily)

        # print(data_in_sample)
        data_in_sampleDF = pd.concat(data_in_sample)
        data_in_sampleDF.dropna(inplace=True)
        return data_in_sampleDF

    def getTestData(self, filePath, date_in_test, factorList):
        fileName = filePath + '\\' + date_in_test[0] + '.csv'
        dataDaily = pd.read_csv(fileName, index_col=0)
        dataDaily = dataDaily.loc[:,
                    ['status', '5dayFutureRet', '5dayFutureRetTWAP']+factorList]
        dataDaily = dataDaily.loc[dataDaily.status != 0, :]
        dataDaily.replace(np.inf, np.NaN, inplace=True)
        dataDaily.loc[dataDaily.count(axis=1) < 4] = np.NaN
        dataDaily.dropna(how='all', inplace=True)
        dataDaily.fillna(dataDaily.mean(), inplace=True)
        dataDaily.dropna(inplace=True)
        dataDaily = self.label_data(dataDaily)
        return dataDaily


    def retFilter(self, retSeries):
        newList = {}
        for i in range(len(retSeries)):
            newList[retSeries.index[i]] = retSeries[i] / 2 if (retSeries.index[i][:2] == '30') or (retSeries.index[i][:2] == '68') else retSeries[i]
        return pd.Series(newList)


    def label_data(self, data, percent_select=[0.5, 0.5]):
        data['return_bin'] = np.NaN
        data = data.sort_values(by='5dayFutureRet', ascending=False)
        n_stock_select = np.multiply(percent_select, data.shape[0])
        n_stock_select = np.round(n_stock_select).astype(int)
        data.iloc[0:n_stock_select[0], -1] = 1
        data.iloc[-n_stock_select[0]:, -1] = 0
        data.dropna(axis=0)
        return data

    def cutTimeSeries(self, timeSeriesList, freq='week', pos='last'):
        if freq == 'week':
            weekList = list(
                map(lambda x: x[:4] + str(datetime.datetime.strptime(x, '%Y-%m-%d').date().isocalendar()[1]),
                    timeSeriesList))
            week_date_contrast = pd.Series(weekList, index=timeSeriesList)
            return week_date_contrast.drop_duplicates(keep=pos).index.tolist()
        elif freq == 'bi_week':
            weekList = list(
                map(lambda x: x[:4] + str(datetime.datetime.strptime(x, '%Y-%m-%d').date().isocalendar()[1]),
                    timeSeriesList))
            week_date_contrast = pd.Series(weekList, index=timeSeriesList)
            return week_date_contrast.drop_duplicates(keep=pos).index.tolist()[::2]
        elif freq == 'month':
            monthList = list(
                map(lambda x: x[:7], timeSeriesList))
            week_date_contrast = pd.Series(monthList, index=timeSeriesList)
            return week_date_contrast.drop_duplicates(keep=pos).index.tolist()
        elif type(freq) == list:
            open_position_date = freq[0]
            holding_position_days = freq[1]
            fct_matrix = timeSeriesList[open_position_date - 1::holding_position_days]
            return fct_matrix
        elif freq == 'daily':
            return pd.Series(timeSeriesList, index=timeSeriesList)

    def calCorrelation(self, fctNames, fctPath):
        fctDates = []
        fctDict = {}
        for fctName in fctNames:
            fct_values = self.readRawData(fctName, fctPath, '2018-08-03')
            fctDates = list(np.sort(list(set(fctDates + fct_values.index.tolist()))))
            fctDict[fctName] = fct_values

        fctValueBag = []
        for fctDate in fctDates:
            fctValueDict = {}
            for key in fctDict.keys():
                if not fctDate in fctDict[key].index:
                    print('{} does not have {} data'.format(key, fctDate))
                    break
                else:
                    fctValueDict[key] = fctDict[key].loc[fctDate, :]

            fctValueDF = pd.DataFrame(fctValueDict)
            fctValueDFCorr = fctValueDF.corr()
            fctValueBag.append(fctValueDFCorr)

        sumCorr = 0
        for i in range(len(fctValueBag)):
            sumCorr = sumCorr + fctValueBag[i]
        meanCorr = sumCorr/len(fctValueBag)
        print()

    def getRes(self, X, y):
        X = X.values.reshape(-1, 1)
        y = y.values.reshape(-1, 1)
        regr = LinearRegression(fit_intercept=False)
        regr.fit(X, y)
        y_pred = regr.predict(X)
        res = y - y_pred
        return res

    def getNEU(self, fctDFName, neuKeys=None):
        print('{} start.'.format(fctDFName))
        indu = pd.read_csv('database\\industry\\total_industry.csv', index_col=0, dtype=str)
        indu.columns = list(map(self.toTicker, indu.columns.tolist()))
        indu.replace(np.inf, np.NaN, inplace=True)

        capL = self.readRawData('LCAP', 'database\\rawFct')
        capL = self.getZScore(capL)
        capL.replace(np.inf, np.NaN, inplace=True)

        fctDF = self.readRawData(fctDFName, 'database\\rawFct')
        fctDF = self.getZScore(fctDF)
        fctDF.replace(np.inf, np.NaN, inplace=True)

        resDict = {}
        for index, row in fctDF.iterrows():
            if (index in indu.index.tolist()) and (index in capL.index.tolist()):
                Y = row.dropna()
                induSelected = indu.loc[index, Y.index.tolist()]
                capLSelected = capL.loc[index, Y.index.tolist()]
                induDummy = pd.get_dummies(induSelected)
                X = pd.concat([induDummy, capLSelected], axis=1)
                X.rename(columns={index:'LCAP'})
                data = pd.concat([Y, X], axis=1)
                data.dropna(inplace=True)
                regr = LinearRegression(fit_intercept=False)
                X = data.iloc[:, 1:]
                Y = data.iloc[:, :1]

                regr.fit(X.values, Y.values)
                Y_hat = regr.predict(X.values)
                res = Y.values - Y_hat
                resDict[index] = pd.Series(res.flatten(), index=data.index.tolist())
        resFct = pd.DataFrame(resDict).T
        a = resFct.dtypes
        resFct.to_pickle('database\\rawFctNeu\\' + fctDFName + '.pkl', compression='gzip')

    def getStockFullTicker(self, stockTickerList):
        stock_code = self.stockCode
        stock_code.index = stock_code.loc[:, 'A股代码(末值)']
        stockFullTickerList = list(
            map(lambda x: stock_code.loc[x, '代码'][-6:] + '.' + stock_code.loc[x, '代码'][:2], stockTickerList))
        return stockFullTickerList

    def CSVtoPickle(self, fileName, dfAddress, saveAddress):
        df = pd.read_csv(dfAddress + '\\' + fileName + '.csv', index_col=0)
        df.to_pickle(saveAddress + '\\' + fileName + '.pkl', compression='gzip')

if __name__ == '__main__':
    # dfAddress = 'database\\Alpha191_rolling10'
    # saveAddress = 'database\\rawFctPickle'
    # fileNames = os.listdir(dfAddress)
    # for fileName in fileNames:
    #     if '.csv' in fileName:
    #         df = toolkits().readRawData(fileName[:-4], dfAddress)
    #         df.to_pickle(saveAddress + '\\' + fileName[:-4] + '.pkl', compression='gzip')

    # fctType = '技术指标'
    # fctInfo = pd.read_excel('F:\\research\\多因子\\机器学习\\结果统计.xlsx', sheet_name='筛选后优矿因子')
    # factorList = fctInfo.loc[fctInfo['大类']==fctType, '具体因子'].tolist()
    #
    a = toolkits()
    # # Amount = a.readRawData('Amount', 'database\\tradingData')
    # # closePrice = a.readRawData('closePrice', 'database\\tradingData')
    corrMatrix = a.calCorrelation(['alpha191rolling10_lightgbm_50', '天软高频_lightgbm_50', ], 'results\\fct')
    # # print(SVM_pred20)
    # # print(Amount, closePrice)
    # for fct in factorList:
    #     a.getNEU(fct)

    # df = pd.read_csv('results\\lstm\\技术指标lstm5.csv', index_col=0)
    # fctDict = {}
    # for index, row in df.iterrows():
    #     fctDict[index] = pd.Series(list(map(lambda x: float(x[1:-1]) if type(x)==str else x, row.tolist())), index=row.index.tolist())
    #     print()
    # df = pd.DataFrame(fctDict).T
    # df.to_csv('results\\lstm\\技术指标lstm5.csv')
    # print()

    # currentData = pd.read_pickle('database\\fctValue\\' + '2017-01-03' + '.pkl', compression='gzip')
    # print()