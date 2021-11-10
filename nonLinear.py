import pandas as pd
from tools import toolkits
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


if __name__ == '__main__':
    rawFct = toolkits().readRawData('BIAS5', 'database\\rawFct')
    rawFct = toolkits().getZScore(rawFct)
    closePrice = toolkits().readRawData('closePrice', 'database\\tradingData')

    begt = '2017-01-01'
    endt = '2021-08-20'
    tradingPeriods = toolkits().getTradingPeriod(begt, endt)
    lastTradingDateInWeek = toolkits().cutTimeSeries(tradingPeriods)
    closePrice = closePrice.loc[lastTradingDateInWeek, :]
    rawFct = rawFct.loc[lastTradingDateInWeek, :]
    paramsDict = {}
    for tradingDate in lastTradingDateInWeek[1:]:
        print(tradingDate)
        rawFctTemp = rawFct.loc[tradingDate, :]
        rawFctTemp_2 = rawFctTemp**2
        rawFctTemp_3 = rawFctTemp**3

        lastFctValue = rawFct.shift(1).loc[tradingDate, :]
        lastFctValue_2 = lastFctValue**(2)
        lastFctValue_3 = lastFctValue**(3)
        lastFctValue_4 = lastFctValue ** (1/2)
        lastFctValue_5 = lastFctValue ** (1/3)

        closePriceTemp = closePrice.loc[tradingDate, :]
        lastClosePriceTemp = closePrice.shift(1).loc[tradingDate, :]
        ret = closePriceTemp/lastClosePriceTemp - 1

        lastValueDict = {}
        lastValueDict['x_1'] = lastFctValue
        lastValueDict['x_2'] = lastFctValue_2
        lastValueDict['x_3'] = lastFctValue_3
        # lastValueDict['x_4'] = lastFctValue_4
        # lastValueDict['x_5'] = lastFctValue_5
        lastValueDict['y'] = ret

        trainData = pd.DataFrame(lastValueDict)
        trainData.dropna(inplace=True)
        X = trainData.loc[:, 'x_1':'x_3']
        y = trainData['y']

        # plt.scatter(X['x_1'], y)
       #  plt.show()

        reg = LinearRegression(fit_intercept=False).fit(X, y)
        params = reg.coef_
        paramsDict[tradingDate] = pd.Series(params)
        y_true = trainData['y']
        y_pred = reg.predict(trainData.loc[:, 'x_1':'x_3'])
        print('Linear mse: {} r_2 {}'.format(mean_squared_error(y_true, y_pred), r2_score(y_true, y_pred)))

        regLasso = Lasso(fit_intercept=False, alpha=.1).fit(X, y)
        y_true = trainData['y']
        y_pred = regLasso.predict(trainData.loc[:, 'x_1':'x_3'])
        print('Lasso mse: {} r_2 {}'.format(mean_squared_error(y_true, y_pred), r2_score(y_true, y_pred)))

        regRidge = Ridge(fit_intercept=False, alpha=.1).fit(X, y)
        y_true = trainData['y']
        y_pred = regRidge.predict(trainData.loc[:, 'x_1':'x_3'])
        print('Ridge mse: {} r_2 {}'.format(mean_squared_error(y_true, y_pred), r2_score(y_true, y_pred)))

        regXgb = xgb.XGBRegressor()
        regXgb.fit(X, y)

        # X = trainData.loc[:, 'x_1':'x_5']
        # svr = SVR(kernel='linear').fit(X, y)
        # y_pred = svr.predict(X)
        # print('SVR Linear {}'.format(mean_squared_error(y_true, y_pred)))
        #
        # svr = SVR(kernel='poly').fit(X, y)
        # y_pred = svr.predict(X)
        # print('SVR poly {}'.format(mean_squared_error(y_true, y_pred)))
        #
        # svr = SVR(kernel='rbf').fit(X, y)
        # y_pred = svr.predict(X)
        # print('SVR rbf {}'.format(mean_squared_error(y_true, y_pred)))
        #
        # svr = SVR(kernel='sigmoid').fit(X, y)
        # y_pred = svr.predict(X)
        # print('SVR sigmoid {}'.format(mean_squared_error(y_true, y_pred)))


    paramsDF = pd.DataFrame(paramsDict).T
    paramsDF_Rolling20 = paramsDF # .rolling(20).mean()
    paramsDF_Rolling20.dropna(inplace=True)
    newFctDict = {}

    newFct = rawFct.loc[paramsDF_Rolling20.index.tolist(), :].mul(paramsDF_Rolling20.iloc[:, 0], axis=0) +\
             (rawFct.loc[paramsDF_Rolling20.index.tolist(), :]**2).mul(paramsDF_Rolling20.iloc[:, 1], axis=0) + \
             (rawFct.loc[paramsDF_Rolling20.index.tolist(), :] ** 3).mul(paramsDF_Rolling20.iloc[:, 2], axis=0)
    newFct.to_csv('results\\fct\BIAS5nonLinear.csv')
    print()