import pandas as pd
import os
import numpy as np
import ast

if __name__ == '__main__':
    # stockCodes = pd.read_csv('database\\stockCode\\stock_code.csv', index_col=0)['代码'].tolist()
    # stockTickers = list(map(lambda x: x[-6:] + '.' + x[:2], stockCodes))
    # fctDates = list(map(lambda x: x[:-4], os.listdir('database\\fctValue')))
    # fctNames = pd.read_csv('database\\fctValue\\2016-01-04.csv', index_col=0).columns
    # fctNames = fctNames[6:].tolist()
    # fctDict = {}
    # for fctDate in fctDates[:10]:
    #     fctValues = pd.read_csv('database\\fctValue\\' + fctDate + '.csv', index_col=0)
    #     fctValues = fctValues.loc[fctValues.status!=0, :]
    #     fctValues.replace(np.inf, np.NaN, inplace=True)
    #     fctValues.fillna(fctValues.mean(), inplace=True)
    #     fctDict[fctDate] = {}
    #     for stockTicker in stockTickers:
    #         if stockTicker in fctValues.index:
    #             fctDict[fctDate][stockTicker] = fctValues.loc[stockTicker, fctNames].tolist()
    # finalDF = pd.DataFrame(fctDict)
    # finalDF.to_csv('database\\fctValueStock\\fctValueStock.csv')

    df = pd.read_csv('database\\fctValueStock\\fctValueStock.csv', index_col=0)
    case1 = ast.literal_eval(df.iloc[0, 0])
    print()