import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MultipleLocator
from tools import toolkits

class resultAnalysis:
    def __init__(self, resultName):
        self.result = self.getResultFile(resultName)

    def plot(self):
        benchmark = pd.read_csv('database\\benchmark\\SH000905_TWAP.csv', index_col=0)
        benchmarkRet = benchmark.twap/benchmark.twap.shift(1)-1
        positiveSample = self.result.loc[self.result.pred_prob>=0.5, :]
        negativeSample = self.result.loc[self.result.pred_prob<0.5, :]

        positiveRet = positiveSample.groupby('tradingDate').mean()
        negativeRet = negativeSample.groupby('tradingDate').mean()

        fig = plt.figure(figsize=(15, 10))
        ax1 = fig.subplots(1, 1)

        posAlpha = positiveRet['1dayFutureRetTWAP'].cumsum().shift(2).dropna() - benchmarkRet.loc[positiveRet['1dayFutureRetTWAP'].cumsum().shift(2).dropna().index].cumsum()
        negAlpha = negativeRet['1dayFutureRetTWAP'].cumsum().shift(2).dropna() - benchmarkRet.loc[negativeRet['1dayFutureRetTWAP'].cumsum().shift(2).dropna().index].cumsum()

        ax1.plot(posAlpha, label='posAlpha')
        ax1.plot(negAlpha, label='negAlpha')

        xmajorLocator = MultipleLocator(20)
        ax1.xaxis.set_major_locator(xmajorLocator)
        for tick in ax1.get_xticklabels():
            tick.set_rotation(15)
        plt.legend()
        plt.show()

    def getResultFile(self, resultName):
        files = os.listdir(resultName)
        fileBag = []
        factorDict = {}
        for file in files:
            if ('.csv' in file) and (file[8:18] > '2018-01-01'):
                singleFile = pd.read_csv(resultName + '\\' + file, index_col=0)
                date = singleFile.tradingDate[0]
                factorDict[date] = singleFile['pred_prob']

                # singleFile = singleFile.loc[(singleFile['pred_prob_rank']<=100) | (singleFile['pred_prob_rev_rank']<=100), :]
        factorDF = pd.DataFrame(factorDict).T
        return factorDF


if __name__ == '__main__':
    resAna = resultAnalysis('results\\xgb')
    resAna.result.to_csv('results\\fct\\xgbFactor5day.csv')
    # resAna.resultToFactor()