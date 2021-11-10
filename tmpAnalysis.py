import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

if __name__ == '__main__':
    targetFolderPath = 'results\\nn'
    backtestResultDict = {}
    for fileName in os.listdir(targetFolderPath):
        file = pd.read_csv(targetFolderPath + '\\' + fileName, index_col=1)
        backtestResultDict[fileName[3:13]] = {'topRet': file.loc[file.YHat.nlargest(100).index.tolist(), '1dayFutureRetTWAP'].mean(),
                                              'bottomRet': file.loc[file.YHat.nsmallest(100).index.tolist(), '1dayFutureRetTWAP'].mean()}
    backtestResult = pd.DataFrame(backtestResultDict).T.shift(2).dropna(how='all')
    benchmark = benchmark = pd.read_csv('database\\benchmark\\SH000905_TWAP.csv', index_col=0)
    benchmarkRet = benchmark.twap/benchmark.twap.shift(1)-1
    backtestResult['benchmark'] = benchmarkRet[backtestResult.index.tolist()]

    backtestResult = backtestResult.loc['2018-01-01':, :]
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.subplots(1, 1)

    ax1.plot(backtestResult.topRet.cumsum() - backtestResult.benchmark.cumsum(), label='topAlpha')
    ax1.plot(backtestResult.bottomRet.cumsum()- backtestResult.benchmark.cumsum(), label='bottomAlpha')

    xmajorLocator = MultipleLocator(20)
    ax1.xaxis.set_major_locator(xmajorLocator)
    for tick in ax1.get_xticklabels():
        tick.set_rotation(15)
    plt.legend()
    plt.show()
    print(backtestResult.cumsum())