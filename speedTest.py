import pandas as pd
import time
from tools import toolkits
import numpy as np
# csv_case = toolkits().readRawData('Stock_RD_StdWBuy_rolling_10', 'database\\rawFct')
# time1 = time.time()
# csv_case = csv_case.loc['2019-11-25':'2019-11-29', :]
# toolkits().getZScore(csv_case, 1)
# time2 = time.time()
# print(time2-time1)
# time3 = time.time()
# pickle_case = pd.read_pickle('database\\fileType\\Stock_RD_StdWBuy_rolling_10.pkl', compression='gzip')
# time4 = time.time()
# print(time4 - time3)


dataFrame1 = pd.DataFrame([1,2,3,54,3425,234,5243,5624,56]*1000, index=list(range(len([1,2,3,54,3425,234,5243,5624,56]*1000))), columns=['a'])
dataFrame2 = pd.DataFrame([1,2,3,54,3425,234,5243,5624,56]*1000, index=list(range(len([1,2,3,54,3425,234,5243,5624,56]*1000))), columns=['b'])


time3 = time.time()
dataFrame1['b'] = np.NaN
dataFrame1['b'] = dataFrame2['b'].tolist()
time4 = time.time()
print(time4 - time3)

time1 = time.time()
c = pd.concat([dataFrame1, dataFrame2], axis=0)
time2 = time.time()
print(time2 - time1)

