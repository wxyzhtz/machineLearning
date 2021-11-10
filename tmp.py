import numpy as np
import pandas as pd
from tools import toolkits
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
df = pd.read_csv('database\\rawFct\\Stock_RD_StdWBuy.csv', index_col=0)
df.replace(0, np.NaN, inplace=True)
qt = QuantileTransformer(n_quantiles=1000)
df = pd.DataFrame(qt.fit_transform(df.T.values).T, index=df.index.tolist(), columns=df.columns.tolist())
for index, row in df.iterrows():
    plt.hist(row.replace(0, np.NaN).dropna(), bins=1000)
    plt.show()
print(df.T.describe())
