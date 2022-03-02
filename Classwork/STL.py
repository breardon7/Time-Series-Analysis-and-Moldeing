import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import STL

df = pd.read_csv(r'C:\Users\brear\OneDrive\Documents\GitHub\Time-Series-Analysis-and-Modeling\Datasets\daily-min-temperatures.csv',
                 header = 0,
                 index_col = 0)

temp = df['Temp']
temp = pd.Series(np.array(df['Temp']), index=pd.date_range('1981-01-01', periods=len(temp), freq='d', name='daily temperatures'))
temp.plot()
plt.show()


STL = STL(temp)
res = STL.fit()
fig = res.plot()
plt.show()

T = res.trend
S = res.seasonal
R = res.resid

adj_seasonal = temp - S
F = np.maximum(0,1 - np.var(array(R)/np.var(np.array(S)+np.array(R))))
print(f'Strength of seasonality is {F}')

detrended_temp = temp - T
F = np.maximum(0,1 - np.var(np.array(R)/np.var()))
print(f'Strength of trend is {F}')