import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import matplotlib.pyplot as plt
import pandas as pd

def plot_rolling_mean_var(data, title):
    rolling_var = []
    for i in range(len(data)):
        # iter_var = data.loc[:i].var()
        rolling_var.append(np.var(data[0:i]))

    rolling_mean = []
    for i in range(len(data)):
        # iter_mean = data.loc[:i].mean()
        rolling_mean.append(np.mean(data[0:i]))

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle(title)
    ax1.plot(rolling_mean)
    ax1.set_title('Rolling Mean')
    ax2.plot(rolling_var)
    ax2.set_title('Rolling Variance')
    plt.tight_layout
    plt.show()

def ADF_Cal(x):
 result = adfuller(x)
 print("ADF Statistic: %f" %result[0])
 print('p-value: %f' % result[1])
 print('Critical Values:')
 for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','LagsUsed'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print(kpss_output)


def difference(dataset, interval=1):
    diff = []
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff

def correlation_coefficent_cal(x,y):
    numer = 0
    denom_x = 0
    denom_y = 0
    for i in range(len(x)):
        numer += ((x[i] - np.mean(x)) * (y[i] - np.mean(y)))
        denom_x += (x[i] - np.mean(x))**2
        denom_y += (y[i] - np.mean(y))**2
    r = numer / (np.sqrt(denom_x) * np.sqrt(denom_y))
    print(r)


