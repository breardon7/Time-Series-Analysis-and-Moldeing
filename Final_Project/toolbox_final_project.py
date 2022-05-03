import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss

def difference(dataset, interval):
   diff = []
   for i in range(interval, len(dataset)):
      value = dataset[i] - dataset[i - interval]
      diff.append(value)
   return diff

def rolling_mv(y, title):
    means = []
    vars = []

    for i in range(1, len(y)):
        means.append(y[:i].mean())
        vars.append(y[:i].var())

    # plot rolling mean
    plt.figure()
    plt.subplot(211)
    plt.plot(means, label= 'Mean')
    plt.title("Rolling Mean "  + str(title))
    plt.xlabel('Time')
    plt.ylabel("Rolling Mean")
    plt.subplot(212)
    plt.plot(vars, label= 'Variance')
    plt.title("Rolling Variance " + str(title))
    plt.xlabel('Time')
    plt.ylabel("Rolling Variance")
    plt.tight_layout()
    plt.show()

def autocorr(x, lag):
    l = range(lag + 1)
    x_br = np.mean(x)
    autocorr = []
    for i in l:
        num = 0
        var = 0
        for j in range(i, len(x)):
            num += np.sum(x[j] - x_br) * (x[j - i] - x_br)
        var = np.sum((x - x_br) ** 2)
        autocorr.append(num / var)
    return autocorr

def autocorr(x, lag):
    l = range(lag + 1)
    x_br = np.mean(x)
    autocorr = []
    for i in l:
        num = 0
        var = 0
        for j in range(i, len(x)):
            num += np.sum(x[j] - x_br) * (x[j - i] - x_br)
        var = np.sum((x - x_br) ** 2)
        autocorr.append(num / var)
    return autocorr

def ACF_Plot(x, lag):
    lg = np.arange(-lag, lag + 1)
    x = x[0:lag + 1]
    rx = x[::-1]
    rxx = rx[:-1] + x
    plt.stem(lg, rxx)

def PACF_ACF_Plot(x,lags,title):
    plt.figure()
    plt.subplot(211)
    plt.xlabel('Lags')
    plt.ylabel('ACF Value')
    plt.title('ACF and PACF plot ' + str(title))
    sm.graphics.tsa.plot_acf(x, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plt.xlabel('Lags')
    plt.ylabel('PACF Value')
    sm.graphics.tsa.plot_pacf(x, ax=plt.gca(), lags=lags)
    plt.tight_layout()
    plt.show()

def ADF_Cal(x):
    result = adfuller(x)

    print("ADF Statistic: %f" %result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

def kpss_test(timeseries):
    print('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'LagsUsed'])
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value
        print(kpss_output)

def mse(errors):
   return  np.sum(np.power(errors,2))/len(errors)