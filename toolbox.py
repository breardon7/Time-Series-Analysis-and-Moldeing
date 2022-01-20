from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import matplotlib.pyplot as plt
import pandas as pd

def plot_rolling_mean_var(data, title):
    rolling_var = []
    for i in range(len(data)):
        iter_var = data.loc[:i].var()
        rolling_var.append(iter_var)

    rolling_mean = []
    for i in range(len(data)):
        iter_mean = data.loc[:i].mean()
        rolling_mean.append(iter_mean)


    fig_sales, (ax1, ax2) = plt.subplots(2)
    fig_sales.suptitle(title)
    ax1.plot(data, rolling_mean)
    ax1.set_title('Rolling Mean')
    ax2.plot(data, rolling_var)
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

# def corr_coef(x, y):


