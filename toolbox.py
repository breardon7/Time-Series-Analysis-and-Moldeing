import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm

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

def autocorrelation(x, lag=1):
    n = len(x)
    variance = x.var()
    x = x - x.mean()
    r = np.correlate(x, x, mode='full')[-n:]
    assert np.allclose(r, np.array([(x[:n - k] * x[-(n - k):]).sum() for k in range(n)]))
    result = r / (variance * (np.arange(n, 0, -1)))

    plt.acorr(result, maxlags = lag)
    plt.title('Autocorrelation')
    plt.ylabel('Magnitude')
    plt.xlabel('Lags')
    plt.grid(True)
    plt.show()

def acf_graph_cal(series, lags):
    ybar = np.mean(series)
    denom = 0
    for y in series:
        denom += (y - ybar) ** 2
    num = list()
    # 0 order for ACF is always 1
    num.append(denom)
    for i in range(1, lags+1):
        lag_num = 0
        for j in range(0, len(series) - i):
            value = (series[j] - ybar) * (series[j+i] - ybar)
            lag_num += value
        num.append(lag_num)


    acf_vals = num / denom
    rev = acf_vals[::-1][:-1]
    acf_vals = np.concatenate([rev, acf_vals])
    integer_range = np.concatenate([np.arange(-1*lags, 0), np.arange(0, lags + 1)])
    plt.stem(integer_range, acf_vals)
    plt.fill_between(integer_range,
                     1.96 / np.sqrt(len(series)), -1.96 / np.sqrt(len(series)),
                     alpha=0.1, color='red')
    plt.xlabel('Lags')
    plt.ylabel('ACF Values')
    plt.title(f'ACF Plot with Lags of {lags}.')
    plt.show()

    return integer_range, acf_vals

def autocorrelation_plot(x, lag=1, title=''):
    n = len(x)
    variance = x.var()
    x = x - x.mean()
    r = np.correlate(x, x, mode='full')[-n:]
    assert np.allclose(r, np.array([(x[:n - k] * x[-(n - k):]).sum() for k in range(n)]))
    result = r / (variance * (np.arange(n, 0, -1)))

    plt.acorr(result, maxlags = lag)
    plt.title(title)
    plt.ylabel('Magnitude')
    plt.xlabel('Lags')
    plt.grid(True)
    plt.show()


def forecasting(train, test, type, alpha=None):
    train_forecast = list()
    test_forecast = list()
    if type == 'average':
        train_forecast.append(train[0])
        for i in range(1, len(train) + 1):
            train_forecast.append(np.mean(train[0:i]))
        for i in range(0, len(test)):
            test_forecast.append(train_forecast[-1])
    elif type == 'naive':
        train_forecast.append(train[0])
        for i in range(1, len(train) + 1):
            train_forecast.append(train[i-1])
        for i in range(0, len(test)):
            test_forecast.append(train_forecast[-1])
    elif type == 'drift':
        train_forecast.append(train[0])
        for i in range(1, len(train) + 1):
            train_forecast.append(train[i-1] + (train[i-1] - train[0]) / i)
        for i in range(0, len(test)):
            test_forecast.append(train[-1] + ((train[-1] - train[0]) / (len(train))) * (i + 1))
    elif type == 'ses':
        if (alpha < 0) or (alpha > 1):
            raise ValueError('Alpha value has to be integer/float between 0 and 1.')
        train_forecast.append(train[0])
        for i in range(1, len(train) + 1):
            train_forecast.append(alpha * train[i-1] + (1-alpha) * train_forecast[i-1])
        test_forecast.append(alpha * train[-1] + (1-alpha) * train_forecast[-1])
        for i in range(1, len(test)):
            test_forecast.append(alpha * train[-1] + (1-alpha) * train_forecast[-1])
    else:
        raise ValueError(f'Value of {type} is not a valid value for type variable.')

    train_error = [a - b for a, b in zip(train_forecast, train)]
    test_error = [a - b for a, b in zip(test_forecast, test)]


    train_labels = [x + 1 for x in list(range(0, len(train)))]
    test_labels = [x + max(train_labels) for x in list(range(0, len(test)))]
    plt.plot(train_labels, train, c='blue', label='train')
    plt.plot(test_labels, test, c='orange', label='test')
    plt.plot(test_labels, test_forecast, c='green', label='forecast')
    plt.xlabel('Time (t)')
    plt.ylabel('Series Value')
    #plt.legend([train, test, test_forecast], ['train', 'test', 'forecast'])
    plt.legend()
    if type == 'ses':
        plt.title(f'Time Series Value and Prediction using {type} method and alpha of {alpha}')
    else:
        plt.title(f'Time Series Value and Prediction using {type} method')
    plt.show()

    # Plot acf of errors
    acf_graph_cal(train_error, lags=5)

    print(f'The train values are {train}')
    print(f'The train predictions are {train_forecast}')
    print(f'The train error is {train_error}')
    print(f'The train squared error is {np.square(train_error)}')
    print(f'The train MSE is {np.mean(np.square(train_error))}')
    print(f'The train error variance is {np.var(train_error)}')
    print(f'The Q values for the training set are {sm.stats.acorr_ljungbox(train_forecast, lags=5)}')
    print(f'The test values are {test}')
    print(f'The test forecasts are {test_forecast}')
    print(f'The test error is {test_error}')
    print(f'The test squared error is {np.square(test_error)}')
    print(f'The test MSE is {np.mean(np.square(test_error))}')
    print(f'The test error variance is {np.var(test_error)}')

    #return train, test, train_forecast, test_forecast
    return None
