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


def average_method(train_data, test_data):
    train_prediction = []
    train_error = []
    test_forecast = []
    test_error = []

    for i in range(1, len(train_data)):
        average = np.average(train_data[:i])
        train_prediction.append(average)
        train_error.append(train_data[i] - average)

    forecast_average = np.average(train_data)

    for i in range(0, len(test_data)):
        test_forecast.append(forecast_average)
        test_error.append(test_data[i] - forecast_average)

    train_MSE = np.average((np.power(train_error, 2)))
    test_MSE = np.average((np.power(test_error, 2)))
    train_variance = np.var(train_error)
    test_variance = np.var(test_error)

    return train_prediction, train_error, train_MSE, train_variance, test_forecast, test_error, test_MSE, test_variance


def naive_method(train_data, test_data):
    train_prediction = []
    train_error = []
    test_forecast = []
    test_error = []

    for i in range(1, len(train_data)):
        naive_prediction = train_data[i - 1]
        train_prediction.append(naive_prediction)
        train_error.append(train_data[i] - naive_prediction)

    naive_forecast = train_data[-1]

    for i in range(0, len(test_data)):
        test_forecast.append(naive_forecast)
        test_error.append(test_data[i] - naive_forecast)

    train_MSE = np.average((np.power(train_error, 2)))
    test_MSE = np.average((np.power(test_error, 2)))
    train_variance = np.var(train_error)
    test_variance = np.var(test_error)

    return train_prediction, train_error, train_MSE, train_variance, test_forecast, test_error, test_MSE, test_variance


def drift_method(train_data, test_data):
    train_prediction = []
    train_error = []
    test_forecast = []
    test_error = []
    train_length = len(train_data)

    for i in range(1, train_length):
        if i == 1:
            denominator = 1
        else:
            denominator = i-1
        drift_prediction = train_data[i-1]+1*(train_data[i-1]-train_data[0])/denominator
        train_prediction.append(drift_prediction)
        train_error.append(train_data[i] - drift_prediction)

    for i in range(1, len(test_data)+1):
        drift_forecast = train_data[-1] + i*(train_data[-1]-train_data[0])/(train_length-1)
        test_forecast.append(drift_forecast)
        test_error.append(test_data[i-1] - drift_forecast)

    train_MSE = np.average((np.power(train_error, 2)))
    test_MSE = np.average((np.power(test_error, 2)))
    train_variance = np.var(train_error)
    test_variance = np.var(test_error)

    return train_prediction, train_error, train_MSE, train_variance, test_forecast, test_error, test_MSE, test_variance


def SES_method(train_data, test_data, alpha):
    train_prediction = []
    train_prediction.append(train_data[0])  #initial condition
    train_error = []
    test_forecast = []
    test_error = []

    for i in range(1, len(train_data)):
        ses_prediction = alpha*train_data[i-1] + (1-alpha)*train_prediction[-1]
        train_prediction.append(ses_prediction)
        train_error.append(train_data[i] - ses_prediction)

    ses_forecast = alpha * train_data[-1] + (1 - alpha) * train_prediction[-1]

    for i in range(0, len(test_data)):
        test_forecast.append(ses_forecast)
        test_error.append(test_data[i] - ses_forecast)


    train_MSE = np.average((np.power(train_error, 2)))
    test_MSE = np.average((np.power(test_error, 2)))
    train_variance = np.var(train_error)
    test_variance = np.var(test_error)

    return train_prediction, train_error, train_MSE, train_variance, test_forecast, test_error, test_MSE, test_variance

def box_pierce_test(train_data, train_error, lags):
    auto_corr = []
    train_data_length = len(train_data)
    train_error_mean = np.mean(train_error)
    train_error_length = len(train_error)
    denominator = 0

    for denom in range(0, train_error_length):
        denominator = denominator + (train_error[denom] - train_error_mean) ** 2

    for tau in range(1, lags+1):  # starts from 1 because auto correlation of 0th lag is 1 which should be excluded in Box-Pierce test
        numerator = 0
        for num in range(tau, train_error_length):
            numerator = numerator + (train_error[num] - train_error_mean) * (
                    train_error[num - tau] - train_error_mean)
        auto_corr.append((numerator / denominator))

    Q = train_data_length * sum(np.power(auto_corr, 2))
    return Q