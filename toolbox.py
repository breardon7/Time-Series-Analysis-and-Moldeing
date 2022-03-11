import numpy as np
import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import math


def cal_rolling_mean_var(y_data, x_data, metric='', unit=''):
    df = pd.DataFrame(columns=['Rolling_Mean', 'Rolling_Variance'])
    for i in range(1, len(y_data) + 1):
        df.loc[i] = [y_data[0:i].mean(), y_data[0:i].var()]

    plt.plot(x_data, df.Rolling_Mean, label=f'Rolling Mean - {metric}')
    plt.title(f'{metric} - Rolling Mean')
    plt.xlabel('Time')
    plt.ylabel(f'Mean - {unit}')
    plt.legend(loc=4)
    plt.grid()
    plt.show()

    plt.plot(x_data, df.Rolling_Variance, label=f'Rolling Variance - {metric}')
    plt.title(f'{metric} - Rolling Variance')
    plt.xlabel('Time')
    plt.ylabel(f'Variance - {unit}^2')
    plt.legend(loc=4)
    plt.grid()
    plt.show()

    return df


def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" % result[0])
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


def difference(data):
    diff = list()
    for i in range(1, len(data)):
        value = data[i] - data[i - 1]
        diff.append(value)
    return diff


def correlation_coefficent_cal(x, y):
    numerator = 0
    denom_x = 0
    denom_y = 0
    xmean = np.mean(x)
    ymean = np.mean(y)
    for i in range(0, len(x)):
        numerator = numerator + ((x[i] - xmean) * (y[i] - ymean))
        denom_x = denom_x + (x[i] - xmean) ** 2
        denom_y = denom_y + (y[i] - ymean) ** 2
    correlation_coefficent_r = numerator / (math.sqrt(denom_x) * math.sqrt(denom_y))
    return correlation_coefficent_r


def ACF(timeseries_data, lags, metric=''):
    auto_corr = []
    timeseries_data_mean = np.mean(timeseries_data)
    length = len(timeseries_data)
    denominator = 0     # 0th lag adjusted
    x_axis = np.arange(0, lags+1)
    m = 1.96/np.sqrt(length)

    for denom_t in range(0, length):
        denominator = denominator + (timeseries_data[denom_t] - timeseries_data_mean) ** 2

    for tau in range(0, lags+1):
        numerator = 0
        for num_t in range(tau, length):
            numerator = numerator + (timeseries_data[num_t] - timeseries_data_mean) * (
                        timeseries_data[num_t - tau] - timeseries_data_mean)
        auto_corr.append((numerator / denominator))

    plt.stem(x_axis, auto_corr, use_line_collection=True)
    plt.stem(-1 * x_axis, auto_corr, use_line_collection=True)
    plt.title(f"ACF Plot - {metric}")
    plt.xlabel("Lags")
    plt.ylabel("ACF")
    plt.axhspan(-m, m, alpha=0.2, color='blue')
    # use plt.show() in main for graphs to show
    return auto_corr


def average_method(train_data, test_data):
    train_prediction = []
    train_error = []
    test_forecast = []
    test_error = []

    for i in range(0, len(train_data)):
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

    naive_forecast = train_data[len(train_data)-1]

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
        drift_forecast = train_data[train_length-1] + i*(train_data[train_length-1]-train_data[0])/(train_length-1)
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
        ses_prediction = alpha*train_data[i-1] + (1-alpha)*train_prediction[len(train_prediction)-1]
        train_prediction.append(ses_prediction)
        train_error.append(train_data[i] - ses_prediction)

    ses_forecast = alpha * train_data[len(train_data)-1] + (1 - alpha) * train_prediction[len(train_prediction)-1]

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

def simulate_MA(array):
    while True:
        try:
            m1 = int(input("Enter first MA: "))
            break
        except ValueError:
            print('Please enter a single integer value.')
            continue
    print("MA 1:", m1)

    # first order MA

    length = len(array)
    t1 = np.zeros(length)
    if m1 % 2 == 0:
        upper_trail = int(m1 / 2)
        lower_trail = int(m1 / 2 - 1)
    else:
        upper_trail = int((m1 - 1) / 2)
        lower_trail = int((m1 - 1) / 2)

    for t in range(length):
        if t > 0:
            t1[t] = sum(array[t - lower_trail:t + upper_trail]) / m1

    # second order MA

    t2 = np.zeros(len(t1))
    if m1 % 2 == 0:
        while True:
            try:
                m2 = int(input("Enter second MA: "))
                break
            except ValueError:
                print('Please enter a single integer value.')
                continue
        print("MA 2:", m2)

    if m1 % 2 == 0:
        upper_trail = int(m1 / 2)
        lower_trail = int(m1 / 2 - 1)
    else:
        upper_trail = int((m1 - 1) / 2)
        lower_trail = int((m1 - 1) / 2)

    for t in range(len(t1)):
        if t > 0:
            t2[t] = sum(t1[t-lower_trail:t+upper_trail])/m1

    if m1 % 2 == 0:
        return t1, t2
    else:
        return t1