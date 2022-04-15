import math
from scipy import signal
from statsmodels.tsa.stattools import adfuller, kpss
import pandas as pd
import numpy as np
from sys import platform
import statsmodels.tsa.holtwinters as ets
import seaborn as sns
from matplotlib import pyplot as plt
import warnings


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
    denominator = 0  # 0th lag adjusted
    x_axis = np.arange(0, lags + 1)
    m = 1.96 / np.sqrt(length)

    for denom_t in range(0, length):
        denominator = denominator + (timeseries_data[denom_t] - timeseries_data_mean) ** 2

    for tau in range(0, lags + 1):
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

    naive_forecast = train_data[len(train_data) - 1]

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
            denominator = i - 1
        drift_prediction = train_data[i - 1] + 1 * (train_data[i - 1] - train_data[0]) / denominator
        train_prediction.append(drift_prediction)
        train_error.append(train_data[i] - drift_prediction)

    for i in range(1, len(test_data) + 1):
        drift_forecast = train_data[train_length - 1] + i * (train_data[train_length - 1] - train_data[0]) / (
                train_length - 1)
        test_forecast.append(drift_forecast)
        test_error.append(test_data[i - 1] - drift_forecast)

    train_MSE = np.average((np.power(train_error, 2)))
    test_MSE = np.average((np.power(test_error, 2)))
    train_variance = np.var(train_error)
    test_variance = np.var(test_error)

    return train_prediction, train_error, train_MSE, train_variance, test_forecast, test_error, test_MSE, test_variance


def SES_method(train_data, test_data, alpha):
    train_prediction = []
    train_prediction.append(train_data[0])  # initial condition
    train_error = []
    test_forecast = []
    test_error = []

    for i in range(1, len(train_data)):
        ses_prediction = alpha * train_data[i - 1] + (1 - alpha) * train_prediction[len(train_prediction) - 1]
        train_prediction.append(ses_prediction)
        train_error.append(train_data[i] - ses_prediction)

    ses_forecast = alpha * train_data[len(train_data) - 1] + (1 - alpha) * train_prediction[len(train_prediction) - 1]

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

    for tau in range(1,
                     lags + 1):  # starts from 1 because auto correlation of 0th lag is 1 which should be excluded in Box-Pierce test
        numerator = 0
        for num in range(tau, train_error_length):
            numerator = numerator + (train_error[num] - train_error_mean) * (
                    train_error[num - tau] - train_error_mean)
        auto_corr.append((numerator / denominator))

    Q = train_data_length * sum(np.power(auto_corr, 2))
    return Q


def moving_average(data, m1, m2):
    ma = np.empty(len(data))
    ma[:] = np.NaN

    if m1 < 2:
        print('Invalid Order')
    elif m1 % 2 != 0:
        k = m1 // 2
        for i in range(k, len(data) - k):
            ma[i] = np.mean(data[i - k:i + k + 1])
    else:
        if m2 % 2 == 1 or m2 < 2:
            print("Invalid Folding Order")
        else:
            k1 = m1 // 2
            ma_intermediate = np.empty(len(data))
            ma_intermediate[:] = np.NaN
            for i in range(k1 - 1, len(data) - k1):
                ma_intermediate[i] = np.mean(data[i - k1 + 1:i + k1 + 1])
            k2 = m2 // 2
            for i in range(k2 - 1, len(ma_intermediate) - k2):
                ma[i + 1] = np.mean(ma_intermediate[i - k2 + 1:i + k2 + 1])
    return ma


def plot_detrended(data, ma, detrended, xlabel, ylabel, title, serieslabel):
    plt.figure(figsize=[8, 5])
    plt.plot_date(data.index, data, ls='solid', label='Original', marker='')
    plt.plot_date(data.index, ma, ls='solid', label=serieslabel, marker='')
    plt.plot_date(data.index, detrended, ls='solid', label='Detrended', marker='')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def simulate_AR(mean, std, T):

    np.random.seed(42)
    e = np.random.normal(mean, std, size=T)
    y = np.zeros(len(e))

    for i in range(len(e)):
        if i == 0:
            y[0] = e[0]
        elif i == 1:
            y[1] = e[1] + (0.5 * y[0])
        else:
            y[i] = + e[i] + (0.5 * y[i - 1]) + (0.2 * y[i - 2])
    return y


def least_square_estimate(y, samples, order):

    np.random.seed(42)
    T_prime = samples - order - 1
    X = pd.DataFrame([-1 * y[1:T_prime + 2], -1 * y[0:T_prime + 1]]).T

    X_transpose = np.transpose(X)

    temp_1 = np.linalg.inv(X_transpose.dot(X))
    temp_2 = X_transpose.dot(y[order:])
    lse = temp_1.dot(temp_2)

    print(lse)
    return lse


def generalized_least_square_estimate(samples, order, ARparam):

    # generate y
    mean = 0
    std = np.sqrt(1)
    np.random.seed(42)
    e = np.random.normal(mean, std, size=samples)
    y = np.zeros(len(e))

    num = [1, 0, 0]
    num = [1]
    num.extend([0] * order)
    den = [1]
    den.extend(ARparam)
    sys = (num, den, 1)
    _, y = signal.dlsim(sys, e)
    y = [item for sublist in y for item in sublist]

    T_prime = samples - order - 1

    X = np.zeros((T_prime + 1, order))
    # y_l = list(y)

    k = 1
    for j in range(order):
        for i in range(T_prime + 1):
            X[i][j] = -1 * y[order + i - k]
        k += 1
    # X = pd.DataFrame([-1*y[1:T_prime+2],-1*y[0:T_prime+1]]).T

    X_transpose = np.transpose(X)
    temp_1 = np.linalg.inv(X_transpose.dot(X))
    temp_2 = X_transpose.dot(y[order:])
    lse = temp_1.dot(temp_2)
    return lse


def simulate_MA(T):

    np.random.seed(42)
    e = np.random.normal(0, 1, size=T)
    y = np.zeros(len(e))

    for i in range(len(e)):
        if i == 0:
            y[0] = e[0]
        elif i == 1:
            y[1] = e[1] + (0.5 * e[0])
        else:
            y[i] = e[i] + (0.5 * e[i - 1]) + (0.2 * e[i - 2])
    return y

def gpac_calc(ry, na, nb):
    result = pd.DataFrame()
    k1 = []
    k2 = []
    k3 = []
    k4 = []
    k5 = []
    k6 = []
    k7 = []
    k8 = []
    k9 = []
    k10 = []
    for k in range(1, nb):
        for j in range(na):
            if k == 1:
                k1.append(round(ry[j + k] / ry[j], 3))
            if k == 2:
                numerator2 = [[ry[j], ry[j + 1]], [ry[j + k - 1], ry[j + k]]]
                denominator2 = [[ry[j], ry[abs(j - k + 1)]], [ry[abs(j + k - 1)], ry[j]]]
                k2.append(round(np.linalg.det(numerator2) / np.linalg.det(denominator2), 3))
            if k == 3:
                numerator3 = [[ry[j], ry[abs(j - 1)], ry[j + 1]],
                              [ry[j + 1], ry[j], ry[j + 2]],
                              [ry[j + k - 1], ry[j + k - 2], ry[j + k]]]
                denominator3 = [[ry[j], ry[abs(j - 1)], ry[abs(j - k + 1)]], [ry[j + 1], ry[j], ry[abs(j - k + 2)]],
                                [ry[j + k - 1], ry[j + k - 2], ry[j]]]
                k3.append(round(np.linalg.det(numerator3) / np.linalg.det(denominator3), 3))
            if k == 4:
                numerator4 = [[ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[j + 1]],
                              [ry[j + 1], ry[j], ry[abs(j - 1)], ry[j + 2]],
                              [ry[j + 2], ry[j + 1], ry[j], ry[j + 3]],
                              [ry[j + k - 1], ry[j + k - 2], ry[j + k - 3], ry[j + k]]]

                denominator4 = [[ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - k + 1)]],
                                [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - k + 2)]],
                                [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - k + 3)]],
                                [ry[j + k - 1], ry[j + k - 2], ry[j + k - 3], ry[j]]]

                k4.append(round(np.linalg.det(numerator4) / np.linalg.det(denominator4), 3))
            if k == 5:
                numerator5 = [[ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[j + 1]],
                              [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[j + 2]],
                              [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[j + 3]],
                              [ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j + 4]],
                              [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)], ry[j + k]]]

                denominator5 = [[ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - k + 1)]],
                                [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - k + 2)]],
                                [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - k + 3)]],
                                [ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - k + 4)]],
                                [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)], ry[j]]]
                k5.append(round(np.linalg.det(numerator5) / np.linalg.det(denominator5), 3))
            if k == 6:
                numerator6 = [[ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[j + 1]],
                              [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[j + 2]],
                              [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[j + 3]],
                              [ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[j + 4]],
                              [ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j + 5]],
                              [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)], ry[abs(j + k - 5)], ry[j + k]]]
                denominator6 = [
                    [ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[abs(j - k + 1)]],
                    [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - k + 2)]],
                    [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - k + 3)]],
                    [ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - k + 4)]],
                    [ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - k + 5)]],
                    [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)], ry[abs(j + k - 5)], ry[j]]]
                k6.append(round(np.linalg.det(numerator6) / np.linalg.det(denominator6), 3))

            if k == 7:
                numerator7 = [
                    [ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[abs(j - 5)], ry[j + 1]],
                    [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[j + 2]],
                    [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[j + 3]],
                    [ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[j + 4]],
                    [ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[j + 5]],
                    [ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j + 6]],
                    [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)], ry[abs(j + k - 5)], ry[abs(j + k - 6)],
                     ry[j + k]]]
                denominator7 = [[ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[abs(j - 5)],
                                 ry[abs(j - k + 1)]],
                                [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)],
                                 ry[abs(j - k + 2)]],
                                [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],
                                 ry[abs(j - k + 3)]],
                                [ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)],
                                 ry[abs(j - k + 4)]],
                                [ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - k + 5)]],
                                [ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - k + 6)]],
                                [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)], ry[abs(j + k - 5)],
                                 ry[abs(j + k - 6)], ry[j]]]

                k7.append(round(np.linalg.det(numerator7) / np.linalg.det(denominator7), 3))

            if k == 8:
                numerator8 = [[ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[abs(j - 5)],
                               ry[abs(j - 6)], ry[j + 1]],
                              [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)],
                               ry[abs(j - 5)], ry[j + 2]],
                              [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],
                               ry[abs(j - 4)], ry[j + 3]],
                              [ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],
                               ry[j + 4]],
                              [ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)],
                               ry[j + 5]],
                              [ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[j + 6]],
                              [ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j + 7]],
                              [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)], ry[abs(j + k - 5)], ry[abs(j + k - 6)],
                               ry[abs(j + k - 7)], ry[j + k]]]

                denominator8 = [[ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[abs(j - 5)],
                                 ry[abs(j - 6)], ry[abs(j - k + 1)]],
                                [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)],
                                 ry[abs(j - 5)], ry[abs(j - k + 2)]],
                                [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],
                                 ry[abs(j - 4)], ry[abs(j - k + 3)]],
                                [ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],
                                 ry[abs(j - k + 4)]],
                                [ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)],
                                 ry[abs(j - k + 5)]],
                                [ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)],
                                 ry[abs(j - k + 6)]],
                                [ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],
                                 ry[abs(j - k + 7)]],
                                [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)], ry[abs(j + k - 5)],
                                 ry[abs(j + k - 6)], ry[abs(j + k - 7)], ry[j]]]

                k8.append(round(np.linalg.det(numerator8) / np.linalg.det(denominator8), 3))

            if k == 9:
                numerator9 = [[ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[abs(j - 5)],
                               ry[abs(j - 6)], ry[abs(j - 7)], ry[j + 1]],
                              [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)],
                               ry[abs(j - 5)], ry[abs(j - 6)], ry[j + 2]],
                              [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],
                               ry[abs(j - 4)], ry[abs(j - 5)], ry[j + 3]],
                              [ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],
                               ry[abs(j - 4)],
                               ry[j + 4]],
                              [ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)],
                               ry[abs(j - 3)],
                               ry[j + 5]],
                              [ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)],
                               ry[abs(j - 2)], ry[j + 6]],
                              [ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)],
                               ry[j + 7]],
                              [ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],
                               ry[j + 8]],
                              [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)], ry[abs(j + k - 5)], ry[abs(j + k - 6)],
                               ry[abs(j + k - 7)], ry[abs(j + k - 8)], ry[j + k]]]

                denominator9 = [[ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[abs(j - 5)],
                                 ry[abs(j - 6)], ry[abs(j - 7)], ry[abs(j - k + 1)]],
                                [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)],
                                 ry[abs(j - 5)], ry[abs(j - 6)], ry[abs(j - k + 2)]],
                                [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],
                                 ry[abs(j - 4)], ry[abs(j - 5)], ry[abs(j - k + 3)]],
                                [ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],
                                 ry[abs(j - 4)],
                                 ry[abs(j - k + 4)]],
                                [ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)],
                                 ry[abs(j - 3)],
                                 ry[abs(j - k + 5)]],
                                [ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)],
                                 ry[abs(j - 2)], ry[abs(j - k + 6)]],
                                [ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],
                                 ry[abs(j - 1)],
                                 ry[abs(j - k + 7)]],
                                [ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],
                                 ry[abs(j - k + 8)]],
                                [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)], ry[abs(j + k - 5)],
                                 ry[abs(j + k - 6)],
                                 ry[abs(j + k - 7)], ry[abs(j + k - 8)], ry[j]]]

                k9.append(round(np.linalg.det(numerator9) / np.linalg.det(denominator9), 3))

            if k == 10:
                numerator10 = [[ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[abs(j - 5)],
                                ry[abs(j - 6)], ry[abs(j - 7)], ry[abs(j - 8)], ry[j + 1]],
                               [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)],
                                ry[abs(j - 5)], ry[abs(j - 6)], ry[abs(j - 7)], ry[j + 2]],
                               [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],
                                ry[abs(j - 4)], ry[abs(j - 5)], ry[abs(j - 6)], ry[j + 3]],
                               [ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],
                                ry[abs(j - 4)], ry[abs(j - 5)],
                                ry[j + 4]],
                               [ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)],
                                ry[abs(j - 3)], ry[abs(j - 4)],
                                ry[j + 5]],
                               [ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)],
                                ry[abs(j - 2)], ry[abs(j - 3)], ry[j + 6]],
                               [ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)],
                                ry[abs(j - 2)],
                                ry[j + 7]],
                               [ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],
                                ry[abs(j - 1)],
                                ry[j + 8]],
                               [ry[j + 8], ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],
                                ry[j],
                                ry[j + 9]],
                               [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)], ry[abs(j + k - 5)],
                                ry[abs(j + k - 6)],
                                ry[abs(j + k - 7)], ry[abs(j + k - 8)], ry[abs(j + k - 9)], ry[j + k]]]

                denominator10 = [[ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[abs(j - 5)],
                                  ry[abs(j - 6)], ry[abs(j - 7)], ry[abs(j - 8)], ry[abs(j - k + 1)]],
                                 [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)],
                                  ry[abs(j - 5)], ry[abs(j - 6)], ry[abs(j - 7)], ry[abs(j - k + 2)]],
                                 [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],
                                  ry[abs(j - 4)], ry[abs(j - 5)], ry[abs(j - 6)], ry[abs(j - k + 3)]],
                                 [ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)],
                                  ry[abs(j - 3)],
                                  ry[abs(j - 4)], ry[abs(j - 5)],
                                  ry[abs(j - k + 4)]],
                                 [ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)],
                                  ry[abs(j - 3)], ry[abs(j - 4)],
                                  ry[abs(j - k + 5)]],
                                 [ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)],
                                  ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - k + 6)]],
                                 [ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],
                                  ry[abs(j - 1)], ry[abs(j - 2)],
                                  ry[abs(j - k + 7)]],
                                 [ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],
                                  ry[abs(j - 1)],
                                  ry[abs(j - k + 8)]],
                                 [ry[j + 8], ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],
                                  ry[j + 1], ry[j],
                                  ry[abs(j - k + 9)]],
                                 [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)], ry[abs(j + k - 5)],
                                  ry[abs(j + k - 6)],
                                  ry[abs(j + k - 7)], ry[abs(j + k - 8)], ry[abs(j + k - 9)], ry[j]]]

                k10.append(round(np.linalg.det(numerator10) / np.linalg.det(denominator10), 3))
    if len(k1) > 0:
        result["1"] = k1
    if len(k2) > 0:
        result["2"] = k2
    if len(k3) > 0:
        result["3"] = k3
    if len(k4) > 0:
        result["4"] = k4
    if len(k5) > 0:
        result["5"] = k5
    if len(k6) > 0:
        result["6"] = k6
    if len(k7) > 0:
        result["7"] = k7
    if len(k8) > 0:
        result["8"] = k8
    if len(k9) > 0:
        result["9"] = k9
    sns.heatmap(result, annot=True, fmt='.3f')
    plt.title("Generalized Partial Autocorrelation(GPAC) Table")
    plt.show()
    print(result)

def auto_correlation_cal(series, lags):
    y = np.array(series).copy()
    y_mean = np.mean(series)
    correlation = []
    for lag in np.arange(1, lags + 1):
        numerator_part_1 = y[lag:] - y_mean
        numerator_part_2 = y[:-lag] - y_mean
        numerator = sum(numerator_part_1 * numerator_part_2)
        denominator = sum((y - y_mean) ** 2)
        correlation.append(numerator / denominator)
    return pd.Series(correlation)