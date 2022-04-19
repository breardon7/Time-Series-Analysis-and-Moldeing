from statsmodels.tsa.stattools import adfuller, kpss
import pandas as pd
import numpy as np
from sys import platform
import statsmodels.tsa.holtwinters as ets
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from numpy import inner, max, diag, eye, Inf, dot
from numpy.linalg import norm, solve

warnings.filterwarnings('ignore')
# i added this security line for mac users because the keychain access blocks unverified ssl from the dataset url
if platform == "darwin":
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context


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


def cal_rolling_mean_var(series):
    rolling_means = []
    rolling_variance = []
    start_index = 0
    observation_size = 2
    for i in range(len(series)):
        mean = np.mean(series)
        variance = np.var(series)
        for j in range(len(series)):
            observation_data = series[start_index: start_index + observation_size]
            mean = observation_data.mean()
            variance = observation_data.var()
        rolling_means.append(mean)
        rolling_variance.append(variance)
        start_index += 1

    df = pd.DataFrame()
    df["Rolling Mean"] = rolling_means
    df["Rolling Variance"] = rolling_variance
    return df


def series_differencing(series, interval=1):
    differencing = list()
    for i in range(interval, len(series)):
        value = series[i] - series[i - interval]
        differencing.append(value)
    return pd.Series(differencing)


def correlation_coefficient_cal(x, y):
    coefficient_numerator = 0
    coefficient_denominator_part_1 = 0
    coefficient_denominator_part_2 = 0
    for i in range(len(x)):
        coefficient_numerator += (x[i] - np.mean(x)) * (y[i] - np.mean(y))
        coefficient_denominator_part_1 += (x[i] - np.mean(x)) ** 2
        coefficient_denominator_part_2 += (y[i] - np.mean(y)) ** 2

    coefficient_denominator = np.sqrt(coefficient_denominator_part_1) * np.sqrt(coefficient_denominator_part_2)
    coefficient = coefficient_numerator / coefficient_denominator
    return np.round(coefficient, 3)


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


def avg_one_step_ahead_prediction(y):
    predictions = []
    for i in range(len(y)):
        if i > 0:
            predictions.append(np.mean(y[0:i]))
    return predictions


def naive_one_step_ahead_prediction(y):
    predictions = y[0:len(y) - 1]
    return predictions


def drift_one_step_prediction(y):
    predictions = []
    for t in range(1, len(y)):
        if t - 1 <= 0:
            predictions.append(y[t - 1])
        else:
            yt = y[t - 1]
            m = (yt - y[0]) / (t - 1)
            h = 1
            predictions.append(yt + m * h)
    return predictions


def ses_one_step_prediction(y):
    predictions = []
    for i in range(1, len(y)):
        if i <= 1:
            predictions.append(y[0])
        else:
            yt = y[0:i]
            ses = ets.ExponentialSmoothing(yt, trend=None, damped_trend=False, seasonal=None).fit(smoothing_level=0.5)
            ses_prediction_model = ses.forecast(steps=1)
            ses_predictions = pd.Series(ses_prediction_model).drop_duplicates()
            predictions.append(ses_predictions)

    return pd.Series(predictions)


def error_calc(y0, y):
    errors = []
    for i in range(len(y)):
        errors.append(y0[i] - y[i])
    return errors


def n_error_calc(y0, y, n):
    errors = []
    for i in range(n):
        errors.append(y0[i] - y[i])
    return errors


def box_pierce_test(series, lags):
    r = auto_correlation_cal(series, lags)
    rk = []
    for value in r:
        rk.append(value ** 2)
    return np.sum(rk) * len(series)


def mse_calc(y0, y):
    mse = (np.square(y0 - y)).mean()
    return mse


def lse_calc(X, Y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)


def variance_estimator_calc(errors, T, k):
    sum_squared_error = 0
    for i in errors:
        sum_squared_error += i ** 2
    return np.sqrt((1 / (T - k - 1)) * sum_squared_error)


def moving_average(numbers, order):
    i = 0
    moving_averages = []
    while i < len(numbers) - order + 1:
        current_order = numbers[i: i + order]
        order_average = sum(current_order) / order
        moving_averages.append(order_average)
        i += 1
    return moving_averages


def seasonality_strength(rt, st_rt):
    a = (np.var(rt) / np.var(st_rt))
    return np.max(a, axis=0)


def trend_strength(rt, tt_rt):
    a = (np.var(rt) / np.var(tt_rt))
    return np.max(a, axis=0)


def ar_calc(inputs, order, e):
    y = np.zeros(len(e))
    for i in range(len(e)):
        if i == 0:
            y[0] = e[0]
        else:
            deduction = 1
            order_calc = e[i]
            for j in range(order):
                yt = y[i - deduction]
                order_calc += inputs[j] * yt
                deduction += 1
            y[i] = order_calc

    return y


def ma_calc(inputs, order, e):
    y = np.zeros(len(e))
    for i in range(len(e)):
        if i == 0:
            y[0] = e[0]
        elif i == 1:
            y[i] = inputs[i - 1] * e[i - 1] + e[i]
        else:
            deduction = 1
            order_calc = e[i]
            for j in range(order):
                et = e[i - deduction]
                order_calc += inputs[j] * et
                deduction += 1
                y[i] = order_calc
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
    for k in range(nb):
        for j in range(na):
            if k == 1:
                numerator1 = ry[j + k]
                denominator1 = ry[j]
                k1.append(numerator1 / denominator1)
            if k == 2:
                numerator2 = [[ry[j], ry[j + 1]], [ry[abs(j + k - 1)], ry[j + k]]]
                denominator2 = [[ry[j], ry[abs(j - k + 1)]], [ry[abs(j + k - 1)], ry[j]]]
                k2.append(np.linalg.det(numerator2) / np.linalg.det(denominator2))
            if k == 3:
                numerator3 = [[ry[j], ry[abs(j - 1)], ry[j + 1]],
                              [ry[j + 1], ry[j], ry[j + 2]],
                              [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[j + k]]]
                denominator3 = [[ry[j], ry[abs(j - 1)], ry[abs(j - k + 1)]], [ry[j + 1], ry[j], ry[abs(j - k + 2)]],
                                [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[j]]]
                k3.append(np.linalg.det(numerator3) / np.linalg.det(denominator3))
            if k == 4:
                numerator4 = [[ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[j + 1]],
                              [ry[j + 1], ry[j], ry[abs(j - 1)], ry[j + 2]],
                              [ry[j + 2], ry[j + 1], ry[j], ry[j + 3]],
                              [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[j + k]]]

                denominator4 = [[ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - k + 1)]],
                                [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - k + 2)]],
                                [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - k + 3)]],
                                [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[j]]]

                k4.append(np.linalg.det(numerator4) / np.linalg.det(denominator4))
            if k == 5:
                numerator5 = [[ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[j + 1]],
                              [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[j + 2]],
                              [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[j + 3]],
                              [ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j + 4]],
                              [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)],
                               ry[j + k]]]

                denominator5 = [[ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - k + 1)]],
                                [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - k + 2)]],
                                [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - k + 3)]],
                                [ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - k + 4)]],
                                [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)], ry[j]]]
                k5.append(np.linalg.det(numerator5) / np.linalg.det(denominator5))
            if k == 6:
                numerator6 = [[ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[j + 1]],
                              [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[j + 2]],
                              [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[j + 3]],
                              [ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[j + 4]],
                              [ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j + 5]],
                              [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)],
                               ry[abs(j + k - 5)], ry[j + k]]]
                denominator6 = [
                    [ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[abs(j - k + 1)]],
                    [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - k + 2)]],
                    [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - k + 3)]],
                    [ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - k + 4)]],
                    [ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - k + 5)]],
                    [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)], ry[abs(j + k - 5)],
                     ry[j]]]
                k6.append(np.linalg.det(numerator6) / np.linalg.det(denominator6))

            if k == 7:
                numerator7 = [
                    [ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[abs(j - 5)], ry[j + 1]],
                    [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[j + 2]],
                    [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[j + 3]],
                    [ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[j + 4]],
                    [ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[j + 5]],
                    [ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j + 6]],
                    [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)], ry[abs(j + k - 5)],
                     ry[abs(j + k - 6)],
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
                                [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)],
                                 ry[abs(j + k - 5)],
                                 ry[abs(j + k - 6)], ry[j]]]

                k7.append(np.linalg.det(numerator7) / np.linalg.det(denominator7))

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
                              [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)],
                               ry[abs(j + k - 5)], ry[abs(j + k - 6)],
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
                                [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)],
                                 ry[abs(j + k - 5)],
                                 ry[abs(j + k - 6)], ry[abs(j + k - 7)], ry[j]]]

                k8.append(np.linalg.det(numerator8) / np.linalg.det(denominator8))

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
                              [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)],
                               ry[abs(j + k - 5)], ry[abs(j + k - 6)],
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
                                [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)],
                                 ry[abs(j + k - 5)],
                                 ry[abs(j + k - 6)],
                                 ry[abs(j + k - 7)], ry[abs(j + k - 8)], ry[j]]]

                k9.append(np.linalg.det(numerator9) / np.linalg.det(denominator9))

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
                               [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)],
                                ry[abs(j + k - 5)],
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
                                 [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)],
                                  ry[abs(j + k - 5)],
                                  ry[abs(j + k - 6)],
                                  ry[abs(j + k - 7)], ry[abs(j + k - 8)], ry[abs(j + k - 9)], ry[j]]]

                k10.append(np.linalg.det(numerator10) / np.linalg.det(denominator10))
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
    if len(k10) > 0:
        result["10"] = k10
    sns.heatmap(result, annot=True, fmt='.3f')
    plt.title("Generalized Partial Autocorrelation(GPAC) Table")
    plt.show()


def line_error(params, args):
    x, y = args
    m, b = params[0:2]
    y_star = m * x + b

    return y - y_star


def numerical_differentiation(params, args, error_function):
    delta_factor = 1e-4
    min_delta = 1e-4

    # Compute error
    y_0 = error_function(params, args)

    # Jacobian
    J = np.empty(shape=(len(params),) + y_0.shape, dtype=np.float)

    for i, param in enumerate(params):
        params_star = params[:]
        delta = param * delta_factor

        if abs(delta) < min_delta:
            delta = min_delta

        # Update single param and calculate error with updated value
        params_star[i] += delta
        y_1 = error_function(params_star, args)

        # Update Jacobian with gradients
        diff = y_0 - y_1
        J[i] = diff / delta

    return J


def LM(seed_params, args,
       error_function, jacobian_function=numerical_differentiation,
       llambda=1e-3, lambda_multiplier=10, kmax=500):
    params = seed_params
    errors = []

    k = 0
    while k < kmax:
        k += 1

        # Retrieve jacobian of function gradients with respect to the params
        J = jacobian_function(params, args, error_function)
        JtJ = inner(J, J)

        # I * diag(JtJ)
        A = eye(len(params)) * diag(JtJ)

        # == Jt * error
        error = error_function(params, args)
        Jerror = inner(J, error)
        rmserror = norm(error)
        error_star = error[:]
        rmserror_star = rmserror + 1
        errors.append(error_star)
        while rmserror_star >= rmserror:
            try:
                delta = solve(JtJ + llambda * A, Jerror)
            except np.linalg.LinAlgError:
                return -1

            # Update params and calculate new error
            params_star = params[:] + delta[:]
            error_star = error_function(params_star, args)
            rmserror_star = norm(error_star)
            errors.append(error_star)
            if rmserror_star < rmserror:
                params = params_star
                llambda /= lambda_multiplier
                break

            llambda *= lambda_multiplier
    return np.array(errors[0:1]).flatten()


def ACF_PACF_Plot(y, lags):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    plt.subplot(211)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()


def ACF_PACF_Plot(y, lags, title):
    fig = plt.figure()
    plt.subplot(211)
    plt.title(title)
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()


def one_step_ahead_prediction(y_train, teta):
    y_hat_t_1 = []
    for i in range(0, len(y_train)):
        if i == 0:
            y_hat_t_1.append(-y_train[i] * teta[0] + teta[2] * y_train[i])
        elif i == 1:
            y_hat_t_1.append(
                -y_train[i] * teta[0] - teta[1] * y_train[i - 1] + teta[2] * (y_train[i] - y_hat_t_1[i - 1]) + teta[
                    3] * (y_train[i - 1]))
        else:
            y_hat_t_1.append(
                -y_train[i] * teta[0] - teta[1] * y_train[i - 1] + teta[2] * (y_train[i] - y_hat_t_1[i - 1]) + teta[
                    3] * (y_train[i - 1] - y_hat_t_1[i - 2]))
    return y_hat_t_1


def h_step_ahead_prediction(y_train, y_test, teta):
    y_hat_t_1 = one_step_ahead_prediction(y_train, teta)
    y_hat_t_h = []
    for h in range(0, len(y_test)):
        if h == 0:
            y_hat_t_h.append(
                -y_train[-1] * teta[0] - teta[1] * y_train[-2] + teta[2] * (y_train[-1] - y_hat_t_1[-2]) + teta[3] * (
                        y_train[-2] - y_hat_t_1[-3]))
        elif h == 1:
            y_hat_t_h.append(
                -y_hat_t_h[h - 1] * teta[0] - teta[1] * y_train[-1] + teta[3] * (y_train[-1] - y_hat_t_1[-2]))

        else:
            y_hat_t_h.append(-y_hat_t_h[h - 1] * teta[0] - teta[1] * y_hat_t_h[h - 2])
    return y_hat_t_h