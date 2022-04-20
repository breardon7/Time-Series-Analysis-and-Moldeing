from statsmodels.tsa.stattools import adfuller, kpss
import pandas as pd
import numpy as np
from sys import platform
import statsmodels.tsa.holtwinters as ets
import seaborn as sns
from matplotlib import pyplot as plt
import warnings

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


def non_seasonal_series_differencing(series, interval=1):
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
