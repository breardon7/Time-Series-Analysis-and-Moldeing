import numpy as np
import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import math
from scipy import signal, linalg
import statsmodels.api as sm
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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
    plt.grid()
    plt.show()

    return auto_corr


def ACF_no_plot(timeseries_data, lags):
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
            y[i] = e[i] + (0.5 * y[i - 1]) + (0.2 * y[i - 2])
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
        k = k + 1

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


def generate_arma():

    np.random.seed(42)
    T = int(input("Enter the number of data samples: "))
    mean_e = float(input("Enter the mean of white noise: "))
    var_e = int(input("Enter the variance of the white noise: "))
    na = int(input("Enter AR order:"))
    nb = int(input("Enter MA order:"))
    AR_coeff = [float(input("Enter the order {} coefficient of AR".format(i))) for i in range(1, na + 1)]
    MA_coeff = [float(input("Enter the order {} coefficient of MA".format(i))) for i in range(1, nb + 1)]
    AR_coeff.insert(0, 1)
    MA_coeff.insert(0, 1)

    arma_process = sm.tsa.ArmaProcess(AR_coeff, MA_coeff)
    mean_y = mean_e * (1 + sum(MA_coeff)) / (1 + sum(AR_coeff))
    y = arma_process.generate_sample(T, scale=np.sqrt(var_e)) + mean_y

    return y


def generate_arma_no_questions(T, mean_e, var_e, na, nb, AR_coeff, MA_coeff):

    np.random.seed(42)
    arma_process = sm.tsa.ArmaProcess(AR_coeff, MA_coeff)
    mean_y = mean_e * (1 + sum(MA_coeff)) / (1 + sum(AR_coeff))
    y = arma_process.generate_sample(T, scale=np.sqrt(var_e)) + mean_y

    return y


def GPAC(acf, len_j, len_k, title="GPAC Table"):

    len_k = len_k + 1   # na starts with 1, nb with 0
    gpac = np.empty(shape=(len_j,   len_k)) # k is x-axis, j is y-axis

    for k in range(1, len_k):   # rows of GPAC table
        num = np.empty(shape=(k, k))
        denom = np.empty(shape=(k, k))
        for j in range(0, len_j):            # columns of GPAC table
            for row in range(0, k):          # rows of kxk matrix
                for column in range(0, k):   # columns of kxk matrix
                    if column < k - 1:       # acf for all except last column
                        num[row][column] = acf[np.abs(j+(row-column))]
                        denom[row][column] = acf[np.abs(j+(row-column))]
                    else:
                        num[row][column] = acf[np.abs(j+row+1)]
                        denom[row][column] = acf[np.abs(j+(row-column))]

            num_determinant = round(np.linalg.det(num), 6)
            denom_determinant = round(np.linalg.det(denom), 6)

            if denom_determinant == 0.0:
                gpac[j][k] = np.inf     # when denominator is 0
            else:
                gpac[j][k] = round((num_determinant/denom_determinant), 2)

    gpac = pd.DataFrame(gpac[:, 1:])                # exclude 0th column as k starts from 1
    gpac.columns = [i for i in range(1, len_k)]     # re-index columns to start from 1

    sns.heatmap(gpac, annot=True)
    plt.title(title)
    plt.show()


def ACF_PACF_Plot(y, lags, title="ACF/PACF"):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    fig.suptitle(title)
    plt.subplot(211)
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.grid()
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    plt.grid()
    fig.tight_layout(pad=3)
    plt.show()


def levenberg_marquardt(y, na, nb):

    def generate_error_terms(theta, na, y):  # calculate errors using dlsim for SSE

        np.random.seed(42)
        den = theta[:na]  # den is na coefficients as theta is stacked na+nb
        num = theta[na:]  # num is nb coefficients as theta is stacked na+nb

        if len(den) > len(num):  # dlsim num and denom formatting
            for x in range(len(den) - len(num)):
                num = np.append(num, 0)
        elif len(num) > len(den):
            for x in range(len(num) - len(den)):
                den = np.append(den, 0)

        den = np.insert(den, 0, 1)
        num = np.insert(num, 0, 1)
        sys = (den, num, 1)
        _, e = signal.dlsim(sys, y)
        return e

    def step1(y, na, nb, delta, theta):  # step1 reusable

        n = na + nb
        e = generate_error_terms(theta, na, y)
        sse_old = np.dot(np.transpose(e), e)

        X = np.empty(shape=(len(y), n))
        for i in range(0, n):               # loop to calculate gradient
            theta[i] = theta[i] + delta
            e_i = generate_error_terms(theta, na, y)
            x_i = (e - e_i) / delta
            X[:, i] = x_i[:, 0]
            theta[i] = theta[i] - delta  # theta reset

        A = np.dot(np.transpose(X), X)
        g = np.dot(np.transpose(X), e)

        return A, g, X, sse_old

    def step2(A, theta, mu, g, na, y):  # step 2 reusable

        delta_theta = np.matmul(linalg.inv(A + (mu * np.identity(A.shape[0]))), g)
        theta_new = theta + delta_theta
        e_new = generate_error_terms(theta_new, na, y)  # calculate new error
        sse_new = np.dot(np.transpose(e_new), e_new)
        if np.isnan(sse_new):
            sse_new = 10 ** 10

        return sse_new, delta_theta, theta_new

    def step3(y, na, nb):

        N = len(y)  # number of samples
        n = na+nb
        mu = 0.01  # step size
        mu_max = 10 ** 20  # max step size (arbitrary)
        max_iterations = 100  # ideal convergence happens within 100 steps
        delta = 10 ** -6  # learning rate
        var_e = 0
        covariance_theta_hat = 0

        sse_list = []

        theta = np.zeros(shape=(n, 1))  # step 0

        for iterations in range(100):
            A, g, X, sse_old = step1(y, na, nb, delta, theta)  # step 1
            sse_new, delta_theta, theta_new = step2(A, theta, mu, g, na, y)  # step 2

            sse_list.append(sse_old[0][0])
            if iterations < max_iterations:

                if sse_new < sse_old:
                    # new parameters better than old parameters
                    if linalg.norm(np.array(delta_theta), 2) < 10 ** -3:
                        theta_hat = theta_new
                        var_e = sse_new / (N - n)
                        # inverse of hessian is covariance, used to calculate confidence
                        covariance_theta_hat = var_e * linalg.inv(A)  # n by n
                        print("Convergence")
                        break
                    else:
                        # new parameters worse than old but algorithm hasn't converged
                        # as mu starts reducing, algorithm goes towards gauss newton method
                        theta = theta_new
                        mu = mu / 10
                while sse_new >= sse_old:
                    # change in learning rate
                    mu = mu * 10
                    if mu > mu_max:
                        print('No Convergence')
                        break
                    sse_new, delta_theta, theta_new = step2(A, theta, mu, g, na, y)
            if iterations > max_iterations:
                print('Maximum Iterations Reached: No Convergence')
                break
            theta = theta_new

        return theta_new, sse_new, var_e, covariance_theta_hat, sse_list

    def confidence_interval(theta, cov):
        print("Confidence Interval of parameters")
        for i in range(len(theta)):
            lb = theta[i] - 2 * np.sqrt(cov[i, i])
            ub = theta[i] + 2 * np.sqrt(cov[i, i])
            print("{} < theta_{} < {}".format(lb, i, ub))

    def find_roots(theta, na):
        den = theta[:na]
        num = theta[na:]
        if len(den) > len(num):
            for x in range(len(den) - len(num)):
                num = np.append(num, 0)
        elif len(num) > len(den):
            for x in range(len(num) - len(den)):
                den = np.append(den, 0)
        else:
            pass

        den = np.insert(den, 0, 1)
        num = np.insert(num, 0, 1)
        print("Roots of numerator:", np.roots(num))
        print("Roots of denominator:", np.roots(den))

    def plot_sse(sse_list):
        plt.plot(sse_list)
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.title('SSE across iterations')
        plt.show()


    # execution step 0,1,2 included in function for step 3
    theta, sse, var_e, covariance_theta_hat, sse_list = step3(y, na, nb)

    print("Coefficients:", theta)
    confidence_interval(theta, covariance_theta_hat)
    print("Covariance Matrix of estimated parameters:\n", covariance_theta_hat)
    print("Estimated variance of error:\n", var_e)
    print(find_roots(theta, na))
    plot_sse(sse_list)
    return theta


