import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statistics import variance
import statsmodels.api as sm
from scipy.stats import linregress

t_train = [112, 118, 132, 129, 121, 134, 148, 163, 119]
t_test = [104, 118, 115, 126, 141]
combined = t_train + t_test

# question 2

print(combined)
def average_forecast(original):
    prediction = []
    for i in range(len(original)):
        prediction.append(round((sum(original[:i+1])/(i+1))))
    return prediction

print(average_forecast(t_train))
print(average_forecast(t_test))
print(average_forecast(combined))

plt.plot(np.arange(1,10), t_train, c='b', label='train')
plt.plot(np.arange(9,14), t_test, c='r', label='test')
plt.plot(np.arange(9,14),average_forecast(combined)[-5:], c='g', label='forecast')
plt.legend(loc = 'upper left')
plt.xlabel('Time (t)')
plt.ylabel('Series Values')
plt.title('Average Forecast')
plt.show()

# question 3

def error(original, prediction):
    error = []
    for i in range(len(original)):
        error.append(original[i] - prediction[i])
    return error

def error_squared(error):
    error_squared = []
    for i in range(len(error)):
        error_squared.append(error[i] ** 2)
    return error_squared

def MSE(original, prediction):
    error_squared = 0
    for i in range(len(original)):
        error = original[i] - prediction[i]
        error_squared += error ** 2
    mse = error_squared/len(original)
    return mse

print('average train MSE:', MSE(t_train, average_forecast(t_train)))
print('average test MSE:', MSE(t_test, average_forecast(t_test)))


# question 4

print('average train variance:', variance(average_forecast(t_train)))
print('average test variance:', variance(average_forecast(t_test)))

# question 5
#print('Q average:', sm.stats.acorr_ljungbox(average_forecast(t_train), lags=[5], return_df=True))

# question 6

def naive(x):
    predicted = []
    predicted.append(x[0])
    for i in range(1, len(x)):
        predicted.append(x[i-1])
    return predicted

plt.plot(np.arange(1,10), t_train, c='b', label='train')
plt.plot(np.arange(9,14), t_test, c='r', label='test')
plt.plot(np.arange(9,14),np.ones(5)*t_train[-1], c='g', label='forecast')
plt.legend(loc = 'upper left')
plt.xlabel('Time (t)')
plt.ylabel('Series Values')
plt.title('Naive Forecast')
plt.show()

print('naive train MSE:', MSE(t_train, naive(t_train)))
print('naive test MSE:', MSE(t_test, np.ones(5)*t_train[-1]))

print('naive train variance:', variance(naive(t_train)))
print('naive test variance:', variance(np.ones(5)*t_train[-1]))

print('Q naive:', sm.stats.acorr_ljungbox(naive(t_train), lags=[5], return_df=True))

# question 7

res = linregress(np.arange(1,len(combined)+1), combined)

plt.plot(np.arange(1,10), t_train, c='b', label='train')
plt.plot(np.arange(9,14), t_test, c='r', label='test')
plt.plot(np.arange(9,14), t_train[-1] + np.arange(9,14)*res.slope, c='g', label='forecast')
plt.legend(loc = 'upper left')
plt.xlabel('Time (t)')
plt.ylabel('Series Values')
plt.title('Drift Forecast')
plt.show()

print('drift train MSE:', MSE(t_train, naive(t_train)))
print('drift test MSE:', MSE(t_test, np.ones(5)*t_train[-1]))

print('drift train variance:', variance(naive(t_train)))
print('drift test variance:', variance(np.ones(5)*t_train[-1]))

print('Q drift:', sm.stats.acorr_ljungbox(naive(t_train), lags=[5], return_df=True))
