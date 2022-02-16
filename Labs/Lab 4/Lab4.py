import matplotlib.pyplot as plt
import numpy as np
from statistics import variance
import statsmodels.api as sm
from toolbox import autocorrelation_plot

t_train = [112, 118, 132, 129, 121, 134, 148, 136, 119]
t_test = [104, 118, 115, 126, 141]
combined = t_train + t_test


# question 2

print(combined)
def average_forecast(train, test, type):
    train_forecast = list()
    test_forecast = list()
    train_forecast.append(train[0])
    for i in range(1, len(train) + 1):
        train_forecast.append(np.mean(train[0:i]))
    for i in range(0, len(test)):
        test_forecast.append(train_forecast[-1])

    if type == 'train':
        return train_forecast
    elif type == 'test':
        return test_forecast

print('----------------------average---------------------')
print(average_forecast(t_train, t_test, type='train'))
print(average_forecast(t_train, t_test, type='test'))

plt.plot(np.arange(1,10), t_train, c='b', label='train')
plt.plot(np.arange(9,14), t_test, c='r', label='test')
plt.plot(np.arange(9,14), average_forecast(t_train, t_test, type='test'), c='g', label='forecast')
plt.legend(loc = 'upper left')
plt.xlabel('Time (t)')
plt.ylabel('Series Values')
plt.title('Average Forecast')
plt.show()

# question 3

def MSE(original, prediction):
    error_squared = 0
    for i in range(len(original)):
        error = original[i] - prediction[i]
        error_squared += error ** 2
    mse = error_squared/len(original)
    return mse

print('average train MSE:', MSE(t_train, average_forecast(t_train, t_test, type='train')))
print('average test MSE:', MSE(t_test, average_forecast(t_train, t_test, type='test')))


# question 4
print('average train mean:', np.mean(average_forecast(t_train, t_test, type = 'train')))
print('average test mean:', np.mean(average_forecast(t_train, t_test, type = 'test')))
print('average train variance:', variance(average_forecast(t_train, t_test, type='train')))
print('average test variance:', variance(average_forecast(t_train, t_test, type='test')))

# question 5
print('Q average:', sm.stats.acorr_ljungbox(average_forecast(t_train, t_test, type='train'), lags=[5], return_df=True))

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


print('----------------------naive---------------------')
print(naive(t_train))
print(naive(t_test))
print('naive train MSE:', MSE(t_train, naive(t_train)))
print('naive test MSE:', MSE(t_test, np.ones(5)*t_train[-1]))

print('naive train mean:', np.mean(naive(t_train)))
print('naive test mean:', np.mean(naive(t_test)))
print('naive train variance:', variance(naive(t_train)))
print('naive test variance:', variance(np.ones(5)*t_train[-1]))

print('Q naive:', sm.stats.acorr_ljungbox(naive(t_train), lags=[5], return_df=True))

# question 7

def drift(train, test, type):
    train_forecast = list()
    test_forecast = list()
    train_forecast.append(train[0])
    for i in range(1, len(train) + 1):
        train_forecast.append(train[i - 1] + (train[i - 1] - train[0]) / i)
    for i in range(0, len(test)):
        test_forecast.append(train[-1] + ((train[-1] - train[0]) / (len(train))) * (i + 1))

    if type == 'train':
        return train_forecast
    elif type == 'test':
        return test_forecast



plt.plot(np.arange(1,10), t_train, c='b', label='train')
plt.plot(np.arange(9,14), t_test, c='r', label='test')
plt.plot(np.arange(9,14), drift(t_train, t_test, type = 'test'), c='g', label='forecast')
plt.legend(loc = 'upper left')
plt.xlabel('Time (t)')
plt.ylabel('Series Values')
plt.title('Drift Forecast')
plt.show()

print('----------------------drift---------------------')
print(drift(t_train, t_test, type = 'train'))
print(drift(t_train, t_test, type = 'test'))
print('drift train MSE:', MSE(t_train, drift(t_train, t_test, type = 'train')))
print('drift test MSE:', MSE(t_test, drift(t_train, t_test, type = 'test')))

print('drift train mean:', np.mean(drift(t_train, t_test, type = 'train')))
print('drift test mean:', np.mean(drift(t_train, t_test, type = 'test')))
print('drift train variance:', variance(drift(t_train, t_test, type = 'train')))
print('drift test variance:', variance(drift(t_train, t_test, type = 'test')))

print('Q drift:', sm.stats.acorr_ljungbox(drift(t_train, t_test, type = 'train'), lags=[5], return_df=True))

# question 8

def ses(train, test, type, alpha=None):
    train_forecast = list()
    test_forecast = list()

    if (alpha < 0) or (alpha > 1):
        raise ValueError('Alpha value has to be integer/float between 0 and 1.')
    train_forecast.append(train[0])
    for i in range(1, len(train) + 1):
        train_forecast.append(alpha * train[i - 1] + (1 - alpha) * train_forecast[i - 1])
    test_forecast.append(alpha * train[-1] + (1 - alpha) * train_forecast[-1])
    for i in range(1, len(test)):
        test_forecast.append(alpha * train[-1] + (1 - alpha) * train_forecast[-1])

    if type == 'train':
        return train_forecast
    elif type == 'test':
        return test_forecast


plt.plot(np.arange(1,10), t_train, c='b', label='train')
plt.plot(np.arange(9,14), t_test, c='r', label='test')
plt.plot(np.arange(9,14), ses(t_train, t_test, type='test', alpha=0.5), c='g', label='forecast')
plt.legend(loc = 'upper left')
plt.xlabel('Time (t)')
plt.ylabel('Series Values')
plt.title('SES Forecast')
plt.show()

print('----------------------ses---------------------')
print(ses(t_train, t_test, type='train', alpha=0.5))
print(ses(t_train, t_test, type='test', alpha=0.5))
print('SES train MSE:', MSE(t_train, ses(t_train, t_test, type='train', alpha=0.5)))
print('SES test MSE:', MSE(t_test, ses(t_train, t_test, type='test', alpha=0.5)))

print('SES train mean:', np.mean(ses(t_train, t_test, type = 'train', alpha=0.5)))
print('SES test mean:', np.mean(ses(t_train, t_test, type = 'test', alpha=0.5)))
print('SES train variance:', variance(ses(t_train, t_test, type='train', alpha=0.5)))
print('SES test variance:', variance(ses(t_train, t_test, type='test', alpha=0.5)))

print('Q SES:', sm.stats.acorr_ljungbox(ses(t_train, t_test, type='train', alpha=0.5), lags=[5], return_df=True))

# question 9
fig, ax = plt.subplots(2, 2)
ax1, ax2, ax3, ax4 = ax.flatten()
fig.suptitle('SES Forecast')
ax1.plot(np.arange(1,10), t_train, c='b', label='train')
ax1.plot(np.arange(9,14), t_test, c='r', label='test')
ax1.plot(np.arange(9,14), ses(t_train, t_test, type='test', alpha=0), c='g', label='forecast: alpha=0')
ax1.legend(loc='upper left')
ax2.plot(np.arange(1,10), t_train, c='b', label='train')
ax2.plot(np.arange(9,14), t_test, c='r', label='test')
ax2.plot(np.arange(9,14), ses(t_train, t_test, type='test', alpha=0.25), c='g', label='forecast: alpha=0.25')
ax2.legend(loc='upper left')
ax3.plot(np.arange(1,10), t_train, c='b', label='train')
ax3.plot(np.arange(9,14), t_test, c='r', label='test')
ax3.plot(np.arange(9,14), ses(t_train, t_test, type='test', alpha=0.75), c='g', label='forecast: alpha=0.75')
ax3.legend(loc='upper left')
ax4.plot(np.arange(1,10), t_train, c='b', label='train')
ax4.plot(np.arange(9,14), t_test, c='r', label='test')
ax4.plot(np.arange(9,14), ses(t_train, t_test, type='test', alpha=0.99), c='g', label='forecast alpha=0.99')
ax4.legend(loc='upper left')
fig.text(0.5, 0.04, 'Time(t)', ha='center')
fig.text(0.04, 0.5, 'Series Values', va='center', rotation='vertical')

plt.show()

# question 11
autocorrelation_plot(np.array(average_forecast(t_train, t_test, type='train')), title='ACF Average Prediction', lag=9)
autocorrelation_plot(np.array(naive(t_train)), title='ACF Naive Prediction', lag=8)
autocorrelation_plot(np.array(drift(t_train, t_test, type='train')), title='ACF Drift Prediction', lag=9)
autocorrelation_plot(np.array(ses(t_train, t_test, type='train', alpha=0.5)), title='ACF SES Prediction', lag=9)
