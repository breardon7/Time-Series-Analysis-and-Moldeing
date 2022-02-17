import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.holtwinters as ets
import numpy as np
from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split
from toolbox import average_forecast, naive_forecast, drift_forecast, ses_forecast, MSE, autocorrelation_plot

df = pd.read_csv(r'C:\Users\brear\OneDrive\Desktop\Grad School\Time-Series-Analysis-and-Moldeing\Datasets\AirPassengers.csv')
X = df['Month']
y = df['#Passengers']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=100)

# question 1

print('----------------------average---------------------')
plt.plot(np.arange(1,10), t_train, c='b', label='train')
plt.plot(np.arange(9,14), t_test, c='r', label='test')
plt.plot(np.arange(9,14), average_forecast(t_train, t_test, type='test'), c='g', label='forecast')
plt.legend(loc = 'upper left')
plt.xlabel('Time (t)')
plt.ylabel('Series Values')
plt.title('Average Forecast')
plt.show()

print('average train MSE:', MSE(t_train, average_forecast(t_train, t_test, type='train')))
print('average test MSE:', MSE(t_test, average_forecast(t_train, t_test, type='test')))
print('average train mean:', np.mean(average_forecast(t_train, t_test, type = 'train')))
print('average test mean:', np.mean(average_forecast(t_train, t_test, type = 'test')))
print('average train variance:', variance(average_forecast(t_train, t_test, type='train')))
print('average test variance:', variance(average_forecast(t_train, t_test, type='test')))
print('Q average:', sm.stats.acorr_ljungbox(average_forecast(t_train, t_test, type='train'), lags=[5], return_df=True))

print('----------------------naive---------------------')
plt.plot(np.arange(1,10), t_train, c='b', label='train')
plt.plot(np.arange(9,14), t_test, c='r', label='test')
plt.plot(np.arange(9,14),np.ones(5)*t_train[-1], c='g', label='forecast')
plt.legend(loc = 'upper left')
plt.xlabel('Time (t)')
plt.ylabel('Series Values')
plt.title('Naive Forecast')
plt.show()

print(naive_forecast(t_train))
print(naive_forecast(t_test))
print('naive train MSE:', MSE(t_train, naive_forecast(t_train)))
print('naive test MSE:', MSE(t_test, np.ones(5)*t_train[-1]))

print('naive train mean:', np.mean(naive_forecast(t_train)))
print('naive test mean:', np.mean(naive_forecast(t_test)))
print('naive train variance:', variance(naive_forecast(t_train)))
print('naive test variance:', variance(np.ones(5)*t_train[-1]))

print('Q naive:', sm.stats.acorr_ljungbox(naive_forecast(t_train), lags=[5], return_df=True))