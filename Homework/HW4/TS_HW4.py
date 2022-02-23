import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.holtwinters as ets
import numpy as np
from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split
import Toolbox
import warnings

warnings.filterwarnings(action='ignore')

df = pd.read_csv(r'C:\Users\brear\OneDrive\Desktop\Grad School\Time-Series-Analysis-and-Moldeing\Datasets\AirPassengers.csv')

df['Month'] = pd.to_datetime(df['Month'])
df['#Passengers'].to_numpy()
df['Month'].to_numpy()

train_data, test_data = train_test_split(df['#Passengers'], shuffle=False, test_size=0.2)
test_data.reset_index(drop=True, inplace=True)
train_xaxis, test_xaxis = train_test_split(df['Month'], shuffle=False, test_size=0.2)
test_xaxis.reset_index(drop=True, inplace=True)


# Questions 1-6


# Average Method

am_train_prediction, am_train_error, am_train_MSE, am_train_variance, \
am_test_forecast, am_test_error, am_test_MSE, am_test_variance = Toolbox.average_method(train_data, test_data)
am_Qvalue = Toolbox.box_pierce_test(train_data, am_train_error, 15)

plt.plot(train_xaxis, train_data, label='train data')
plt.plot(test_xaxis, test_data, label='test data')
plt.plot(test_xaxis, am_test_forecast, label='h-step forecast')
plt.xlabel('time')
plt.ylabel('quantity')
plt.title('Average Method')
plt.legend(loc=2)
plt.grid()
plt.show()

# Naive Method

nm_train_prediction, nm_train_error, nm_train_MSE, nm_train_variance, \
nm_test_forecast, nm_test_error, nm_test_MSE, nm_test_variance = Toolbox.naive_method(train_data, test_data)
nm_Qvalue = Toolbox.box_pierce_test(train_data, nm_train_error, 15)

plt.plot(train_xaxis, train_data, label='train data')
plt.plot(test_xaxis, test_data, label='test data')
plt.plot(test_xaxis, nm_test_forecast, label='h-step forecast')
plt.xlabel('time')
plt.ylabel('quantity')
plt.title('Naive Method')
plt.legend(loc=2)
plt.grid()
plt.show()

# Drift Method

dm_train_prediction, dm_train_error, dm_train_MSE, dm_train_variance, \
dm_test_forecast, dm_test_error, dm_test_MSE, dm_test_variance = Toolbox.drift_method(train_data, test_data)
dm_Qvalue = Toolbox.box_pierce_test(train_data, dm_train_error, 15)

plt.plot(train_xaxis, train_data, label='train data')
plt.plot(test_xaxis, test_data, label='test data')
plt.plot(test_xaxis, dm_test_forecast, label='h-step forecast')
plt.xlabel('time')
plt.ylabel('quantity')
plt.title('Drift Method')
plt.legend(loc=2)
plt.grid()
plt.show()

# SES

ses1_train_prediction, ses1_train_error, ses1_train_MSE, ses1_train_variance, \
ses1_test_forecast, ses1_test_error, ses1_test_MSE, ses1_test_variance = Toolbox.SES_method(train_data, test_data, 0.5)
ses1_Qvalue = Toolbox.box_pierce_test(train_data, ses1_train_error, 15)

plt.plot(train_xaxis, train_data, label='train data')
plt.plot(test_xaxis, test_data, label='test data')
plt.plot(test_xaxis, ses1_test_forecast, label='h-step forecast')
plt.xlabel('time')
plt.ylabel('quantity')
plt.title('SES Method')
plt.legend(loc=2)
plt.grid()
plt.show()

# Holt's Linear Method

holtlinear_prediction = ets.ExponentialSmoothing(train_data, trend='mul', damped_trend=True, seasonal=None).fit()
holtlinear_forecast = holtlinear_prediction.forecast(steps=len(test_data))
holtlinear_forecast.reset_index(drop=True, inplace=True)

fitted_holtlinear = holtlinear_prediction.fittedvalues
hl_train_error = train_data - fitted_holtlinear
hl_test_error = test_data - holtlinear_forecast

hl_train_MSE = np.average((np.power(hl_train_error, 2)))
hl_test_MSE = np.average((np.power(hl_test_error, 2)))
hl_train_variance = np.var(hl_train_error)
hl_test_variance = np.var(hl_test_error)
hl_Qvalue = Toolbox.box_pierce_test(train_data, hl_train_error, 15)

plt.plot(train_xaxis, train_data, label='train data')
plt.plot(test_xaxis, test_data, label='test data')
plt.plot(test_xaxis, holtlinear_forecast, label='h-step forecast')
plt.xlabel('time')
plt.ylabel('quantity')
plt.title("Holt's Linear Method")
plt.legend(loc=2)
plt.grid()
plt.show()

# Holt Winter Method

holtwinter_prediction = ets.ExponentialSmoothing(train_data, trend='mul', damped_trend=True, seasonal='mul', seasonal_periods=12).fit()
holtwinter_forecast = holtwinter_prediction.forecast(steps=len(test_data))
holtwinter_forecast.reset_index(drop=True, inplace=True)

fitted_holtwinter = holtwinter_prediction.fittedvalues
hw_train_error = train_data - fitted_holtwinter
hw_test_error = test_data - holtwinter_forecast

hw_train_MSE = np.average((np.power(hw_train_error, 2)))
hw_test_MSE = np.average((np.power(hw_test_error, 2)))
hw_train_variance = np.var(hw_train_error)
hw_test_variance = np.var(hw_test_error)
hw_Qvalue = Toolbox.box_pierce_test(train_data, hw_train_error, 15)

plt.plot(train_xaxis, train_data, label='train data')
plt.plot(test_xaxis, test_data, label='test data')
plt.plot(test_xaxis, holtwinter_forecast, label='h-step forecast')
plt.xlabel('time')
plt.ylabel('quantity')
plt.title("Holt-Winter Seasonal Method")
plt.legend(loc=2)
plt.grid()
plt.show()


# Question 5

Toolbox.ACF(am_test_error, 5, 'Average Method Forecast Error')
plt.show()
Toolbox.ACF(nm_test_error, 5, 'Naive Method Forecast Error')
plt.show()
Toolbox.ACF(dm_test_error, 5, 'Drift Method Forecast Error')
plt.show()
Toolbox.ACF(ses1_test_error, 5, 'SES Method Forecast Error')
plt.show()
Toolbox.ACF(hl_test_error, 5, 'Holt Linear Method Forecast Error')
plt.show()
Toolbox.ACF(hw_test_error, 5, 'Holt-Winter Seasonal Method Forecast Error')
plt.show()


# Question 7

am_corr = Toolbox.correlation_coefficent_cal(am_test_error, test_data)
nm_corr = Toolbox.correlation_coefficent_cal(nm_test_error, test_data)
dm_corr = Toolbox.correlation_coefficent_cal(dm_test_error, test_data)
ses1_corr = Toolbox.correlation_coefficent_cal(ses1_test_error, test_data)
hl_corr = Toolbox.correlation_coefficent_cal(hl_test_error, test_data)
hw_corr = Toolbox.correlation_coefficent_cal(hw_test_error, test_data)


# Question 8

results = pd.DataFrame({'Method': ['Average', 'Naive', 'Drift', 'SES', "Holt's Linear Model", 'Holt-Winter Seasonal Model'],
                        'MSE_forecast': [am_test_MSE, nm_test_MSE, dm_test_MSE, ses1_test_MSE, hl_test_MSE, hw_test_MSE],
                        'Variance of prediction error': [am_train_variance, nm_train_variance,
                                                         dm_train_variance, ses1_train_variance,
                                                         hl_train_variance, hw_train_variance],
                        'Variance of forecast error': [am_test_variance, nm_test_variance,
                                                       dm_test_variance, ses1_test_variance,
                                                       hl_test_variance, hw_test_variance],
                        'Q Values': [am_Qvalue, nm_Qvalue, dm_Qvalue, ses1_Qvalue, hl_Qvalue, hw_Qvalue],
                        'Correlation Coefficient': [am_corr, nm_corr, dm_corr, ses1_corr, hl_corr, hw_corr]})

print(results.to_string())
