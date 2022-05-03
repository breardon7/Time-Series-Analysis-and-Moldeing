import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import toolbox_final_project as toolbox
from statsmodels.tsa.seasonal import STL
import statsmodels.tsa.holtwinters as ets
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',FutureWarning)

# read the data
df = pd.read_csv(r'C:\Users\brear\OneDrive\Desktop\Grad School\Time-Series-Analysis-and-Moldeing\Final_Project\AAPL.csv')
df['Date']=pd.to_datetime(df['Date'])
df.set_index('Date', inplace = True)
print(df.head())
print(df.shape)

# Move Target to first column
target = 'Close'
first_col = df.pop(target)
df.insert(0, target, first_col)

# 6.a: plot the Appliance vs time----------------------------------------------------


# applied second order differencing
Diff_1 = toolbox.difference(df.Close, 1)
Diff_2 = toolbox.difference(df.Close, 1)
# Diff_1 = np.log(df.Close)
DF = pd.DataFrame(Diff_1, index=df.index[1:])
DF.rename(columns={0:'Close'}, inplace=True)
print(DF)


plt.figure()
plt.subplot(211)
plt.plot(df.index, df.Close)
plt.xlabel('Date')
plt.xticks(df.index[:10287], fontsize= 10)
plt.ylabel('Close Price (USD')
plt.title('Close Price Over Time')
plt.subplot(212)
plt.plot(DF.index, DF.Close)
plt.xlabel('Date')
plt.xticks(DF.index[::10287], fontsize= 10)
plt.ylabel('Close Price (USD)')
plt.title('Differenced Close Price Over Time')
plt.tight_layout()
plt.show()

# =============== #
# Rolling Average #
# =============== #

# plot rolling average
toolbox.rolling_mv(df.Close, 'of Original Data')
toolbox.rolling_mv(DF.Close, 'of Differenced Data')

# 6.b: ACF/PACF plots ACF Plot----------------------------------------------------

# ACF plot of raw data
plt.figure()
bound = 1.96/np.sqrt(len(df.Close))
toolbox.ACF_Plot(toolbox.autocorr(df.Close,90),90)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Close Values (USD) 90 Lags')
plt.axhspan(-bound,bound,alpha = .1, color = 'black')
plt.tight_layout()
plt.show()

# ACF plot of differenced data
plt.figure()
bound = 1.96/np.sqrt(len(DF.Close))
toolbox.ACF_Plot(toolbox.autocorr(DF.Close,90),90)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Differenced Close Values (USD) with 90 Lags')
plt.axhspan(-bound,bound,alpha = .1, color = 'black')
plt.tight_layout()
plt.show()

toolbox.PACF_ACF_Plot(df.Close,500, 'of Raw Data')
toolbox.PACF_ACF_Plot(DF.Close,500, 'of Differenced Data')

# Note: use ARMA model based on ACF/PACF drop off

# 6c: Matrix Correlation
# Q2: Heat Map
plt.figure()
corr_t = df.corr()
# create correlation coefficient heatmap for x,y,g, and z
ax = sns.heatmap(corr_t, vmin=-1, vmax= 1, center=0, cmap = sns.diverging_palette(20,220, n=200), square= True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment= 'right')
plt.title('Correlation Plot')
plt.tight_layout()
plt.show()

# 6.d: train/test split
X = df.values[:, 1:]
y = df.values[:, 0]

# raw data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# differenced data
y_train_diff, y_test_diff = train_test_split(DF, shuffle= False, test_size=0.2)

# 7 Stationary check
# ADF null - data is not stationary
# ADF alternative - data is stationary
toolbox.ADF_Cal(df.Close) # raw data - non-stationary, p-value 1.0 & test statistic > critical values
toolbox.ADF_Cal(DF.Close) # differenced - Data appeared to be stationary with p-value of 0.00000 & test statistic < critical values


# kpss null - data is stationary
# kpss alternative - data is non-stationary
toolbox.kpss_test(df.Close) # raw data - p-value < 0.05, reject null; data is non-stationary
toolbox.kpss_test(DF.Close) # differenced - p-value < 0.05, reject null; data is non-stationary

# 8: Time series decomposition

Close = df['Close']
Close = pd.Series(np.array(df['Close']),index = pd.date_range('1980-12-15 00:00:00', periods= len(Close)), name= 'Close Values (USD)')

res = STL(Close).fit()
fig = res.plot()
plt.ylabel('Residual')
plt.xlabel('Iterations')
plt.tight_layout()
plt.show()

T = res.trend
S = res.seasonal
R = res.resid

R = np.array(R)
S = np.array(S)
T = np.array(T)

# strength of seasonality and Trend
Ft = np.max([0,1 - np.var(R)/np.var(T+R)])
Fs = np.max([0,1 - np.var(R)/np.var(S+R)])

print('The strength of trend for this data set is', Ft.round(4))
# highly trended; 0.9998

print('The strength of seasonality for this data set is ', Fs.round(4))
# low seasonality; 0.333

plt.figure()
plt.plot(df.index,T, label= 'Trend')
plt.plot(df.index,R, label= 'Residual')
plt.plot(df.index,S, label= 'Seasonal')
plt.title('Trend, Residual, and Seasonal Plot')
plt.xticks(df.index[::10287], fontsize= 10)
plt.ylabel('Close Values (USD)')
plt.xlabel('Time')
plt.legend()
plt.tight_layout()
plt.show()

adjusted_seasonal = Close - S # Adjusted Seasonal Dataset
adjusted_seasonal.to_frame(name = 'Val')
detrended = Close - T # detrended data
detrended.to_frame(name = 'Val')


R = np.array(R)
A_S = np.array(adjusted_seasonal)
D_T = np.array(detrended)



plt.figure()
plt.plot(df.index,Close, label= 'Original Data', color = 'black')
plt.plot(df.index,adjusted_seasonal.values, label= 'Adjusted Seasonal', color = 'yellow')
plt.xticks(df.index[::10287], fontsize= 10)
plt.title('Seasonaly Adjusted Data vs. Differenced')
plt.xlabel('Date')
plt.ylabel('Close Values (USD)')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(df.index,Close, label= 'Original Data')
plt.plot(df.index,detrended.values, label= 'Detrended')
plt.xticks(df.index[::10287], fontsize= 10)
plt.title('Detrended Data vs. Original')
plt.xlabel('Date')
plt.ylabel('Close Values (USD)')
plt.legend()
plt.tight_layout()
plt.show()

# 9: Holt-Winter method

# use training data to fit model
model = ets.ExponentialSmoothing(y_train, damped_trend= True, trend='add').fit()

# prediction on train set
train_forecast = model.forecast(steps=len(y_train))
train_forecast = pd.DataFrame(train_forecast, columns=['Close'])
# made prediction on test set
test_forecast = model.forecast(steps=len(y_test))
test_forecast = pd.DataFrame(test_forecast, columns=['Close'])



# print the summary
print(model.summary())

# model assessment

# train data
train_forecast_error = np.array(y_train - train_forecast)
print("Mean square error for the Holt-Winter method prediction on Close Values (USD) is ", toolbox.mse(train_forecast_error).round(4))
print(sm.stats.acorr_ljungbox(train_forecast_error, lags=[5], boxpierce=True, return_df=True))
print('The Q value was found to be 23100.165439 with a p-value of 0.0')
print('the mean of the Holt-winter model prediction error is', np.mean(train_forecast_error))
print('the variance of the Holt-winter model prediction error is', np.var(train_forecast_error))
print('the RMSE of the Holt-winter model prediction error is, ', mean_squared_error(y_train, train_forecast, squared=False))

# test data
test_forecast_error = np.array(y_test - test_forecast)
print("Mean square error for the Holt-Winter method forecasting on Close Values (USD) is ", toolbox.mse(test_forecast_error).round(4))
print(sm.stats.acorr_ljungbox(test_forecast_error, lags=[5], boxpierce=True, return_df=True))
print('The Q value was found to be 5138.065419 with a p-value of 0.0')
print('the mean of the Holt-winter model error is', np.mean(test_forecast_error))
print('the variance of the Holt-winter model error is', np.var(test_forecast_error))
print('the RMSE of the Holt-winter model error is, ', mean_squared_error(y_test['Close'], test_forecast['Close'], squared=False))
print('the variance of the prediction error appeared larger than the variance of the testing error')


# plot Holt-Winter model

# plot of full model
plt.figure()
plt.xlabel('Time')
plt.ylabel('Close Values (USD)')
plt.title('Holt-Winter Method on Data')
plt.plot(y_train.index,y_train.Close,label= "Train Data", color = 'blue')
plt.plot(y_test.index,y_test.Close,label= "Test Data", color = 'red')
plt.plot(test_forecast, label = 'Forecasting Data', color = 'yellow')
plt.xticks(y.index[::10287], fontsize= 10)
plt.legend()
plt.tight_layout()
plt.show()


# plot of test data
plt.figure()
plt.plot(y_test.index,y_test.Close,label= "Test Data", color = 'red')
plt.plot(test_forecast, label = 'Forecasting Data', color = 'yellow')
plt.xlabel('Time')
plt.ylabel('Close Values (USD)')
plt.title(f'Holt-Winter Method on Data with MSE = {toolbox.mse(test_forecast_error).round(4)}')
plt.xticks(y_test.index[::(10287*.2)], fontsize= 10)
plt.legend()
plt.tight_layout()
plt.show()

# note
# mse 87444 # s 288

# holt-winter train data
plt.figure()
m_pred_f = 1.96/np.sqrt(len(y_train.Close))
toolbox.ACF_Plot(toolbox.autocorr(train_forecast_error,90),90)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('Holt-Winter Train Error ACF Plot with 90 Lags')
plt.axhspan(-m_pred_f,m_pred_f,alpha = .1, color = 'black')
plt.tight_layout()
plt.show()


# holt winter test data
plt.figure()
m_pred_f = 1.96/np.sqrt(len(y_test.Close))
toolbox.ACF_Plot(toolbox.autocorr(test_forecast_error,90),90)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('Holt-Winter Test Error ACF Plot with 90 Lags')
plt.axhspan(-m_pred_f,m_pred_f,alpha = .1, color = 'black')
plt.tight_layout()
plt.show()