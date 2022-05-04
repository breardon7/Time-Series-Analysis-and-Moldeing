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


# 6.a: plot the Close vs time----------------------------------------------------

y = df['Close']
X = df.drop(columns=['Close'])

# applied second order differencing
Diff_1 = toolbox.difference(y, 1)
Diff_2 = toolbox.difference(y, 1)
# Diff_1 = np.log(df.Close)
DF = pd.DataFrame(Diff_1, index=df.index[1:])
DF.rename(columns={0:'Close'}, inplace=True)
print(DF)


plt.figure()
plt.subplot(211)
plt.plot(y)
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.ylabel('Close Price (USD')
plt.title('Close Price Over Time')
plt.subplot(212)
plt.plot(DF)
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.ylabel('Close Price (USD)')
plt.title('Differenced Close Price Over Time')
plt.tight_layout()
plt.show()

# =============== #
# Rolling Average #
# =============== #

# plot rolling average
toolbox.rolling_mv(y, 'of Original Data')
toolbox.rolling_mv(DF, 'of Differenced Data')

# 6.b: ACF/PACF plots ACF Plot----------------------------------------------------

# ACF plot of raw data
plt.figure()
bound = 1.96/np.sqrt(len(y))
toolbox.ACF_Plot(toolbox.autocorr(y,90),90)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Close Values (USD) 90 Lags')
plt.axhspan(-bound,bound,alpha = .1, color = 'black')
plt.tight_layout()
plt.show()

# ACF plot of differenced data
plt.figure()
bound = 1.96/np.sqrt(len(DF))
toolbox.ACF_Plot(toolbox.autocorr(DF.Close,90),90)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Differenced Close Values (USD) with 90 Lags')
plt.axhspan(-bound,bound,alpha = .1, color = 'black')
plt.tight_layout()
plt.show()

toolbox.PACF_ACF_Plot(y,500, 'of Raw Data')
toolbox.PACF_ACF_Plot(DF,500, 'of Differenced Data')

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

# raw data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle= False)
# differenced data
y_train_diff, y_test_diff = train_test_split(DF, shuffle= False, test_size=0.2)

# 7 Stationary check
# ADF null - data is not stationary
# ADF alternative - data is stationary
toolbox.ADF_Cal(y) # raw data - non-stationary, p-value 1.0 & test statistic > critical values
toolbox.ADF_Cal(DF) # differenced - Data appeared to be stationary with p-value of 0.00000 & test statistic < critical values


# kpss null - data is stationary
# kpss alternative - data is non-stationary
toolbox.kpss_test(y) # raw data - p-value < 0.05, reject null; data is non-stationary
toolbox.kpss_test(DF) # differenced - p-value < 0.05, reject null; data is non-stationary

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
plt.xticks(rotation=45)
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
plt.xticks(rotation=45)
plt.title('Seasonaly Adjusted Data vs. Differenced')
plt.xlabel('Date')
plt.ylabel('Close Values (USD)')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(df.index,Close, label= 'Original Data')
plt.plot(df.index,detrended.values, label= 'Detrended')
plt.xticks(rotation=45)
plt.title('Detrended Data vs. Original')
plt.xlabel('Date')
plt.ylabel('Close Values (USD)')
plt.legend()
plt.tight_layout()
plt.show()

# 9: Holt-Winter method

holtmodel = ets.ExponentialSmoothing(y_train, trend='add', seasonal='add', damped_trend =True, seasonal_periods=56).fit()
holt_ws = holtmodel.forecast(steps=len(y_test))
holt_ws = pd.DataFrame(holt_ws).set_index(y_test.index)
fig, ax = plt.subplots()
ax.plot(y_train, label='Train Data')
ax.plot(y_test, label='Test Data')
ax.plot(holt_ws, label='Holt Winter Linear Forecast')
plt.legend()
plt.xlabel('Time (Year-Month)')
plt.ylabel('Close Values (USD)')
plt.title('Additive Holt-Winters Linear Model')
plt.show()
holtmse = mean_squared_error(y_test, holt_ws)
print(f'The mean squared error of the Holt-Winters forecast is: {holtmse}')

# 10. Feature reduction

X_fr = sm.add_constant(X_train)
model = sm.OLS(y_train, X_fr)
results = model.fit()
print(results.params)
print("All:\n",results.summary())

# p-value is 0 for all features, so no feature reduction to do.

# 11. Base Models

avg = toolbox.average_forecast(y_train, y_test, type='test')
holt_ws = pd.DataFrame(avg).set_index(y_test.index)
fig, ax = plt.subplots()
ax.plot(y_train, label = 'Train Data')
ax.plot(y_test, label = 'Test Data')
ax.plot(avg, label = 'Average Forecast')
plt.title('Average Forecast Method')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Close')
plt.show()

naive = toolbox.naive(y_test)
naive = pd.DataFrame(naive).set_index(y_test.index)
fig, ax = plt.subplots()
ax.plot(y_train, label = 'Train Data')
ax.plot(y_test, label = 'Test Data')
ax.plot(naive, label = 'Naive Forecast')
plt.title('Naive Forecast Method')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Close')
plt.show()

drift = toolbox.drift(y_train, y_test, type='test')
drift = pd.DataFrame(drift).set_index(y_test.index)
fig, ax = plt.subplots()
ax.plot(y_train, label = 'Train Data')
ax.plot(y_test, label = 'Test Data')
ax.plot(drift, label = 'Drift Forecast')
plt.title('Drift Forecast Method')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Close')
plt.show()

ses = toolbox.ses(y_train, y_test, type='test', alpha=0.5)
ses = pd.DataFrame(ses).set_index(y_test.index)
fig, ax = plt.subplots()
ax.plot(y_train, label = 'Train Data')
ax.plot(y_test, label = 'Test Data')
ax.plot(ses, label = 'SES Forecast')
plt.title('SES Forecast Method')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Close')
plt.show()

# 12. Multiple Linear Regression
X_test_reg = sm.add_constant(X_test)
y_pred = model.predict(X_test_reg)
fig, ax = plt.subplots()
ax.plot(y_train, label='train')
ax.plot(y_test, label='test')
ax.plot(y_pred, label='pred', ls='dashed', color='r')
plt.legend()
plt.ylabel('Arrests')
plt.xlabel('Time')
plt.title('Regression Forecast')
plt.show()
A = np.identity(len(model.params))
A = A[1:,:]
print(f'F test: {model.f_test(A)}')
yerror_for = y_test - y_pred
toolbox.PACF_ACF_Plot(yerror_for, 100, 'MLP PACF ACF')
var_forecast = toolbox.est_var(yerror_for,X_test)
print(f'variance of error: {var_forecast}\nmean of error:{sum(yerror_for)/len(yerror_for)}')

