import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
import TS_functions
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL

df = pd.read_csv("question2.csv")
print(df.head())
start_date = '1981-01-01'
date = pd.date_range(start_date, periods=len(df), freq='D')
print("========ROLLING MEAN & VARIANCE========")
plt.title("Rolling Mean from 1981")
plt.xlabel("Frequency")
plt.ylabel("Mean")
plt.plot(date, df.rolling(1).mean())
plt.grid()
plt.legend(['Mean'], loc='lower right')
plt.show()

plt.title("Rolling Variance from 1981")
plt.xlabel("Frequency")
plt.ylabel("Variance")
plt.plot(date, df.rolling(1).var())
plt.grid()
plt.legend(['Variance'], loc='lower right')
plt.show()

TS_functions.ADF_Cal(df.values)
TS_functions.kpss_test(df.values)

stl_data = df.copy()
stl_data.index = [i for i in range(stl_data.shape[0])]
stl_res = STL(stl_data, period=12).fit()
print("The strength of the seasonality is: ", str(TS_functions.seasonality_strength(df,stl_res.seasonal)))
print("The strength of the trend is: ", str(TS_functions.trend_strength(df,stl_res.trend)))
lags = 20
difference_data = TS_functions.series_differencing(df.values)
ry = sm.tsa.acf(difference_data, nlags=lags)
TS_functions.gpac_calc(ry, 7, 7)
na = 2
nb = 2
TS_functions.ACF_PACF_Plot(difference_data, 50, "ACF/PACF Difference Data")

# model = sm.tsa.ARMA(df.values, (na, nb)).fit(trend='nc', disp=0)
model = sm.tsa.ARIMA(endog=df.values, order=(na, 0, nb)).fit()
predictions = model.predict(start=0, end=len(df) - 1)
errors = df.values - predictions

for i in range(na):
    print("The AR Coefficient a{}".format(i), "is:", model.params[i])

for i in range(na):
    print("The MA Coefficient b{}".format(i), "is:", model.params[i + na])

intervals = model.conf_int()
for i in range(na):
    print("The Confidence Interval for a{}".format(i), "is:", intervals[i])
    print("The p-value for a{}".format(i), "is:", model.pvalues[i])
    print("The Standard Error for a{}".format(i), "is:", model.bse[i])
    print("\n")

for i in range(na):
    print("The Confidence Interval for b{}".format(i), "is:", intervals[i + na])
    print("The p-value for b{}".format(i), "is:", model.pvalues[i + na])
    print("The Standard Error for b{}".format(i), "is:", model.bse[i + na])
    print("\n")


print("The correlation coefficient between the data and prediction is: " + str(TS_functions.correlation_coefficient_cal(df.values, predictions)))

plt.title("Raw Data vs Forecast")
plt.xlabel("Frequency")
plt.ylabel("Raw Data")
plt.plot(df.values)
plt.plot(predictions)
plt.grid()
plt.legend(["Raw Data", "Predictions"], loc='lower right')
plt.show()

