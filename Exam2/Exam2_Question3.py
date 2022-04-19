import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
import Helper

df = pd.read_csv("question3.csv")
print(df.head())
print("========ROLLING MEAN & VARIANCE========")
plt.title("Rolling Mean")
plt.xlabel("Frequency")
plt.ylabel("Mean")
plt.plot(df.rolling(1).mean())
plt.grid()
plt.legend(['Mean'], loc='lower right')
plt.show()

plt.title("Rolling Variance")
plt.xlabel("Frequency")
plt.ylabel("Variance")
plt.plot(df.rolling(1).var())
plt.grid()
plt.legend(['Variance'], loc='lower right')
plt.show()

Helper.ADF_Cal(df.values)
Helper.kpss_test(df.values)

first_order_differencing = df - df.shift(1)

df = pd.read_csv("question3.csv")
print(df.head())
print("========ROLLING MEAN & VARIANCE First Order Difference========")
plt.title("Rolling Mean")
plt.xlabel("Frequency")
plt.ylabel("Mean")
plt.plot(first_order_differencing.rolling(1).mean())
plt.grid()
plt.legend(['Mean'], loc='lower right')
plt.show()

plt.title("Rolling Variance")
plt.xlabel("Frequency")
plt.ylabel("Variance")
plt.plot(first_order_differencing.rolling(1).var())
plt.grid()
plt.legend(['Variance'], loc='lower right')
plt.show()
Helper.ADF_Cal(first_order_differencing[1:])
Helper.kpss_test(first_order_differencing[1:])

lags = 20
ry = sm.tsa.acf(first_order_differencing, nlags=lags)
Helper.gpac_calc(ry, 7, 7)

Helper.ACF_PACF_Plot(first_order_differencing, 50, "Seasonal Difference ACF & PACF")

plt.title("Raw Data vs Seasonal Difference Data")
plt.xlabel("Frequency")
plt.ylabel("Raw Data")
plt.plot(df.values)
plt.plot(first_order_differencing)
plt.grid()
plt.legend(["Raw Data", "Season Differenced"], loc='lower right')
plt.show()

