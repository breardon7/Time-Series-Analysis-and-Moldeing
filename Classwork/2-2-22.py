import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.holtwinters as ets
from sklearn.model_selection import train_test_split
import statsmodels.api as sa

df = pd.read_csv(r'C:\Users\brear\OneDrive\Documents\GitHub\Time-Series-Analysis-and-Moldeing\Datasets\AirPassengers.csv')

y = df['#Passengers']
yt, yf = train_test_split(y,shuffle=False, test_size=0.2)

holtt = ets.ExponentialSmoothing(yt, trend=None, damped_trend=False, seasonal=None).fit()
holtf = holtt.forecast(steps=len(yf))
holtf = pd.DataFrame(holtf).set_index(yf.index)

fig, ax = plt.subplots()
ax.plot(yt, label = 'Train')
ax.plot(yf, label = 'Test')
ax.plot(holtf, label='SES')
plt.show()