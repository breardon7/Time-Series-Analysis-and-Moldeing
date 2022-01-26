import pandas as pd
import matplotlib.pyplot as plt
from toolbox import plot_rolling_mean_var, ADF_Cal, kpss_test

# data read in
df = pd.read_csv(r'C:\Users\brear\OneDrive\Documents\GitHub\Time-Series-Analysis-and-Moldeing\Datasets\AirPassengers.csv')

# question 1
print(df.head())

# question 2
plt.plot(df['Month'], df['#Passengers'], label='Sales')
plt.xlabel('Month')
plt.ylabel('Sales Number')
plt.title('Air passengers Dataset without differencing')
plt.legend(loc='upper left')
plt.show()

'''
The trent is seasonal as similar change
in trend occurs at an interval of a set
period of time.
'''

# question 3
plot_rolling_mean_var(df['#Passengers'], 'Rolling Mean & Variance: Passengers')

'''
Based on the rolling mean and variance, the data looks to be
non-stationary.
'''

# question 4
ADF_Cal(df['#Passengers'])
"""
p-value >0,05; fail to reject null hypothesis. Assume root (data is stationary)
"""

# question 5
kpss_test(df['#Passengers'])
'''
The test statistic for Passengers is higher than the critical value given
a confidence interval of 95% (critical value 5%), which aligns with the p-value 
being <0.05. This means we reject the null hypothesis, making the assumption 
that the dataset is non-stationary.
'''
