import pandas as pd
import matplotlib.pyplot as plt
from toolbox import plot_rolling_mean_var, ADF_Cal, kpss_test, difference

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
The trend is seasonal as similar change
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
p-value >0,05; fail to reject null hypothesis. Assume root (data is non-stationary)
"""

# question 5
kpss_test(df['#Passengers'])
'''
The test statistic for Passengers is higher than the critical value given
a confidence interval of 95% (critical value 5%), which aligns with the p-value 
being <0.05. This means we reject the null hypothesis, making the assumption 
that the dataset is non-stationary.
'''

# Question 6
print('---question 6---')
diff_1 = difference(df['#Passengers'])
plot_rolling_mean_var(diff_1, '1st Order Difference')
ADF_Cal(diff_1)

'''
ADF-test: p-value >0.05, reject null hypothesis; assume non-stationary.
Visual test looks to be non-stationary given the change in the metrics as samples are added.
1st order dataset is non-stationary.
'''

# Question 7
print('---question 7---')
diff_2 = difference(diff_1)
plot_rolling_mean_var(diff_2, '2nd Order Difference')
ADF_Cal(diff_2)

'''
ADF-test: p-value <0.05, fail to reject null hypothesis; assume stationary.
The plots seem to present non-stationary, but this test is subjective.
2nd order dataset is stationary per the ADF-test results.
'''

# Question 8
print('---question 8---')
diff_3 = difference(diff_2)
plot_rolling_mean_var(diff_3, '3rd Order Difference')
ADF_Cal(diff_3)

'''
ADF-test: p-value <0.05, fail to reject null hypothesis; assume stationary.
The plots seem to present non-stationary, but this test is subjective.
3rd order dataset is stationary per the ADF-test results.
'''

# Question 9
'''
The dataset is stationary after the second order difference, so 
a log transformation is unnecessary.
'''