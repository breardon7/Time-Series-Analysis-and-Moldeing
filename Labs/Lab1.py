# packages
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from toolbox import ADF_Cal, kpss_test



# load data
'''OR_PATH = os.getcwd()
os.chdir('..')
PATH = os.getcwd()
DATA_DIR = PATH + os.path.sep + "Datasets" + os.path.sep
os.chdir(OR_PATH)'''

df = pd.read_csv(r'C:\Users\brear\OneDrive\Documents\GitHub\Time-Series-Analysis-and-Moldeing\Datasets\tute1.csv')
df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
print(df.head())

# Question 1
'''fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle('Plots')
ax1.plot(df['Date'], df['Sales'])
ax2.plot(df['Date'], df['AdBudget'])
ax3.plot(df['Date'], df['GDP'])'''

plt.plot(df['Date'], df['Sales'])
plt.grid(color='gray', linestyle='-', linewidth=1)
plt.legend()
plt.title('Sales by Year')
plt.xlabel('Date')
plt.ylabel('USD')
plt.show()

plt.plot(df['Date'], df['AdBudget'])
plt.grid(color='gray', linestyle='-', linewidth=1)
plt.legend()
plt.title('AdBudget by Year')
plt.xlabel('Date')
plt.ylabel('USD')
plt.show()

plt.plot(df['Date'], df['GDP'])
plt.grid(color='gray', linestyle='-', linewidth=1)
plt.legend()
plt.title('GDP by Year')
plt.xlabel('Date')
plt.ylabel('USD')
plt.show()

print('---------Question 2---------')
sales_mean = df['Sales'].mean()
sales_var = df['Sales'].var()
sales_std = df['Sales'].std()
print('The Sales mean is : {} and the variance is : {} with standard deviation : {}'.format(sales_mean, sales_var, sales_std))

adb_mean = df['AdBudget'].mean()
adb_var = df['AdBudget'].var()
adb_std = df['AdBudget'].std()
print('The AdBudget mean is : {} and the variance is : {} with standard deviation : {}'.format(adb_mean, adb_var, adb_std))

# Question 3

#  Sales
sales_rolling_mean = []
for i in range(len(df['Sales'])):
    iter_mean = df['Sales'].loc[:i].mean()
    sales_rolling_mean.append(iter_mean)
# print(sales_rolling_mean)

sales_rolling_var = []
for i in range(len(df['Sales'])):
    iter_var = df['Sales'].loc[:i].var()
    sales_rolling_var.append(iter_var)
# print(sales_rolling_var)

fig_sales, (ax1, ax2) = plt.subplots(2)
fig_sales.suptitle('Sales')
ax1.plot(df['Date'], sales_rolling_mean)
ax1.set_title('Rolling Mean')
ax2.plot(df['Date'], sales_rolling_var)
ax2.set_title('Rolling Variance')
plt.tight_layout
plt.show()

#  ADBudget
adb_rolling_mean = []
for i in range(len(df['AdBudget'])):
    iter_mean = df['AdBudget'].loc[:i].mean()
    adb_rolling_mean.append(iter_mean)
# print(abd_rolling_mean)

adb_rolling_var = []
for i in range(len(df['AdBudget'])):
    iter_var = df['AdBudget'].loc[:i].var()
    adb_rolling_var.append(iter_var)
# print(abd_rolling_var)

fig_sales, (ax1, ax2) = plt.subplots(2)
fig_sales.suptitle('AdBudget')
ax1.plot(df['Date'], adb_rolling_mean)
ax1.set_title('Rolling Mean')
ax2.plot(df['Date'], adb_rolling_var)
ax2.set_title('Rolling Variance')
plt.tight_layout
plt.show()

#  GDP
gdp_rolling_mean = []
for i in range(len(df['GDP'])):
    iter_mean = df['GDP'].loc[:i].mean()
    gdp_rolling_mean.append(iter_mean)
# print(gdp_rolling_mean)

gdp_rolling_var = []
for i in range(len(df['GDP'])):
    iter_var = df['GDP'].loc[:i].var()
    gdp_rolling_var.append(iter_var)
# print(gdp_rolling_var)

fig_sales, (ax1, ax2) = plt.subplots(2)
fig_sales.suptitle('GDP')
ax1.plot(df['Date'], gdp_rolling_mean)
ax1.set_title('Rolling Mean')
ax2.plot(df['Date'], gdp_rolling_var)
ax2.set_title('Rolling Variance')
plt.tight_layout
plt.show()

# Question 4
print('---------Question 4---------')
'''
Sales:
The rolling mean begins to stabilize, but still shows
some variance as samples are included. I would consider it
to be stationary as samples are added. The rolling variance seems
to be unstable all the way through each additional sample,
rendering it non-stationary.

ADBudget:
Both rolling mean and rolling variance seem to stabilize
early on in the addition of samples and seem to both become stationary
as there is little variance in the rolling values from then on.

GDP:
The rolling mean and variance both become stable with little variance
about halfway through the addition of samples, but are not stable
preceding the halfway point and have high variance with the addition of
each sample. This shows being non-stationary with the first half of the
samples, but stationary with the addition of the latter half of samples.

'''

# Question 5
print('---------Question 5---------')
print('--Sales ADF--')
ADF_Cal(df['Sales'])

print('--AdBudget ADF--')
ADF_Cal(df['AdBudget'])

print('--GDP ADF--')
ADF_Cal(df['GDP'])

'''
Sales: p-value <5%; reject null hypothesis. Assume no root (data is not stationary)
AdBudget: p-value >5%; fail to reject null hypothesis. Assume root (data is stationary)
GDP: p-value <5%; reject null hypothesis. Assume no root (data is not stationary)

The ADF test reinforces my observations in question 4.
'''

# Question 6
print('---------Question 6---------')
print('--Sales kpss--')
kpss_test(df['Sales'])

print('--AdBudget kpss--')
kpss_test(df['AdBudget'])

print('--GDP kpss--')
kpss_test(df['GDP'])

'''
The test statistics for Sales, AdBudget, and GDP are all lower than the critical value given
a confidence interval of 95% (critical value 5%), which aligns with the p-value for each 
subset of data being >0.05. This means we fail to reject the null hypothesis for each
subset of data, making the assumption that all three subsets of data are stationary.

The results of the kpss do not reinforce the observations of the previous steps for 
the Sale and GDP subsets, but does for the AdBudget subset.
'''