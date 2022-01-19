from statsmodels.tsa.stattools import adfuller

def rolling_mean(data):
    rolling_mean = []
    for i in range(len(data)):
        iter_mean = data.loc[:i].mean()
        rolling_mean.append(iter_mean)
    # print(gdp_rolling_mean)
        return rolling_mean

def rolling_var(data):
    rolling_var = []
    for i in range(len(data)):
        iter_var = data.loc[:i].var()
        rolling_var.append(iter_var)
    # print(gdp_rolling_var)
        return rolling_mean

def ADF_Cal(x):
 result = adfuller(x)
 print("ADF Statistic: %f" %result[0])
 print('p-value: %f' % result[1])
 print('Critical Values:')
 for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


