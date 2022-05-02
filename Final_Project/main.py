import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

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
def difference(dataset, interval):
   diff = []
   for i in range(interval, len(dataset)):
      value = dataset[i] - dataset[i - interval]
      diff.append(value)
   return diff


# applied 1 differencing as
# there was 1 interval per day
Diff_1 = difference(df.Close, 1)
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
def rolling_mv(y, title):
    means = []
    vars = []

    for i in range(1, len(y)):
        means.append(y[:i].mean())
        vars.append(y[:i].var())

    # plot rolling mean
    plt.figure()
    plt.subplot(211)
    plt.plot(means, label= 'Mean')
    plt.title("Rolling Mean "  + str(title))
    plt.xlabel('Time')
    plt.ylabel("Rolling Mean")
    plt.subplot(212)
    plt.plot(vars, label= 'Variance')
    plt.title("Rolling Variance " + str(title))
    plt.xlabel('Time')
    plt.ylabel("Rolling Variance")
    plt.tight_layout()
    plt.show()

# plot rolling average
rolling_mv(df.Close, 'of Original Data')
rolling_mv(DF.Close, 'of Differenced Data')

# 6.b: ACF/PACF plots ACF Plot----------------------------------------------------
def autocorr(x,lag):
     l = range(lag+1)
     x_br = np.mean(x)
     autocorr = []
     for i in l:
         num = 0
         var = 0
         for j in range(i, len(x)):
             num += np.sum(x[j]- x_br) *(x[j-i] -x_br)
         var = np.sum((x - x_br) ** 2)
         autocorr.append(num/var)
     return autocorr

def ACF_Plot(x,lag):
    lg = np.arange(-lag, lag + 1)
    x = x[0:lag + 1]
    rx = x[::-1]
    rxx = rx[:-1] + x
    plt.stem(lg, rxx)



# ACF plot of raw data
plt.figure()
bound = 1.96/np.sqrt(len(df.Close))
ACF_Plot(autocorr(df.Close,90),90)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Close Values (USD) 90 Lags')
plt.axhspan(-bound,bound,alpha = .1, color = 'black')
plt.tight_layout()
plt.show()

# ACF plot of differenced data
plt.figure()
bound = 1.96/np.sqrt(len(DF.Close))
ACF_Plot(autocorr(DF.Close,90),90)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Differenced Close Values (USD) with 90 Lags')
plt.axhspan(-bound,bound,alpha = .1, color = 'black')
plt.tight_layout()
plt.show()


def PACF_ACF_Plot(x,lags,title):
    plt.figure()
    plt.subplot(211)
    plt.xlabel('Lags')
    plt.ylabel('ACF Value')
    plt.title('ACF and PACF plot ' + str(title))
    sm.graphics.tsa.plot_acf(x, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plt.xlabel('Lags')
    plt.ylabel('PACF Value')
    sm.graphics.tsa.plot_pacf(x, ax=plt.gca(), lags=lags)
    plt.tight_layout()
    plt.show()

PACF_ACF_Plot(df.Close,500, 'of Raw Data')
PACF_ACF_Plot(DF.Close,500, 'of Differenced Data')

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)