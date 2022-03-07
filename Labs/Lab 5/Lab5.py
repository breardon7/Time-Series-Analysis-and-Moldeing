
import pandas as pd
import numpy as np
from numpy import linalg as la
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from scipy.stats import ttest_ind
warnings.filterwarnings('ignore')

df = pd.read_csv(r'C:\Users\brear\OneDrive\Desktop\Grad School\Time-Series-Analysis-and-Moldeing\Datasets\auto.clean.csv')
#print(df.columns)
#print(df.head())
#print(df.shape)

y = df['price'].copy()
X = df[['normalized-losses', 'wheel-base', 'length', 'width',
'height', 'curb-weight', 'engine-size', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg']]
X = sm.add_constant(X, prepend=True)


# question 1
# train test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

# question 2
# correlation heatmap
corr = df.corr()
fig = plt.figure(figsize=(11, 10))
ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.title('Correlation of Car Variables', fontsize=16)
plt.show()

# question 3
# SVD analysis
X_matrix = X_train.values
print(X_matrix[0])
y_matrix = y_train.values
H = np.matmul(X_matrix.T, X_matrix)

u, s, v = np.linalg.svd(H) #, full_matrices=True)
print('Singular values of original = ', s)
print('Co-linearity exists in this dataset and is indicated by the small eigenvalues '
      'in the singular values array.')
print('\nThe condition number of original = {}'.format(la.cond(X)),
      '\nThe conditional number being {} indicates that the matrix'
      'is ill-conditioned and highly sensitive to small changes,'
      'and that co-linearity exists.'.format(la.cond(X)))
print('Two features will be removed to avoid the co-linearity.')

# question 4
# estimate the regression model using LSE method
estimate_model = np.matmul(np.linalg.inv(np.matmul(X_matrix.T, X_matrix)), np.matmul(X_matrix.T, y_matrix))
print('Estimate Regression Model = ', estimate_model)
print()

# question 5
#Use OLS function to find the unknown coefficients
model = sm.OLS(y_train, X_train).fit()
print('Model Summary of Original Training Data with All Features: \n')
print(model.summary())
print('The unknown coefficients from step 4 and 5 are identical.')

# question 6
#Use backward stepwise regression to reduce the feature space dimension

#----------------------
#Removing 'bore' feature
#----------------------
X_train.drop(['bore'], axis=1, inplace=True)
model = sm.OLS(y_train, X_train).fit()
print('\nSummary of Training Data After Removing "bore" Feature:\n')
print(model.summary())

#----------------------
#Removing 'normalized-losses' feature
#----------------------
X_train.drop(['normalized-losses'], axis=1, inplace=True)
model = sm.OLS(y_train, X_train).fit()
print('\nSummary of Training Data After Removing "normalized-losses" Feature:\n')
print(model.summary())

#----------------------
#Removing 'curb-weight' feature
#----------------------
X_train.drop(['curb-weight'], axis=1, inplace=True)
model = sm.OLS(y_train, X_train).fit()
print('\nSummary of Training Data After Removing "curb-weight" Feature:\n')
print(model.summary())

#----------------------
#Removing 'length' feature
#----------------------
X_train.drop(['length'], axis=1, inplace=True)
model = sm.OLS(y_train, X_train).fit()
print('\nSummary of Training Data After Removing "length" Feature:\n')
print(model.summary())

#----------------------
#Removing 'height' feature
#----------------------
X_train.drop(['height'], axis=1, inplace=True)
model = sm.OLS(y_train, X_train).fit()
print('\nSummary of Training Data After Removing "height" Feature:\n')
print(model.summary())

#----------------------
#Removing 'highway-mpg' feature
#----------------------
X_train.drop(['highway-mpg'], axis=1, inplace=True)
model = sm.OLS(y_train, X_train).fit()
print('\nSummary of Training Data After Removing "highway-mpg" Feature:\n')
print(model.summary())

#----------------------
#Removing 'city-mpg' feature
#----------------------
X_train.drop(['city-mpg'], axis=1, inplace=True)
model = sm.OLS(y_train, X_train).fit()
print('\nSummary of Training Data After Removing "city-mpg" Feature:\n')
print(model.summary())

#----------------------
#Removing 'width' feature
#----------------------
X_train.drop(['width'], axis=1, inplace=True)
model = sm.OLS(y_train, X_train).fit()
print('\nSummary of Training Data After Removing "width" Feature:\n')
print(model.summary())

#----------------------
#Removing 'const' feature
#----------------------
X_train.drop(['const'], axis=1, inplace=True)
model = sm.OLS(y_train, X_train).fit()
print('\nSummary of Training Data After Removing "const" Feature:\n')
print(model.summary())

#----------------------
#Removing 'wheel-base' feature
#----------------------
X_train.drop(['wheel-base'], axis=1, inplace=True)
model = sm.OLS(y_train, X_train).fit()
print('\nSummary of Training Data After Removing "wheel-base" Feature:\n')
print(model.summary())

#----------------------
#Removing 'peak-rpm' feature
#----------------------
X_train.drop(['peak-rpm'], axis=1, inplace=True)
model = sm.OLS(y_train, X_train).fit()
print('\nSummary of Training Data After Removing "peak-rpm" Feature:\n')
print(model.summary())

print('The features recommended for keeping are engine-size, stroke, compression-ratio, and horsepower.'
      'The rest are recommended to be eliminated.')

# question 7
#Use OLS function on the reduced feature space
# model output from above

# question 8
#drop the columns in X_test that were dropped in X_train
X_test.drop(['const', 'normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'bore',
             'peak-rpm', 'city-mpg', 'highway-mpg'], axis=1, inplace=True)

#prediction values for the prediction (train) set and forecast (test) set
predictions_pred = model.predict(X_train)
predictions_fore = model.predict(X_test)

#plot the training set, testing set, and forecasts of regression model
plt.figure()
plt.plot(y_train, label='Training Set')
plt.plot(y_test, label='Testing Set')
plt.plot(predictions_fore, label='Forecasts')
plt.xlabel('Car ID')
plt.ylabel('Sales Price ($USD)')
plt.title('Prediction of Cars Sales Price Using Multiple Linear Regression')
plt.grid()
plt.legend()
plt.show()

# question 9 & 10

#calculate the predictions error
df_pred = pd.DataFrame([y_train, predictions_pred]).transpose()
df_pred.columns = ['Y_train', 'Predictions']
df_pred['pred_error'] = df_pred['Y_train'] - df_pred['Predictions']

#calculate the forecast error
df_fore = pd.DataFrame([y_test, predictions_fore]).transpose()
df_fore.columns = ['Y_test', 'Forecast']
df_fore['forecast_error'] = df_fore['Y_test'] - df_fore['Forecast']

def ACF(timeseries_data, lags, metric=''):
    auto_corr = []
    timeseries_data_mean = np.mean(timeseries_data)
    length = len(timeseries_data)
    denominator = 0     # 0th lag adjusted
    x_axis = np.arange(0, lags+1)
    m = 1.96/np.sqrt(length)

    for denom_t in range(0, length):
        denominator = denominator + (timeseries_data[denom_t] - timeseries_data_mean) ** 2

    for tau in range(0, lags+1):
        numerator = 0
        for num_t in range(tau, length):
            numerator = numerator + (timeseries_data[num_t] - timeseries_data_mean) * (
                        timeseries_data[num_t - tau] - timeseries_data_mean)
        auto_corr.append((numerator / denominator))

    plt.stem(x_axis, auto_corr, use_line_collection=True)
    plt.stem(-1 * x_axis, auto_corr, use_line_collection=True)
    plt.title(f"ACF Plot - {metric}")
    plt.xlabel("Lags")
    plt.ylabel("ACF")
    plt.axhspan(-m, m, alpha=0.2, color='blue')
    # use plt.show() in main for graphs to show
    return auto_corr

#Find ACF of prediction error and forecast error
ACF(df_pred['pred_error'].to_numpy(), 20, 'ACF')
plt.show()
ACF(df_fore['forecast_error'].to_numpy(), 20, 'ACF')
plt.show()

# question 11

#calculate estimated variance for prediction errors
sse_pred = 0
for i in range(len(df_pred)):
    sse_pred += (df_pred.iloc[i, 2]) ** 2

variance_pred = np.sqrt(sse_pred / (len(df_pred) - 4 - 1))
print('\nThe estimated variance of prediction error is: %0.6f' % variance_pred)

#calculate estimated variance for forecast errors
sse_fore = 0
for i in range(len(df_fore)):
    sse_fore += (df_fore.iloc[i, 2]) ** 2

variance_fore = np.sqrt(sse_fore / (len(df_fore) - 1 - 1))
print('\nThe estimated variance of forecast error is: %0.6f' % variance_fore)

# results explained
print('\nThe estimated variance of the prediction errors ({}) is larger \
than that of the estimated\nvariance of the forecast errors ({}). This \
means the forecast is more accurate than the prediction.'.format(variance_pred, variance_fore))


