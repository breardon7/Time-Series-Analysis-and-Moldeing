
import pandas as pd
import numpy as np
from numpy import linalg as la
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
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
