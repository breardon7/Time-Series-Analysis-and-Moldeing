import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

df = pd.read_csv(r'C:\Users\brear\OneDrive\Desktop\Grad School\Time-Series-Analysis-and-Moldeing\Datasets\auto.clean.csv')
#print(df.columns)
#print(df.head())
#print(df.shape)

y = df['price'].copy()
X = df.drop(columns='price').copy()
X['ones'] = np.ones(201)
first_column = X.pop('ones')
X.insert(0, 'ones', first_column)

# question 1
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

# question 2
sns.heatmap(df.corr())
plt.show()
plt.tight_layout()


# question 3
X_matrix = X.values
print(X_matrix)
u, s, v = np.linalg.svd(X_matrix, full_matrices=True)

var_explained = np.round(s ** 2 / np.sum(s ** 2), decimals=3)

sns.barplot(x=list(range(1, len(var_explained) + 1)),
            y=var_explained, color="limegreen")
plt.xlabel('SVs', fontsize=16)
plt.ylabel('Percent Variance Explained', fontsize=16)
plt.show()

'''
print(X.dtypes)
X_new = X.select_dtypes(exclude=['object'])
print(X_new)
print(X_new.shape)
svd = TruncatedSVD(n_components=X_new.shape[1])
X_reduced = svd.fit_transform(X_new)
print(X_reduced)
print(X_reduced.shape)
'''

# question 4
