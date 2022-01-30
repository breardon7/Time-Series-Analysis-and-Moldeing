import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Question 1

def correlation_coefficent_cal(x,y):
    numer = 0
    denom_x = 0
    denom_y = 0
    for i in range(len(x)):
        numer += ((x[i] - np.mean(x)) * (y[i] - np.mean(y)))
        denom_x += (x[i] - np.mean(x))**2
        denom_y += (y[i] - np.mean(y))**2
    r = numer / (np.sqrt(denom_x) * np.sqrt(denom_y))
    print(r)

# Question 2
x = [1,2,3,4,5]
y = [1,2,3,4,5]
z = [-1,-2,-3,-4,-5]
g = [1,1,0,-1,-1,0,1]
h = [0,1,1,1,-1,-1,-1]

correlation_coefficent_cal(x, y)
correlation_coefficent_cal(x, z)
correlation_coefficent_cal(g, h)

# Question 3
df = pd.read_csv(r'C:\Users\brear\OneDrive\Documents\GitHub\Time-Series-Analysis-and-Moldeing\Datasets\tute1.csv')
df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
print(df.head())

plt.scatter(df['Sales'], df['GDP'], c = 'g')
plt.legend(loc='upper left')
plt.xlabel('GDP')
plt.ylabel('Sales')
plt.title('Sales by GDP')
plt.grid(color='gray', linestyle='-', linewidth=1)
plt.show()

correlation_coefficent_cal(df['Sales'], df['GDP'])
'''
The calculated correlation coefficient makes sense with respect
to the scatter plot since the scatter trends negatively and the 
correlation coefficient is -0.63.
'''

# question 4
plt.scatter(df['Sales'], df['AdBudget'], c = 'r')
plt.legend(loc='upper left')
plt.xlabel('AdBudget')
plt.ylabel('Sales')
plt.title('Sales by AdBudget')
plt.grid(color='gray', linestyle='-', linewidth=1)
plt.show()

correlation_coefficent_cal(df['Sales'], df['AdBudget'])
'''
The calculated correlation coefficient makes sense with respect
to the scatter plot since the scatter trends positively and the 
correlation coefficient is 0.90.
'''

# question 5
plt.scatter(df['GDP'], df['AdBudget'], c = 'b')
plt.legend(loc='upper left')
plt.xlabel('AdBudget')
plt.ylabel('GDP')
plt.title('GDP by AdBudget')
plt.grid(color='gray', linestyle='-', linewidth=1)
plt.show()

correlation_coefficent_cal(df['AdBudget'], df['GDP'])
'''
The calculated correlation coefficient makes sense with respect
to the scatter plot since the scatter trends negatively and the 
correlation coefficient is -0.76.
'''

# question 6
sns.pairplot(df, kind="kde", diag_kind = 'hist')
plt.show()
sns.pairplot(df, kind="hist", diag_kind = 'hist')
plt.show()

# question 7
sns.heatmap(df[['Sales','AdBudget','GDP']].corr(), annot=True)
plt.show()