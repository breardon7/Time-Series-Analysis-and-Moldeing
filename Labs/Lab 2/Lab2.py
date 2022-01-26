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

