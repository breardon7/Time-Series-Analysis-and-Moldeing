import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from toolbox import ACF, cal_rolling_mean_var
from scipy import signal


# question 1
sample_count = 1000
e = np.random.normal(0, 1, sample_count)
y = np.zeros(len(e))
step = 2
for t in range(sample_count):
    if t >= step:
        y[t] = 0.5*y[t-1] + 0.2*y[t-2] + e[t]

y = np.array(y)

plt.plot(np.arange(sample_count), y)
plt.title('AR 2-Step')
plt.ylabel('Value')
plt.xlabel('Sample')
plt.show()

ACF(y, 20, 'AR Data')
plt.show()

cal_rolling_mean_var(y, np.arange(sample_count), metric='AR', unit='Sample')
plt.show()

print('First five values of y(t): ', y[:5])
print('The rolling variance and mean of the sample data\nshows that the data is stationary since it becomes constant as samples are added. \nAdditionally, the ACF plot indicates that there is no relationship \nbetween present and past values, which allows us to assume the data is stationary.')

# question 2
tf = ([1.0,], [1.0, -1.0], 1.0)
t_in = [0.0, 1.0, 2.0, 3.0]
u = e
t_out, y = signal.dlsim(tf, u, t=t_in)
print(y.T)

# question 3
X_matrix = X_train.values
print(X_matrix[0])
y_matrix = y_train.values
H = np.matmul(X_matrix.T, X_matrix)

# question 4
# estimate the regression model using LSE method
estimate_model = np.matmul(np.linalg.inv(np.matmul(X_matrix.T, X_matrix)), np.matmul(X_matrix.T, y_matrix))