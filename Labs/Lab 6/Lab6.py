import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from toolbox import ACF, cal_rolling_mean_var

# question 1
sample_count  = 1000
y = np.random.normal(0, .1, sample_count)
e_t = []
step = 2
for t in range(sample_count):
    if t >= step:
        e_t.append(y[t] - 0.5*y[t-1] - 0.2*y[t-2])

e_t = np.array(e_t)

plt.plot(np.arange(2, 1000), e_t)
plt.title('AR 2-Step')
plt.ylabel('Value')
plt.xlabel('Sample')
plt.show()

ACF(e_t, 20, 'AR Data')
plt.show()

cal_rolling_mean_var(e_t, np.arange(2, 1000), metric='AR', unit='Sample')
plt.show()

print('First five values of y(t): ', y[:5])
print('The rolling variance and mean of the sample data shows that the data is stationary as it levels off. \nAdditionally, the ACF plot indicates that there is no relationship between present and past values, which allows us to assume the data is stationary.')