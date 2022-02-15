import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statistics import variance

# question 2
t_train = [112, 118, 132, 129, 121, 134, 148, 163, 119]
t_test = [104, 118, 115, 126, 141]
combined = t_train + t_test
print(combined)
def average_forecast(original):
    prediction = []
    for i in range(len(original)):
        prediction.append(round((sum(original[:i+1])/(i+1))))
    return prediction

print(average_forecast(t_train))
print(average_forecast(t_test))
print(average_forecast(combined))

plt.plot(np.arange(1,10), t_train, c='b') #legend='train'
plt.plot(np.arange(9,14), t_test, c='r') #legend='test'
plt.plot(np.arange(9,14),average_forecast(combined)[-5:], c='g') #legend='forecast')
plt.xlabel('Time (t)')
plt.ylabel('Series Values')
plt.title('Average Forecast')
plt.show()

# question 3

def error(original, prediction):
    error = []
    for i in range(len(original)):
        error.append(original[i] - prediction[i])
    return error

def error_squared(error):
    error_squared = []
    for i in range(len(error)):
        error_squared.append(error[i] ** 2)
    return error_squared

def MSE(original, prediction):
    error_squared = 0
    for i in range(len(original)):
        error = original[i] - prediction[i]
        error_squared += error ** 2
    mse = error_squared/len(original)
    return mse

print('train MSE:', MSE(t_train, average_forecast(t_train)))
print('test MSE:', MSE(t_test, average_forecast(t_test)))


# question 4

print('train variance:', variance(average_forecast(t_train)))
print('test variance:', variance(average_forecast(t_test)))

