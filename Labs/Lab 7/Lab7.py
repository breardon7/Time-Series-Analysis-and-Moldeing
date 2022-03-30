import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt
import toolbox

print("====================QUESTION 1==========================")
samples_size = input("Enter # of samples: ")
mean = input("Enter Mean (WN): ")
var = input("Enter Variance (WN): ")
ar_order = input("Enter AR order: ")
ma_order = input("Enter MA order: ")

ar_inputs = []
if int(ar_order) > 0:
    for i in range(int(ar_order)):
        text = "Enter a" + str(i + 1) + ": "
        input_value = input(text)
        ar_inputs.append(float(input_value))

ma_inputs = []
if int(ma_order) > 0:
    for i in range(int(ma_order)):
        text = "Enter b" + str(i + 1) + ": "
        input_value = input(text)
        ma_inputs.append(float(input_value))

ar = np.r_[1, ar_inputs]
ma = np.r_[1, ma_inputs]

samples_size = int(samples_size)
ar_order = int(ar_order)
ma_order = int(ma_order)
mean = float(mean)
var = float(var)

arma_process = sm.tsa.ArmaProcess(ar, ma)
mean_y = mean * (1 + np.sum(ma_inputs)) / (1 + np.sum(ar_inputs))
y = arma_process.generate_sample(samples_size, scale=np.sqrt(var) + mean_y)


acf_lags = 60
ry = arma_process.acf(lags=acf_lags)
toolbox.gpac_calc(ry, 5, 5)

acf_lags = 15
new_ry = toolbox.auto_correlation_cal(y, acf_lags)
toolbox.gpac_calc(new_ry, 7, 7)

lags = 20

acf = arma_process.acf(lags=lags)
pacf = arma_process.pacf(lags=lags)

fig, axs = plt.subplots(2, 1)
fig.subplots_adjust(hspace=1.5, wspace=0.5)
axs = axs.ravel()

ry = arma_process.acf(lags=lags)
a1 = ry
a2 = a1[::-1]
a = np.concatenate((a2[:-1], a1))
x1 = np.arange(0, lags)
x2 = -x1[::-1]
x = np.concatenate((x2[:-1], x1))
(marker, stemlines, baselines) = axs[0].stem(x, a,
                                             use_line_collection=True, markerfmt='o')
plt.setp(marker, color='red', marker='o')
plt.setp(baselines, color='gray', linewidth=2, linestyle='-')
m = 1.96 / np.sqrt(100)
axs[0].axhspan(-m, m, alpha=.2, color='blue')
axs[0].set_title("ACF GPAC")
axs[0].set_ylabel("ACF")
axs[0].set_xlabel("Frequency")

ry = arma_process.pacf(lags=lags)
a1 = ry
a2 = a1[::-1]
a = np.concatenate((a2[:-1], a1))
x1 = np.arange(0, lags)
x2 = -x1[::-1]
x = np.concatenate((x2[:-1], x1))
(marker, stemlines, baselines) = axs[1].stem(x, a,
                                             use_line_collection=True, markerfmt='o')
plt.setp(marker, color='red', marker='o')
plt.setp(baselines, color='gray', linewidth=2, linestyle='-')
m = 1.96 / np.sqrt(100)
axs[1].axhspan(-m, m, alpha=.2, color='blue')
axs[1].set_title("PACF GPAC")
axs[1].set_ylabel("PACF")
axs[1].set_xlabel("Frequency")

plt.show()


samples_size = 5000
arma_process = sm.tsa.ArmaProcess(ar, ma)
mean_y = mean * (1 + np.sum(ma_inputs)) / (1 + np.sum(ar_inputs))
arma_process.generate_sample(samples_size, scale=np.sqrt(var) + mean_y)


acf_lags = 60
ry = arma_process.acf(lags=acf_lags)
toolbox.gpac_calc(ry, 5, 5)


samples_size = 10000
arma_process = sm.tsa.ArmaProcess(ar, ma)
mean_y = mean * (1 + np.sum(ma_inputs)) / (1 + np.sum(ar_inputs))
arma_process.generate_sample(samples_size, scale=np.sqrt(var) + mean_y)

acf_lags = 60
ry = arma_process.acf(lags=acf_lags)
toolbox.gpac_calc(ry, 5, 5)
