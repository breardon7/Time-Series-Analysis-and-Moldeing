import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib

import toolbox_updated


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

samples_size = 10000
ar_order = int(ar_order)
ma_order = int(ma_order)
mean = 0
var = 1

acf_lags = 20
arma_process = sm.tsa.ArmaProcess(ar, ma)
mean_y = mean * (1 + np.sum(ma_inputs)) / (1 + np.sum(ar_inputs))
y = arma_process.generate_sample(samples_size, scale=np.sqrt(var) + mean_y)
ry = arma_process.acf(lags=acf_lags)

na = 5
nb = 5

if len(ar_inputs) + len(ma_inputs) > 2:
    na = 7
    nb = 7


toolbox_updated.gpac_calc(ry, na, nb)
model = sm.tsa.ARMA(y, (na, nb)).fit(trend='nc', disp=0)
for i in range(na):
    print("The AR Coefficient a{}".format(i), "is:", model.params[i])

for i in range(na):
    print("The MA Coefficient b{}".format(i), "is:", model.params[i + na])

intervals = model.conf_int()
for i in range(na):
    print("The Confidence Interval for a{}".format(i), "is:", intervals[i])
    print("The p-value for a{}".format(i), "is:", model.pvalues[i])
    print("The Standard Error for a{}".format(i), "is:", model.bse[i])
    print("\n")

for i in range(na):
    print("The Confidence Interval for b{}".format(i), "is:", intervals[i + na])
    print("The p-value for b{}".format(i), "is:", model.pvalues[i + na])
    print("The Standard Error for b{}".format(i), "is:", model.bse[i + na])
    print("\n")

print(model.summary())