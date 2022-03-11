from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from toolbox import ACF

# y(t) + 0.5y(t-1) + 0.06y(t-2) = e(t)
# e(t) ~ WN(0,1)

# dlsim()
num = [1, 0, 0]
den = [1, 0.5, 0.06]
e = np.random.normal(0, 1, 1000)
system = (num,den,1)
_,y = signal.dlsim(system, e)

plt.figure()
plt.plot(y)
plt.show()

plt.figure()
plt.hist(y, bins=50)
plt.show()

# stats model
N = int(input('enter number of input: '))
an = [0.5, 0.06]
bn = [0, 0]
arparams = np.array(an)
maparams = np.array(bn)

ar = np.r_[1, arparams]
ma = np.r_[1, maparams]

mean_e = int(input('enter the mean of WN: '))
var_e = int(input('enter the var of WN: '))

arma_process = sm.tsa.ArmaProcess(ar, ma)

y_mean = mean_e/(1+np.sum(an))
y = arma_process.generate_sample(1000, scale=np.sqrt(var_e)) + y_mean

e = np.random.normal(mean_e, np.sqrt((var_e)), N)

plt.figure()
plt.plot(y, 'b', label='output')
plt.plot(e, 'r', label='white noise')
plt.show()

plt.figure()
plt.hist(y, bins=50)
plt.show()

plt.figure()
plt.hist(e, bins=50)
plt.show()

ACF(y, 50, 'y')
plt.show()

ACF(e, 50, 'e')
plt.show()