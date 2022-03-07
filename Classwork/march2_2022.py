import numpy as np

N = 1000
mean_e = 1
var_e = 2

e = np.random.normal(mean_e, var_e, N)

y = np.zerso(len(e))
for t in range(N):
    if t = 0:
        y[0] = e[0]
    else:
        y[t] = -0.5*y[t-1] + e[t]

plt.plot(y)
plt.show()
print(f'mean of y data is {np.mean(y):.2f}')