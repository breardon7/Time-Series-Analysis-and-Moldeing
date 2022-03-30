import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt
import toolbox

mean_e = 2
a1 = .5
b1 = .8
mean_y = mean_e*(1+b1)/a1
print('mean: ', mean_y)

g0 = 1
g1 =