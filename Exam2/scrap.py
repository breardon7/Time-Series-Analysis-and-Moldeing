import statsmodels.api as sm
import numpy as np
na=1
nb=1
model = sm.tsa.ARIMA(endog = np.random.normal(size=1000), order=(na, 0, nb)).fit()