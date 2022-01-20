import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(r'/toolbox.py')
from toolbox import ADF_Cal, difference, plot_rolling_mean_var
#kpss_test

df = pd.read_excel(r'C:\Users\brear\OneDrive\Documents\GitHub\Time-Series-Analysis-and-Moldeing\Datasets\CO2_1970-2015_dataset_of_CO2_report_2016.xls',
                   header=0, parse_dates=[0], index_col=[0], squeeze=True)

CO2_In = df.loc['Indonesia']
CO2_Sw = df.loc['Sweden']

Year = np.arange(1970,2016)
plt.figure()
plt.plot(Year, CO2_In, label = 'Indonesia')
plt.plot(Year, CO2_Sw, label = 'Sweden')
plt.legend()
plt.title('CO2 Emission Comparison')
plt.xlabel('Date')
plt.ylabel('CO2 Emissions')
plt.show()

ADF_Cal(CO2_In)

# add from lab data into functions .py
plot_rolling_mean_var(CO2_In, 'Europe')
# kpss_test(CO2_In)

Europe = ['Estonia', 'Sweden', 'France', 'Germany', 'Hungary', 'Italy']
CO2 = df.loc[Europe, :]
CO2.loc['year'] = Year
CO2.T.plot(x = 'year', legend=None)
plt.legend(Europe)
plt.xlabel('Year')
plt.ylabel('CO2 Emissions')
plt.title('CO2 Emissions by European Countries')
plt.grid()
plt.show()

CO2_diff_01 = difference(CO2_In, 1)
CO2_diff_02 = difference(CO2_diff_01, 1)
plt.figure()
plt.plot(CO2_diff_01)
plt.show()
plt.figure()
plt.plot(CO2_diff_02)
plt.show()