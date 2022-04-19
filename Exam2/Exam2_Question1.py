import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from matplotlib import pyplot as plt

df = pd.read_csv("question1.csv")
print(df.head())

aneuploid_tumor = df[df["type"] == 1]
diploid_tumor = df[df["type"] == 2]
new_df = pd.DataFrame()
new_df["Type1"] = pd.to_numeric(np.array(aneuploid_tumor.delta))
new_df["Type2"] = pd.to_numeric(np.concatenate((diploid_tumor.delta, np.repeat(np.array(diploid_tumor["type"].median()), 24))))


kmf = KaplanMeierFitter()
T = df['time']
E = df['delta']
type = df['type']
ix1 = (type == 1)
ix2 = (type == 2)
kmf.fit(T[ix1], E[ix1], label='Aneuploid')
ax = kmf.plot_survival_function()
kmf.fit(T[ix2], E[ix2], label='Diploid')
ax1 = kmf.plot_survival_function(ax=ax)
plt.xlabel("Time")
plt.ylabel("Survival")
plt.show()
