import matplotlib.pyplot as plt
import seaborn as sns
from toolbox import kpss_test

name = sns.get_dataset_names()

df = sns.load_dataset('flights')
plt.figure()
df.plot(x='year', y='passengers')
plt.show()

kpss_test(df.passengers)