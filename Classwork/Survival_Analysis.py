from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

durations = [5,6,6,2,5,4]
event = [1,0,0,1,1,1]
ax = plt.subplot(111)
kmf = KaplanMeierFitter()
kmf.fit(durations, event, label='number of minutes a person stays on website')
kmf.plot_survival_function(ax=ax)
plt.show()