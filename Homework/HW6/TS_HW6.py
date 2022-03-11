import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from toolbox import simulate_MA

# question 1
df = pd.read_csv(r'C:\Users\brear\OneDrive\Desktop\Grad School\Time-Series-Analysis-and-Moldeing\Datasets\AirPassengers.csv')
passengers = df['#Passengers']

t1, t2 = simulate_MA(passengers)

plt.plot(t1)
plt.title('t1')
plt.show()

plt.plot(t2)
plt.title('t2')
plt.show()


# question 2
t3 = simulate_MA(passengers)
t5 = simulate_MA(passengers)
t7 = simulate_MA(passengers)
t9 = simulate_MA(passengers)
fig, ax = plt.subplots(2,2)
ax1, ax2, ax3, ax4 = ax.flatten()
fig.suptitle('Trend vs MA')
ax1.plot(passengers, label='original', c='g')
ax1.plot(t3, label='MA-3', c='b')
ax2.plot(passengers, label='original', c='g')
ax2.plot(t5, label='MA-5', c='b')
ax3.plot(passengers, label='original', c='g')
ax3.plot(t7, label='MA-7', c='b')
ax4.plot(passengers, label='original', c='g')
ax4.plot(t9, label='MA-9', c='b')
ax1.set_title('MA-3')
ax2.set_title('MA-5')
ax3.set_title('MA-7')
ax4.set_title('MA-9')
plt.tight_layout
plt.legend(loc='upper left')
plt.show()



