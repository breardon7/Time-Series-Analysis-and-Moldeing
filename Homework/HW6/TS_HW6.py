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
ax1.plot(passengers[:50], label='original', c='g')
ax1.plot(t3[:50], label='MA-3', c='b')
ax2.plot(passengers[:50], label='original', c='g')
ax2.plot(t5[:50], label='MA-5', c='b')
ax3.plot(passengers[:50], label='original', c='g')
ax3.plot(t7[:50], label='MA-7', c='b')
ax4.plot(passengers[:50], label='original', c='g')
ax4.plot(t9[:50], label='MA-9', c='b')
ax1.set_title('MA-3')
ax2.set_title('MA-5')
ax3.set_title('MA-7')
ax4.set_title('MA-9')
plt.tight_layout
plt.legend(loc='upper left')
plt.show()

# question 3
t_first, t4 = simulate_MA(passengers)
t_second, t6 = simulate_MA(passengers)
t_third, t8 = simulate_MA(passengers)
t_fourth, t10 = simulate_MA(passengers)
fig, ax = plt.subplots(2,2)
ax1, ax2, ax3, ax4 = ax.flatten()
fig.suptitle('Trend 2xN vs MA')
ax1.plot(passengers[:50], label='original', c='g')
ax1.plot(t4[:50], label='MA-4', c='b')
ax2.plot(passengers[:50], label='original', c='g')
ax2.plot(t6[:50], label='MA-6', c='b')
ax3.plot(passengers[:50], label='original', c='g')
ax3.plot(t8[:50], label='MA-8', c='b')
ax4.plot(passengers[:50], label='original', c='g')
ax4.plot(t10[:50], label='MA-10', c='b')
ax1.set_title('MA-4')
ax2.set_title('MA-6')
ax3.set_title('MA-8')
ax4.set_title('MA-10')
plt.tight_layout
plt.legend(loc='upper left')
plt.show()



