import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

#%%
my_data = np.genfromtxt('../data/originald08.csv', delimiter=',', skip_header=1)
v_spe = my_data[:, 4]
v_dir = my_data[:, 7] 

del my_data

#%%
T = 50
speed = v_spe[-T:]#v_spe[:T]
direction = np.radians(v_dir[:T])
t = np.linspace(0, 10*T, T)
plt.plot(t, speed)
plt.plot(t, direction)
plt.show()

fs = interpolate.interp1d(t, speed, kind='cubic')
fd = interpolate.interp1d(t, direction, kind='cubic')
fine_t = np.linspace(0, 10*T, 10*T)
int_speed = fs(fine_t)
int_direction = fd(fine_t)
plt.plot(fine_t, int_speed)
plt.plot(fine_t, int_direction)
plt.show()


xv = int_speed * np.cos(int_direction)
yv = int_speed * np.sin(int_direction)
plt.plot(xv)
plt.plot(yv)
plt.show()

#%%

XX = np.ones((10, 10))
YY = np.ones_like(XX)

for i in range(0, len(xv), 25):
  plt.quiver(XX*xv[i], YY*yv[i])
  plt.show()
