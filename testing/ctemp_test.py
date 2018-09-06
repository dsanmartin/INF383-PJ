import sys
sys.path.append('../')
import firemodels.temperature as ctemp
import numpy as np
import matplotlib.pyplot as plt

def temperatureFocus(M, N):
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, M)
    X, Y = np.meshgrid(x, y)
    return 1.5e3*np.exp(-1000*((X-.5)**2 + (Y-.5)**2))
  
# The resolution have to be lower than discrete version for computation of F
u = lambda x, y: 1e3*np.exp(-100*((x-1)**2 + (y-1)**2))

## Parameters
#mu = 1/5 
#gamma = 10
#T = 500
#dt = 1e-4
#T_ref = 50
#V = (1, -1)
#A = np.ones_like(u0)
#y0 = np.ones_like(u0)
#Z = 10
#h = 1e0
#Ea = 1
#H = 550

# Domain
M, N = 20, 20  
T = 10
dt = 1e-2
x = np.linspace(0, 2, N)
y = np.linspace(0, 2, M)
t = np.linspace(0, dt*T, T)
X, Y = np.meshgrid(x, y)
dx = x[1] - x[0]
dy = y[1] - y[0]

print(dt, dx, dy)

# Initial conditions
#u0 = temperatureFocus(M, N)

# Parameters
u0 = u(X, Y)
mu = 1/5 #* dt / (dx * dy)
gamma = 10 # dt * dx
T_ref = 30
V = (-.1, -.1)
A = np.ones_like(u0)
y0 = np.ones_like(u0)
Z = .1
h = 1e-2
Ea = 1
H = 5500


plt.contourf(X, Y, u0, cmap=plt.cm.jet)
plt.colorbar()
plt.show()

#ct = ctemp.continuous(initial, mu, dt, T)
ct = ctemp.continuous(x, y, t, u0, y0, V, mu, gamma, A, Z, H, Ea, h, T_ref)

U, AA, Y = ct.solvePDE()

for i in range(T):
  #if i %10 == 0:
  ct.plotTemperatures(i, U)
  plt.imshow(AA[i], origin="lower", cmap=plt.cm.afmhot)
  plt.show()
  plt.imshow(Y[i], origin="lower")
  plt.show()
  
#ct.plotTemperatures(len(U)-1, U)
#plt.imshow(Y[-1], origin="lower")

#%%
