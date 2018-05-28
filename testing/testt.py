import sys
sys.path.append('../')
import firemodels.temperature as temp
import numpy as np
import matplotlib.pyplot as plt

def temperatureFocus(M, N):
    temperature = np.zeros((M,N))
    A = np.zeros((M,N))
    A[M//2,N//2] = 1
    A[M//2+1,N//2] = 1
    temperature = temperature + A * 600
    #A = np.zeros((M,N))
    return temperature,A

def temperatureFocusExp(M, N):
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, M)
    X, Y = np.meshgrid(x, y)
    A = np.zeros((M,N))
    return 1e3*np.exp(-1000*((X-.5)**2 + (Y-.5)**2)), A
  

  
# The resolution have to be lower than discrete version for computation of F
M, N = 500, 500

# Initial conditions
initial, A = temperatureFocus(M, N)


# Parameters
mu = 1/5 
T = 1000
dt = 1e-4
b = 8
maxTemp = 1000

#%%
# We have to include border conditions, for now only 
# use dirichlet f(x,y) = u(x,y) for (x,y) \in \partial\Omega
ct = temp.continuous(initial, mu, dt, T, b, maxTemp, A=A)

pde1, As, W = ct.solvePDE(1/8, 2000)
#spde1 = ct.solveSPDE1(1/30)
#spde2 = ct.solveSPDE2(1/5)

for i in range(T):
  if i % 100 == 0:
    ct.plotTemperatures(i, pde1)
#%%
Ea = 1
Z = .1
H = 5500
## Discrete
#dtemp = temp.discrete(mu, initial, T, A, b, maxTemp)
dtemp = temp.discrete(mu, initial, T, A, b, maxTemp, Ea, Z, H)
dtemps, _, fuel = dtemp.propagate()#4/30, 20)
#
for i in range(T):
  #if i % 100 == 0:
  dtemp.plotTemperatures(i, dtemps)