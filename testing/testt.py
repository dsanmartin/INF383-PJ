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
    A = np.zeros((M,N))
    return temperature,A

def temperatureFocus(M, N):
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, M)
    X, Y = np.meshgrid(x, y)
    A = np.zeros((M,N))
    return 1e3*np.exp(-1000*((X-.5)**2 + (Y-.5)**2)), A
  

  
# The resolution have to be lower than discrete version for computation of F
M, N = 100, 100

# Initial conditions
initial,A = temperatureFocus(M, N)


# Parameters
mu = 1/5 
T = 300
dt = 1e-4
b = 8000
maxTemp = 1000

# We have to include border conditions, for now only 
# use dirichlet f(x,y) = u(x,y) for (x,y) \in \partial\Omega
ct = temp.continuous(initial, mu, dt, T, b, maxTemp, A)
pde1, AA = ct.solvePDE()
#spde1 = ct.solveSPDE1(1/30)
#spde2 = ct.solveSPDE2(1/5)

for i in range(T):
  if i % 10 == 0:
    ct.plotTemperatures(i, pde1)

## Discrete
#dtemp = temp.discrete(mu, initial, T, A, b, maxTemp)
#dtemps, _ = dtemp.propagate(1/5, 1/5)
#
#for i in range(T):
#  if i % 10 == 0:
#    dtemp.plotTemperatures(i, dtemps)