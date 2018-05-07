import sys
sys.path.append('../')
import firemodels.continuous.temperature2 as ctemp
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
    return 1e3*np.exp(-1000*((X-.5)**2 + (Y-.5)**2)),A
  

  
# The resolution have to be lower than discrete version for computation of F
M, N = 50, 50

# Initial conditions
initial,A = temperatureFocus(M, N)


# Parameters
mu = 1/5 
T = 100
dt = 1e-3
b = 10
maxTemp = 1000

# We have to include border conditions, for now only 
# use dirichlet f(x,y) = u(x,y) for (x,y) \in \partial\Omega
ct = ctemp.new(initial, mu, dt, T, b, maxTemp)
pde1 = ct.solvePDE(A)
#spde1 = ct.solveSPDE1()
#spde2 = ct.solveSPDE2()

for i in range(T):
  ct.plotTemperatures(i, pde1)

