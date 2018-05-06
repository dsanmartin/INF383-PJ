# -*- coding: utf-8 -*-
"""
Created on Sun May  6 14:37:35 2018

@author: iaaraya
"""

import firemodels.continuous.temperature as ctemp
import numpy as np
import matplotlib.pyplot as plt

def temperatureFocus(M, N):
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, M)
    X, Y = np.meshgrid(x, y)
    return 1e3*np.exp(-1000*((X-.5)**2 + (Y-.5)**2))
  
# The resolution have to be lower than discrete version for computation of F
M, N = 50, 50  

# Initial conditions
initial = temperatureFocus(M, N)


# Parameters
mu = 1/5 
T = 100
dt = 1e-4

# We have to include border conditions, for now only 
# use dirichlet f(x,y) = u(x,y) for (x,y) \in \partial\Omega
ct = ctemp.new(initial)

t, U = ct.solveStochasticPDE1(mu, dt, T)

for i in range(0, len(t)):
  ct.plotTemperatures(i)