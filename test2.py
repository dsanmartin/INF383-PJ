import firemodels.discrete as temperature
import numpy as np

def fireFocus(M, N, i, j, size):
    focus = np.zeros((M, N))
    focus[i-size:i+size, j-size:j+size] = 1e3*np.ones((2*size, 2*size)) 
    return focus
  
M, N = 100, 100
initial = fireFocus(M, N, 50, 50, 2)

times = 100

dtemp = temperature.new(initial)
states = dtemp.propagate(times)

for i in range(len(states)):
  dtemp.plotTemperatures(i)

