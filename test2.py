import firemodels.temperature as temperature
import numpy as np

def temperatureFocus(M, N):
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, M)
    X, Y = np.meshgrid(x, y)
    return 1e3*np.exp(-1000*((X-.5)**2 + (Y-.5)**2))
  
M, N = 100, 100
initial = temperatureFocus(M, N)

times = 100

dtemp = temperature.new(initial, 1/30)
states = dtemp.propagate(times)

for i in range(len(states)):
  dtemp.plotTemperatures(i)

