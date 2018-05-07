# -*- coding: utf-8 -*-
"""
Created on Sun May  6 15:15:59 2018

@author: iaaraya
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm
#from matplotlib import cm

class new:
  temperatures = []
  
  def __init__(self, initial):
    self.temperatures = [initial]

  def propagate(self, timesteps,A):
    for t in range(1, timesteps):
      grid = self.temperatures[t-1]
      #noise = np.random.normal(0, self.sigma, size=grid.shape)
      east = np.roll(grid, 1, axis=0) 
      west = np.roll(grid, -1, axis=0)
      north = np.roll(grid, -1, axis=1)
      south = np.roll(grid, 1, axis=1)
      
      #temp = (1/5)*(grid + east + west + north + south) \
      #        + A*(grid*((1000 - grid)/8000 + 4/5) + \
      #      (east + west + north + south)*((1000 - grid)/2000 - 1/5))
      
      temp = (1 - A)*(1/5)*(grid + east + west + north + south) \
              + A*(grid*(1000 - grid)/8000 + grid)
      
      self.temperatures.append(temp)
      
      A[temp >= 600] = 1
      A[temp < 600] = 0
     
      
    return self.temperatures
  
  def plotTemperatures(self, t):
    #cmap = colors.ListedColormap(['green', 'yellow'])
    plt.imshow(self.temperatures[t], origin='lower', cmap=cm.jet)#, vmin=0)#, 
               #vmin=0, vmax=np.max(np.array(self.temperatures)))
    plt.colorbar()
    plt.show()
    