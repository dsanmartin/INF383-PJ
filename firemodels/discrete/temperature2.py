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
  #temperatures = []
  
  def __init__(self, initial, timesteps, A):
    self.initial = initial
    self.timesteps = timesteps
    self.A = [A]

  def propagate(self, n_t=0 , sigma=0):
    self.temperatures = [self.initial]
    for t in range(1, self.timesteps):
      A = self.A[t-1]
      grid = self.temperatures[t-1]
      #noise = np.random.normal(0, self.sigma, size=grid.shape)
      east = np.roll(grid, 1, axis=0) 
      west = np.roll(grid, -1, axis=0)
      north = np.roll(grid, -1, axis=1)
      south = np.roll(grid, 1, axis=1)
      
      #temp = (1/5)*(grid + east + west + north + south) \
      #        + A*(grid*((1000 - grid)/8000 + 4/5) + \
      #      (east + west + north + south)*((1000 - grid)/2000 - 1/5))
      
      n1 = np.zeros_like(grid)
      n2 = np.zeros_like(grid)
            
      if n_t == 1:
        n1 = np.random.normal(0, sigma, size=grid.shape)
      elif n_t == 2:
        n2 = np.random.normal(0, sigma, size=grid.shape)
      
      temp = (1 - A)*(1/5 + n1)*(grid + east + west + north + south) \
              + A*(grid*(1000 - grid)/8000 + grid) + n2
      
      self.temperatures.append(temp)
      
      tmp = np.zeros(A.shape)
      tmp[temp >= 400] = 1
      tmp[temp < 400] = 0
      
      self.A.append(tmp)
      
    return self.temperatures, self.A
  
  def plotTemperatures(self, t, temperatures):
    plt.imshow(temperatures[t], origin='lower', cmap=cm.jet)
    plt.colorbar()
    plt.show()
    