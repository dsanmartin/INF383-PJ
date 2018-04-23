import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

class new:   
  states = []
  
  def __init__(self, M, N, initial, rule, neighborhood):
    self.states = [initial]
    self.M = M
    self.N = N
    self.rule = rule
    self.neighborhood = neighborhood
 
  def propagate(self, timesteps):    
    for t in range(1, timesteps):
      tmp = self.checkNeighborhood(t-1)
      self.states.append(tmp)
      
    return self.states
  

  def checkNeighborhood(self, t):
    grid = self.states[t]
    summ = np.zeros_like(grid)

    if self.neighborhood == 'moore':
      summ = np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) \
        + np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) \
        + np.roll(np.roll(grid, 1, axis=0), 1, axis=1) \
        + np.roll(np.roll(grid, 1, axis=0), -1, axis=1) \
        + np.roll(np.roll(grid, -1, axis=0), 1, axis=1) \
        + np.roll(np.roll(grid, -1, axis=0), -1, axis=1) 
      summ /= 8
        
    elif self.neighborhood == 'vonneumann':
      summ = np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) \
        + np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1)
      summ /= 4
      
    summ[summ < self.rule] = 0
    summ[summ >= self.rule] = 1    
     
    return summ
    
    
  def plotStates(self, t):
    cmap = colors.ListedColormap(['green', 'yellow'])
    plt.imshow(self.states[t], origin='lower', cmap=cmap)
    plt.show()

            