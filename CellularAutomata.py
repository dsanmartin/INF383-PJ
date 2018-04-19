import numpy as np
import matplotlib.pyplot as plt

class CA:   
  states = []
  
  def __init__(self, M, N, initial, rule, neighborhood):
    self.M = M
    self.N = N
    self.initial = initial
    self.rule = rule
    self.neighborhood = neighborhood
    self.initial = initial
 
  def propagate(self, timesteps):
    self.states.append(self.initial)
    
    for t in range(1, timesteps):
      tmp = np.zeros((self.M, self.N))
      for i in range(1, self.M-1):
        for j in range(1, self.N-1):
          tmp[i, j] = self.checkNeighboorhod(t-1, i, j)
      
      self.states.append(tmp)
      self.plotStates(t)
      
    return self.states
  
  def checkNeighboorhod(self, t, i, j):
    grid = self.states[t]
    
    if self.neighborhood == 'moore':
      summ = grid[i-1, j-1] + grid[i-1, j] + grid[i-1, j + 1] \
        + grid[i, j-1] + grid[i, j] + grid[i, j+1] \
        + grid[i+1, j-1] + grid[i+1, j] + grid[i+1, j+1]        
    elif self.neighborhood == 'vonneumann':
      summ = grid[i-1, j] + grid[i+1, j] + grid[i, j-1] + grid[i, j+1]
      
    if summ >= self.rule: return 1
    else: return 0
    
  def plotStates(self, t):
    plt.imshow(self.states[t], origin='lower')
    plt.show()

            