import numpy as np
import matplotlib.pyplot as plt

class new:   
  states = []
  
  def __init__(self, M, N, initial, rule, neighborhood):
    self.states = []
    self.M = M
    self.N = N
    self.initial = initial
    self.rule = rule
    self.neighborhood = neighborhood
    self.initial = initial
 
  def propagate(self, timesteps):
    self.states.append(self.initial)
    
    for t in range(1, timesteps):
      #tmp = np.zeros((self.M, self.N))
      #for i in range(1, self.M-1):
      #  for j in range(1, self.N-1):
      #    tmp[i, j] = self.checkNeighboorhod(t-1, i, j)
      
      tmp = self.checkNeighborhood(t-1)
      self.states.append(tmp)
      
    return self.states
  
  def checkNeighborhoodOld(self, t, i, j):
    grid = self.states[t]
    
    if self.neighborhood == 'moore':
      summ = grid[i-1, j-1] + grid[i-1, j] + grid[i-1, j + 1] \
        + grid[i, j-1] + grid[i, j] + grid[i, j+1] \
        + grid[i+1, j-1] + grid[i+1, j] + grid[i+1, j+1]        
    elif self.neighborhood == 'vonneumann':
      summ = grid[i-1, j] + grid[i+1, j] + grid[i, j-1] + grid[i, j+1]
      
    if summ >= self.rule: return 1
    else: return 0
    
  def checkNeighborhood(self, t):
    grid = self.states[t]
    summ = np.zeros_like(grid)

    if self.neighborhood == 'moore':
      summ = grid + np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) \
        + np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) \
        + np.roll(np.roll(grid, 1, axis=0), 1, axis=1) \
        + np.roll(np.roll(grid, 1, axis=0), -1, axis=1) \
        + np.roll(np.roll(grid, -1, axis=0), 1, axis=1) \
        + np.roll(np.roll(grid, -1, axis=0), -1, axis=1) 
        
    elif self.neighborhood == 'vonneumann':
      summ = grid + np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) \
        + np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) + grid
      
    summ[summ < self.rule] = 0
    summ[summ >= self.rule] = 1    
     
    return summ
    
    
  def plotStates(self, t):
    plt.imshow(self.states[t], origin='lower')
    plt.show()

            