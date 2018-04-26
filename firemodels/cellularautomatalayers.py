import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

class new:   
  states = []
  
  def __init__(self, initial, world, neighborhood, alpha, beta):
    self.states = [initial]
    self.world = world
    self.alpha = alpha
    self.beta = beta
    self.neighborhood = neighborhood
 
  def propagate(self, timesteps):    
    for t in range(1, timesteps):
      tmp = self.checkNeighborhood(t-1)
      self.states.append(tmp)
      
    return self.states

    
  def checkNeighborhood(self, t):
    grid = self.states[t]
    neigh = np.zeros_like(grid)
    
    env = np.divide(np.multiply(self.world[0], self.world[1]), \
                            np.multiply(self.world[2], self.world[3]))
    # Check neihborhood
    if self.neighborhood == 'moore':
      neigh = np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) \
        + np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) \
        + np.roll(np.roll(grid, 1, axis=0), 1, axis=1) \
        + np.roll(np.roll(grid, 1, axis=0), -1, axis=1) \
        + np.roll(np.roll(grid, -1, axis=0), 1, axis=1) \
        + np.roll(np.roll(grid, -1, axis=0), -1, axis=1) 
        
      neigh /= 8
        
    elif self.neighborhood == 'vonneumann':
      neigh = np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) \
        + np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1)
    
      neigh /= 4
    
    # Cells with possibility of burning
    possible = np.zeros_like(grid)
    possible[neigh > 0] = 1
    
    # Compute threshold
    threshold = (self.alpha * neigh + self.beta * env) * possible
    
    # Random values
    #prob = np.random.uniform(size=grid.shape)
  
    # New burning trees
    #prob[prob <= threshold] = 1 
    #prob[prob != 1] = 0
    
    # New states
    #out = prob + grid # Keep old states burning + new states
    #out[out > 1] = 1
    threshold[threshold < 0.5] = 0
    threshold[threshold >= 0.5] = 1 
    
    out = threshold
       
    if False:
      print("Neighborhood")
      print(np.min(neigh), np.max(neigh))
      plt.pcolor(neigh)
      plt.show()
      
      print("Environment")
      print(np.min(env), np.max(env))
      plt.pcolor(env)
      plt.show()
      
      print("Prob")
      print(np.min(prob), np.max(prob))
      plt.pcolor(prob)
      plt.show()
      
      print("Threshold")
      print(np.min(threshold), np.max(threshold))
      plt.pcolor(threshold)
      plt.show()
      
      
     
    return out
    
    
  def plotStates(self, t):
    cmap = colors.ListedColormap(['green', 'yellow'])
    plt.imshow(self.states[t], origin='lower', cmap=cmap)
    plt.show()

            