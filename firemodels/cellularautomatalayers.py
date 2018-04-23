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
    
    threshold = self.alpha * neigh + self.beta * env
    
    prob = np.random.uniform(size=grid.shape)
  
    #out = np.ones_like(grid)
    prob[prob <= threshold] = 1
    prob[prob != 1] = 0
    #
    #out *= prob
       
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
      
      
     
    return prob#out
    
    
  def plotStates(self, t):
    cmap = colors.ListedColormap(['green', 'yellow'])
    plt.imshow(self.states[t], origin='lower', cmap=cmap)
    plt.show()

            