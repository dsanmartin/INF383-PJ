import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

class new:   
  states = []
  
  def __init__(self, initial, world, neighborhood, rule, alpha, beta):
    self.states = [initial]
    self.world = world
    self.alpha = alpha
    self.beta = beta
    self.neighborhood = neighborhood
    self.rule = rule
 
  def propagate(self, timesteps):    
    for t in range(1, timesteps):
      tmp = self.checkNeighborhood(t-1)
      self.states.append(tmp)
      
    return self.states

    
  def checkNeighborhood(self, t):
    grid = self.states[t]
    neigh = np.zeros_like(grid)
    
    env = np.divide(np.multiply(self.world[0], self.world[1]), \
                            np.multiply(self.world[3], self.world[4]))
    
    # Convert degrees to cardinal
    wind = self.windDirectionConversion(self.world[2], self.neighborhood)

    # Get winds weights
    winds = self.createWindWeights(wind, self.neighborhood)
    
    # Check neihborhood
    if self.neighborhood == 'moore':
      neigh = winds[4]*np.roll(grid, 1, axis=0) + winds[0]*np.roll(grid, -1, axis=0) \
        + winds[6]*np.roll(grid, 1, axis=1) + winds[2]*np.roll(grid, -1, axis=1) \
        + winds[5]*np.roll(np.roll(grid, 1, axis=0), 1, axis=1) \
        + winds[3]*np.roll(np.roll(grid, 1, axis=0), -1, axis=1) \
        + winds[7]*np.roll(np.roll(grid, -1, axis=0), 1, axis=1) \
        + winds[1]*np.roll(np.roll(grid, -1, axis=0), -1, axis=1) 
#      neigh = np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) \
#        + np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) \
#        + np.roll(np.roll(grid, 1, axis=0), 1, axis=1) \
#        + np.roll(np.roll(grid, 1, axis=0), -1, axis=1) \
#        + np.roll(np.roll(grid, -1, axis=0), 1, axis=1) \
#        + np.roll(np.roll(grid, -1, axis=0), -1, axis=1) 
#        
      neigh /= 8
        
    elif self.neighborhood == 'vonneumann':
      neigh = winds[2]*np.roll(grid, 1, axis=0) + winds[0]*np.roll(grid, -1, axis=0) \
        + winds[3] * np.roll(grid, 1, axis=1) + winds[1]*np.roll(grid, -1, axis=1)
      #neigh = np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) \
      #  + np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1)
    
      neigh /= 4
    
    # Cells with possibility of burning
    possible = np.zeros_like(grid)
    possible[neigh > 0] = 1
    
    # Compute threshold
    threshold = (self.alpha * neigh + self.beta * env) * possible
    
    if self.rule > 0:
      threshold[threshold < self.rule] = 0
      threshold[threshold >= self.rule] = 1 
      out = threshold
    else:
      # Random values
      prob = np.random.uniform(size=grid.shape)
    
      # New burning trees
      prob[prob <= threshold] = 1 
      prob[prob != 1] = 0
      
      # New states
      out = prob + grid # Keep old states burning + new states
      out[out > 1] = 1

    
    return out
  
  def windDirectionConversion(self, wind_direction, neigh):
    if neigh == 'vonneumann':
      return np.around((wind_direction%360)/90).astype(int)
    elif neigh == 'moore':
      return np.around((wind_direction%360)/45).astype(int)
    else:
      return False
    
  def createWindWeights(self, wind, neigh):
    winds = list()
    if self.rule > 0:
      a, b = 1.1 + self.rule, .9
    else:
      a = 1.1 + self.rule #1.5
      b = 1-(a-1)#.9#.5#0.8#0.2#(1-a) / 3
      #c = 1#.8#.5
    
    if neigh == 'vonneumann':
      N = np.ones_like(wind).astype(float) #* c
      E = np.ones_like(wind).astype(float) #* c
      S = np.ones_like(wind).astype(float) #* b
      W = np.ones_like(wind).astype(float) #* c
      N[wind == 0] = a 
      S[wind == 0] = b
      N[wind == 4] = a
      S[wind == 4] = b
      E[wind == 1] = a
      W[wind == 1] = b
      S[wind == 2] = a
      N[wind == 2] = b
      W[wind == 3] = a
      E[wind == 3] = b
      #N[wind == 1] = b
      #N[wind == 3] = b
      #E[wind == 1] = a
      #S[wind == 2] = b
      #W[wind == 3] = a
      N[wind == 4] = a
      #print(np.max(N))
      winds = [N, E, S, W]
      
    elif neigh == 'moore':
      N = np.ones_like(wind).astype(float) #* b
      NE = np.ones_like(wind).astype(float) #* b
      E = np.ones_like(wind).astype(float) #* b
      SE = np.ones_like(wind).astype(float) #* b
      S = np.ones_like(wind).astype(float) #* b
      SW = np.ones_like(wind).astype(float) #* b
      W = np.ones_like(wind).astype(float) #* b
      NW = np.ones_like(wind).astype(float) #* b
      N[wind == 0] = a
      S[wind == 0] = b
      NE[wind == 1] = a
      SW[wind == 1] = b
      E[wind == 2] = a
      W[wind == 2] = b
      SE[wind == 3] = a
      NW[wind == 3] = b
      S[wind == 4] = a
      N[wind == 4] = b
      SW[wind == 5] = a
      NE[wind == 5] = b
      W[wind == 6] = a
      E[wind == 6] = b
      NW[wind == 7] = a
      SE[wind == 7] = b
      N[wind == 8] = a
      S[wind == 8] = b
      winds = [N, NE, E, SE, S, SW, W, NW]      
      
    return winds
    
    
  def plotStates(self, t):
    cmap = colors.ListedColormap(['green', 'yellow'])
    plt.imshow(self.states[t], origin='lower', cmap=cmap)
    plt.show()

            