import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm
#from matplotlib import cm

class new:
  temperatures = []
  
  def __init__(self, initial, sigma):
    self.temperatures = [initial]
    self.sigma = sigma
    
  def propagate(self, timesteps):
    for t in range(1, timesteps):
      grid = self.temperatures[t-1]
      noise = np.random.normal(0, self.sigma, size=grid.shape)
      temp = (1/5+noise)*(grid + np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) \
              + np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1))
      
      self.temperatures.append(temp)
      
    return self.temperatures
  
  def plotTemperatures(self, t):
    #cmap = colors.ListedColormap(['green', 'yellow'])
    plt.imshow(self.temperatures[t], origin='lower', cmap=cm.jet)#, vmin=0)#, 
               #vmin=0, vmax=np.max(np.array(self.temperatures)))
    plt.colorbar()
    plt.show()
    