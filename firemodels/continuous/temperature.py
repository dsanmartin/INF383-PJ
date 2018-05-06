import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp2d
#from matplotlib import cm

class new:
  temperatures = []
  
  def __init__(self, u0):
    self.u0 = u0
    self.temperatures = [u0.flatten()]
    self.M, self.N = u0.shape
    self.x = np.linspace(0, 1, self.M)
    self.y = np.linspace(0, 1, self.N)
    self.dx = self.x[1] - self.x[0]
    self.dy = self.y[1] - self.y[0]
    
  
  def F(self, U, t, mu):    
    U = U.reshape((self.M, self.N))
    W = np.zeros_like(U)
    
    # Laplacian
    #W[1:-1,1:-1] = mu*(np.diff(U[:,1:-1], n=2, axis=0) / self.dx**2 + \
    # np.diff(U[1:-1,:], n=2, axis=1) / self.dy**2)
    dx = self.dx
    dy = self.dy
    W = mu * (np.gradient(np.gradient(U, dx, axis=0), dx, axis=0)+ \
      np.gradient(np.gradient(U, dy, axis=1), dy, axis=1))
    
    return W.flatten() # Flatten for odeint
    
    
  # Solve PDE# Solve 
  def solvePDE(self, mu, dt, T):
    t = np.linspace(0, dt*T, T)
    # Method of lines
    U = odeint(self.F, self.u0.flatten(), t, args=(mu,)) 
    
    self.temperatures.extend(U)
    
    return t, self.temperatures

  def plotTemperatures(self, t):
    fine = np.linspace(0, 1, 2*self.N)
    fu = interp2d(self.x, self.y, self.temperatures[t].reshape(self.u0.shape), kind='cubic')
    U = fu(fine, fine)
    plt.imshow(U, origin='lower', cmap=plt.cm.jet)
    plt.colorbar()
    plt.show()
    