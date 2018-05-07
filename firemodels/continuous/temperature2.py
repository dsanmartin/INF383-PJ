# -*- coding: utf-8 -*-
"""
Created on Sun May  6 20:16:41 2018

@author: iaaraya
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp2d
#from matplotlib import cm

class new:
  
  def __init__(self, u0, mu, dt, T, b, maxTemp):
    self.u0 = u0
    self.mu = mu
    self.dt = dt
    self.T = T
    self.M, self.N = u0.shape
    self.x = np.linspace(0, 1, self.M)
    self.y = np.linspace(0, 1, self.N)
    self.dx = self.x[1] - self.x[0]
    self.dy = self.y[1] - self.y[0]
    #self.dt = self.dx
    self.t = np.linspace(0, dt*T, self.T)
    self.dt = self.t[1] - self.t[0]
    self.b = b
    self.maxTemp = maxTemp
    
  
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
    
    
  # Solve PDE
  def solvePDE(self,A):
    
    U = np.zeros((self.T+1, self.u0.flatten().shape[0]))
    U[0,:] = self.u0.flatten()
    A = A.flatten()
    
    for i in range(1, self.T+1):
        A[U[i-1,:] >= 400] = 1
        A[U[i-1,:] < 400] = 0
        
        W =  self.F(U[i-1,:], self.t, self.mu)
        
        U[i,:] = U[i-1,:] + (1 - A)*W*self.dt + A*(self.maxTemp - U[i-1,:])*U[i-1,:]*self.dt/self.b   
        
        
    return U

  def solveSPDE1(self):
    # Solve
    U = np.zeros((self.T+1, self.u0.flatten().shape[0]))
    U[0,:] = self.u0.flatten()
    
    for i in range(1, self.T+1):
        W =  self.F(U[i-1,:], self.t, self.mu)
        U[i,:] = U[i-1,:] + W*self.dt + np.random.normal(0, self.dt, W.shape)
    
    return U
    

  def solveSPDE2(self):
    # Solve
    U = np.zeros((self.T+1,self.u0.flatten().shape[0]))
    U[0,:] = self.u0.flatten()
    for i in range(1, self.T+1):
        W =  self.F(U[i-1,:], self.t, self.mu)
        U[i,:] = U[i-1,:] + W*self.dt + np.random.normal(0, self.dt, W.shape)*W
    
    return U
    

  def plotTemperatures(self, t, temperatures):
    fine = np.linspace(0, 1, 2*self.N)
    fu = interp2d(self.x, self.y, temperatures[t].reshape(self.u0.shape), kind='cubic')
    U = fu(fine, fine)
    plt.imshow(U, origin='lower', cmap=plt.cm.jet)
    plt.colorbar()
    plt.show()
    