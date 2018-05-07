# -*- coding: utf-8 -*-
"""
Created on Sun May  6 15:35:05 2018

@author: iaaraya
"""
import sys
sys.path.append('../')
import firemodels.discrete.temperature2 as temperature
import numpy as np

def temperatureFocus(M, N):
    temperature = np.zeros((M,N))
    A = np.zeros((M,N))
    A[M//2,N//2] = 1
    A[M//2+1,N//2] = 1
    temperature = temperature + A * 600
    return temperature,A

  
M, N = 100, 100
initial,A = temperatureFocus(M, N)

times = 200

dtemp = temperature.new(initial)
states = dtemp.propagate(times,A)

for i in range(len(states)):
  dtemp.plotTemperatures(i)