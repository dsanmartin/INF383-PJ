import CellularAutomata
import numpy as np
import matplotlib.pyplot as plt

def fireFocus(M, N, i, j, size):
  focus = np.zeros((M, N))
  focus[i-size:i+size, j-size:j+size] = np.ones((size, size)) 
  return focus
  
# Testing
M = 101
N = 101
initial = fireFocus(M, N, 50, 50, 1)
rule = 1
neighborhood = 'vonneumann'#'moore'
times = 10

CA = CellularAutomata.CA(M, N, initial, rule, neighborhood)

sta = CA.propagate(times)
