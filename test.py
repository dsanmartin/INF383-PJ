#import firemodels.cellularautomata as ca
import firemodels.cellularautomatalayers as cal
import numpy as np
#import matplotlib.pyplot as plt
#import ipywidgets as widgets

def fireFocus(M, N, i, j, size):
    focus = np.zeros((M, N))
    focus[i-size:i+size, j-size:j+size] = np.ones((2*size, 2*size)) 
    return focus
  
# Testing
temperature = np.load('data/temperature100x100.npy')
wind_speed = np.load('data/wind_speed100x100.npy')
humidity = np.load('data/humidity100x100.npy')
pressure = np.load('data/pressure100x100.npy')
wind_direction = np.load('data/wind_direction100x100.npy')

#%%

temperature = temperature / np.max(temperature)
wind_speed = wind_speed / np.max(wind_speed)
humidity = humidity / np.max(humidity)
pressure = pressure / np.max(pressure)

wd = np.ones_like(temperature)*0
#ws = np.ones_like(temperature)
#print(wd)

(M, N) = temperature.shape
world = [temperature, wind_speed, wd, humidity, pressure]
initial = fireFocus(M, N, 50, 50, 2)
#neighborhood = 'vonneumann'
neighborhood = 'moore'
alpha = .5
beta = 1-alpha
times = 30
rule = .3

automata = cal.new(initial, world, neighborhood, rule, alpha, beta)
states = automata.propagate(times)

for i in range(len(states)):
  automata.plotStates(i)