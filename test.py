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

temperature = temperature / np.max(temperature)
wind_speed = wind_speed / np.max(wind_speed)
humidity = humidity / np.max(humidity)
pressure = pressure / np.max(pressure)

(M, N) = temperature.shape
world = [temperature, wind_speed, humidity, pressure]
initial = fireFocus(M, N, 5, 5, 4)
#neighborhood = 'vonneumann'
neighborhood = 'moore'
alpha = .7
beta = 1-alpha
times = 10

automata = cal.new(initial, world, neighborhood, alpha, beta)
states = automata.propagate(times)

for i in range(len(states)):
  automata.plotStates(i)