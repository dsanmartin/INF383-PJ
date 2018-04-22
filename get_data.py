import numpy as np
import time
import requests
import matplotlib.pyplot as plt
from matplotlib import cm
#from WunderWeather import weather
from scipy import interpolate

#extractor = weather.Extract('35c89267da3bc202')
#[location,current] = extractor.features("-38.7290843,-72.6378264",(('geolookup',''),('now','')))
#print("Current Temperature in %s is: %s" %(location.data.city, current.temp_c))

base = "http://api.wunderground.com/api/35c89267da3bc202/conditions/q/"

def getWeather(start_lat, start_lng, end_lat, end_lng, M, N):
  delta_lat = (end_lat - start_lat) / M
  delta_lng = (end_lng - start_lng) / N
  
  temperature = np.zeros((M, N))
  wind_speed = np.zeros((M, N))
  wind_direction = np.zeros((M, N))
  relative_humidity = np.zeros((M, N))
  pressure = np.zeros((M, N))
  
  
  for i in range(M):
    for j in range(N):
      lat = start_lat + i * delta_lat
      lng = start_lng + j * delta_lng
            
      #_, current = extractor.features(str(lat) + "," + str(lng),(('geolookup',''),('now','')))

      url = base + str(lat) + "," + str(lng) + ".json" 
      data = requests.get(url).json()['current_observation']

      t = float(data['temp_c'])
      ws = float(data['wind_kph'])
      wd = float(data['wind_degrees'])
      rh = float(data['relative_humidity'][:-1])
      p = float(data['pressure_mb'])
    
      print("Temperature: ", t)
      print("Wind speed: ", ws)
      print("Wind direction: ", wd)
      print("Relative humidity: ", rh)
      print("Pressure: ", p)
        
      temperature[i, j] = t
      wind_speed[i, j] = ws
      wind_direction[i, j] = wd
      relative_humidity[i, j] = rh
      pressure[i, j] = p
      
      time.sleep(1)
      
  return temperature, wind_speed, wind_direction, relative_humidity, pressure

def interpolateScalar(data, kind='linear'):
  x = np.linspace(0, 1, len(data))
  xfine = np.linspace(0, 1, 100)
  Z = None

  if kind == "rbf":
    xrbf, yrbf = np.meshgrid(x, x)
    rbfi = interpolate.Rbf(xrbf.flatten(), yrbf.flatten(), 
                           data.flatten(), epsilon=.25) 
    X, Y = np.meshgrid(xfine, xfine)
    Z = rbfi(X, Y)
  else:
    f = interpolate.interp2d(x, x, data, kind=kind)
    Z = f(xfine, xfine)
    
  return Z

def interpolateVector(X, Y):
  WX = X.flatten().reshape(-1, 1)
  WY = Y.flatten().reshape(-1, 1)
  W = np.concatenate((WX, WY), axis=1)
  x = np.linspace(0, 1, len(X))
  xfine = np.linspace(0, 1, 100)
  
  xrbf, yrbf = np.meshgrid(x, x)
  print(xrbf.flatten().shape, yrbf.flatten().shape, W.shape)
  rbfi = interpolate.Rbf(xrbf.flatten(), yrbf.flatten(), 
                         W, epsilon=.25) 
  X, Y = np.meshgrid(xfine, xfine)
  Z = rbfi(X, Y)
  
  return Z
  
def plotScalar(data, cmap_):
  #plt.contour(temperature, cmap=cm.jet)
  #plt.imshow(temperature, cmap=cm.jet)
  plt.pcolor(data, cmap=cmap_)
  plt.show()
  
def plotVector(U, V):
  plt.quiver(U, V)
  plt.show()
  
def createWind(wind_speed, wind_direction):
  angle = np.radians((wind_direction + 180) % 360)
  X = np.multiply(wind_speed, np.cos(angle))
  Y = np.multiply(wind_speed, np.sin(angle))
  
  return X, Y

#M, N = 10, 10
#temperature = getWeather(-35.3803, -72.3652, -35.5235, -72.1894, M, N)
#_, _, wind_direction, relative_humidity, pressure = getWeather(-35.3803, -72.3652, -35.5235, -72.1894, M, N)
#np.save('data/temperature.npy', temperature)
#np.save('data/wind_speed.npy', wind_speed)
#np.save('data/wind_direction.npy', wind_direction)
#np.save('data/humidity.npy', relative_humidity)
#np.save('data/pressure.npy', pressure)
  
temperature = np.load('data/temperature.npy')
wind_speed = np.load('data/wind_speed.npy')
wind_direction = np.load('data/wind_direction.npy')
humidity = np.load('data/humidity.npy')
pressure = np.load('data/pressure.npy')

WX, WY = createWind(wind_speed, wind_direction)

W = interpolateVector(WX, WY)
plotVector(W)

T = interpolateScalar(temperature)
WS = interpolateScalar(wind_speed)
H = interpolateScalar(humidity)
P = interpolateScalar(pressure, 'rbf')

plotScalar(T, cm.jet)
plotScalar(WS, cm.Blues)
plotScalar(H, cm.GnBu)
plotScalar(P, cm.Purples)