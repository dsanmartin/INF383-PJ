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

def interpolateVector(X, Y, kind='linear'):
  WX = interpolateScalar(X, kind)
  WY = interpolateScalar(Y, kind)
  
  return WX, WY

  
def plotScalar(data, title, cmap_):
  #plt.contour(temperature, cmap=cm.jet)
  #plt.imshow(temperature, cmap=cm.jet)
  pc = plt.pcolor(data, cmap=cmap_)
  plt.title(title)
  plt.colorbar(pc)
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

#%%

WX, WY = createWind(wind_speed, wind_direction)

T = interpolateScalar(temperature)
WS = interpolateScalar(wind_speed)
WD = interpolateScalar(wind_direction)
H = interpolateScalar(humidity)
P = interpolateScalar(pressure, 'rbf')
U, V = interpolateVector(WX, WY, 'rbf')

#np.save('data/temperature100x100.npy', T)
#np.save('data/wind_speed100x100.npy', WS)
#np.save('data/wind_direction100x100.npy', WD)
#np.save('data/humidity100x100.npy', H)
#np.save('data/pressure100x100.npy', P)

plotScalar(T, "Temperature", cm.jet)
plotScalar(WS, "Wind speed", cm.Blues)
plotScalar(H, "Humidity", cm.GnBu)
plotScalar(P, "Pressure", cm.Purples)
plotVector(U[::5, ::5], V[::5, ::5])