import firemodels.cellularautomata as ca
import numpy as np
import matplotlib.pyplot as plt
import requests

#def fireFocus(M, N, i, j, size):
#  focus = np.zeros((M, N))
#  focus[i-size:i+size, j-size:j+size] = np.ones((size, size)) 
#  return focus
#  
## Testing
#M = 101
#N = 101
#initial = fireFocus(M, N, 50, 50, 1)
#rule = 1
#neighborhood = 'vonneumann'#'moore'
#times = 10
#
#ca = ca.new(M, N, initial, rule, neighborhood)
#
#sta = ca.propagate(times)
base = "http://api.wunderground.com/api/35c89267da3bc202/conditions/q/"
#base = "http://api.openweathermap.org/data/2.5/weather?"
#lat=-38.7290843&lon=-72.6377406
#end = "&units=metric&appid=bfba84a8e1deeb1bd7a3f034aaba4009"

def getWeather(start_lat, start_lng, end_lat, end_lng, M, N):
  delta_lat = (end_lat - start_lat) / M
  delta_lng = (end_lng - start_lng) / N
  
  temp = np.zeros((M, N))
  
  for i in range(M):
    for j in range(N):
      lat = start_lat + i * delta_lat
      lng = start_lng + j * delta_lng
      
      #url = base + "lat=" + str(lat) + "&lon=" + str(lng) + end #+ ".json"      
      url = base + str(lat) + "," + str(lng) + ".json" 
      #data = requests.get(url).json()['main']
      data = requests.get(url).json()['current_observation']

      #d = float(data['temp'])
      d = float(data['temp_c'])
      #print(d)
      

      temp[i, j] = float(d)
      
  return temp

M, N = 100, 100
temperatura = getWeather(-35.3803, -72.3652, -35.5235, -72.1894, M, N)

plt.imshow(temperatura)
plt.show()
#print(data['temp_c'])

# Resampling image
#subimg = np.zeros((SO_SIZE, SO_SIZE))
#for i in range(SO_SIZE):
#    for j in range(SO_SIZE):
#        subimg[i, j] = img[5*i+3, 5*j+3]
