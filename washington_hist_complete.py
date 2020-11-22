#packages and libraries
import requests
from datetime import datetime
import time
import numpy as np
import pandas as pd


#visual crossing api key
w_api_key = 'ADD API KEY'
key1 = w_api_key

#list ofr extraction

list = [['1970-01-01', '1979-12-31'], ['1980-01-01', '1989-12-31'], ['1990-01-01', '1999-12-31'], ['2000-01-01', '2020-10-31']]

for j in list:
    data = requests.get('https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/weatherdata/history?aggregateHours=24&combinationMethod=aggregate&startDateTime={}T00%3A00%3A00&endDateTime={}T00%3A00%3A00&dayEndTime=23%3A0%3A0&includeAstronomy=true&maxStations=-1&maxDistance=-1&contentType=json&unitGroup=us&locationMode=single&key={}&dataElements=default&locations=Washington%20State%2C%20United%20States'.format(j[0], j[1], key1))
    data = data.json()
    wash_lat = data['location']['latitude']
    wash_long = data['location']['longitude']
    wash = np.array(data['location']['values'])
    state = 'WA'
    dfw = pd.DataFrame(columns=['sunrise', 'temp', 'maxt', 'visibility', 'wspd', 'heatindex', 'cloudcover', 'mint', 'datetime', 'precip', 'weathertype', 'moonphase', 'snowdepth', 'sunset', 'humidity', 'wgust', 'conditions', 'windchill', 'state', 'lat', 'long'])
    for p in wash:
        wash_row = [str(p['sunrise']), str(p['temp']), str(p['maxt']), str(p['visibility']), str(p['wspd']), 
                    str(p['heatindex']), str(p['cloudcover']), str(p['mint']), str(p['datetime']), str(p['precip']), 
                    str(p['weathertype']), str(p['moonphase']), str(p['snowdepth']), str(p['sunset']), str(p['humidity']), 
                    str(p['wgust']), str(p['conditions']), str(p['windchill']), state, wash_lat, wash_long] 
        dfw.loc[-1,:]=wash_row
        dfw.index = dfw.index+1
    #sor the data by sunrise date
    wash = dfw.sort_values('sunrise')  
    #replace null values to 0 
    wash = wash.replace(to_replace='None', value=0)    
    #converting sunrise and sunset to date time values
    wash['sunrise'] = pd.to_datetime(wash['sunrise'], format='%Y-%m-%d %H:%M:%S')
    wash['sunset'] = pd.to_datetime(wash['sunset'], format='%Y-%m-%d %H:%M:%S')
    time.sleep(10)
    with open('ADD FILE PATH FOR CSV EXPORT ', 'a') as f:
        wash.to_csv(f, mode='a', index=False, header=False)

