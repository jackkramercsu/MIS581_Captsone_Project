#packages and libraries
import requests
from datetime import datetime
from datetime import date
import time
from datetime import timedelta
import numpy as np
import pandas as pd 


#API Key
w_api_key = 'ADD API KEY'
key1 = w_api_key

#convert yesterdays date to correct format can only look at yesterdays data for full data available makes for a bit of a prediction lag
date = date.today()
date = date - timedelta(days=1)
date = date.strftime('%Y-%m-%d')


#set j 10 1 to make while loop work
j = 1

#while loop to start extraction process daily may be able to use free api call here
while j == 1:
    data = requests.get('https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/weatherdata/history?aggregateHours=24&combinationMethod=aggregate&startDateTime={}T00%3A00%3A00&endDateTime={}T00%3A00%3A00&dayEndTime=23%3A0%3A0&includeAstronomy=true&maxStations=-1&maxDistance=-1&contentType=json&unitGroup=us&locationMode=single&key={}&dataElements=default&locations=Washington%20State%2C%20United%20States'.format(date, date, key1))
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
    with open('ADD FILE PATH', 'a') as f:
        wash.to_csv(f, mode='a', index=False, header=False)
    time.sleep(86405)