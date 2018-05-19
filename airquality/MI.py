import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import numpy as np

BASE_PATH = r'C:\Users\chenshuai\Documents\airquality\\'

sjz_files = ['sjz_q1.csv','sjz_q2.csv','sjz_q3.csv','sjz_q4.csv']
xt_files =  ['xt_q1.csv','xt_q2.csv','xt_q3.csv','xt_q4.csv']
X_SJZ = ['sjz_staticstability', 'sjz_temperature',
          'sjz_high', 'sjz_ground_wind',
       'sjz_max_temperature', 'sjz_min_temperature',
       'sjz_water','sjz_pressure',
       'sjz_humidity', 'sjz_min_humidity',
       'sjz_wind', 'sjz_max_wind',
       'sjz_sunshine']
X_XT = ['xt_staticstability', 'xt_temperature',
          'xt_high', 'xt_ground_wind',
       'xt_max_temperature', 'xt_min_temperature',
       'xt_water','xt_pressure',
       'xt_humidity', 'xt_min_humidity',
       'xt_wind', 'xt_max_wind',
       'xt_sunshine']

# 石家庄
for file in sjz_files:
    df = pd.read_csv(BASE_PATH+file)
    df.date = pd.to_datetime(df.date)
    df = df.set_index('date')
    df = df['2014-1-1':'2017-1-1']
    X = df[X_SJZ]
    Y = [df.sjz_pm2, df.sjz_pm10, df.sjz_so2, df.sjz_no2, df.sjz_co2, df.sjz_o3]
    result = []
    for y in Y:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)
        dtr = DecisionTreeRegressor(random_state=0)
        dtr.fit(X_train, y_train)
        print(dtr.score(X_test, y_test))
        result.append(dtr.feature_importances_)
        pass
    result = pd.DataFrame(np.reshape(result, (-1, 13)), index=['pm2', 'pm10', 'so2', 'no2', 'co2', 'o3'], columns=X_SJZ)
    print(result.T)
    result.to_csv(BASE_PATH+'IG_sjz'+str(file.split('_')[1][:2])+'.csv')


for file in xt_files:
    df = pd.read_csv(BASE_PATH+file)
    df.date = pd.to_datetime(df.date)
    df = df.set_index('date')
    df = df['2014-1-1':'2017-1-1']
    X = df[X_XT]
    Y = [df.xt_pm2, df.xt_pm10, df.xt_so2, df.xt_no2, df.xt_co2, df.xt_o3]
    result = []
    for y in Y:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)
        dtr = DecisionTreeRegressor(random_state=0)
        dtr.fit(X_train, y_train)
        print(dtr.score(X_test, y_test))
        result.append(dtr.feature_importances_)
        pass
    result = pd.DataFrame(np.reshape(result, (-1, 13)), index=['pm2', 'pm10', 'so2', 'no2', 'co2', 'o3'], columns=X_XT)
    print(result.T)
    result.to_csv(BASE_PATH+'IG_xt'+str(file.split('_')[1][:2])+'.csv')