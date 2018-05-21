import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Question 1 显示各城市各污染物的CHI数值

BASE_PATH = r'C:\Users\chenshuai\Documents\airquality\result'
files = os.listdir(BASE_PATH)
COLUMNS_SJZ = ['sjz_ground_wind', 'sjz_high', 'sjz_humidity', 'sjz_max_temperature',
              'sjz_max_wind', 'sjz_min_humidity',
            'sjz_min_temperature', 'sjz_pressure', 'sjz_staticstability', 'sjz_sunshine',
            'sjz_temperature', 'sjz_water', 'sjz_wind']
COLUMNS_XT = ['xt_ground_wind', 'xt_high', 'xt_humidity', 'xt_max_temperature',
              'xt_max_wind', 'xt_min_humidity',
            'xt_min_temperature', 'xt_pressure', 'xt_staticstability', 'xt_sunshine',
            'xt_temperature', 'xt_water', 'xt_wind']
sjz = []
xt = []
for file in files:
    # 获取SJZ_IG信息文件
    if 'IG' in file and 'sjz' in file:
        df = pd.read_csv(BASE_PATH+'\\'+file)
        df = df.set_index('Unnamed: 0')
        df = df[COLUMNS_SJZ]
        sjz.append(df)
    if 'IG' in file and 'xt' in file:
        df = pd.read_csv(BASE_PATH+'\\'+file)
        df = df.set_index('Unnamed: 0')
        df = df[COLUMNS_XT]
        xt.append(df)

print(len(sjz), len(xt))
x = range(1, 14)
i = 1

# 石家庄
# for index, title in zip(
#                     ['sjz_pm2', 'sjz_pm10', 'sjz_so2', 'sjz_no2', 'sjz_co2', 'sjz_o3'],
#                     ['sjz_pm2', 'sjz_pm10', 'sjz_so2', 'sjz_no2', 'sjz_co', 'sjz_o3']
#                     ):
#     plt.figure(i)
#     i += 1
#     qs = [df.loc[index] for df in sjz]
#     plt.plot(x, qs[0], label='Q1')
#     plt.plot(x, qs[1], label='Q2')
#     plt.plot(x, qs[2], label='Q3')
#     plt.plot(x, qs[3], label='Q4')
#     plt.xlabel('Attributes')
#     plt.ylabel(index.split('_')[1])
#     plt.title(title)
#     plt.legend()
#     plt.show()

# 邢台
for index, title in zip(
                    ['xt_pm2', 'xt_pm10', 'xt_so2', 'xt_no2', 'xt_co2', 'xt_o3'],
                    ['xt_pm2', 'xt_pm10', 'xt_so2', 'xt_no2', 'xt_co', 'xt_o3']
                    ):
    plt.figure(i)
    i += 1
    qs = [df.loc[index] for df in xt]
    plt.plot(x, qs[0], label='Q1')
    plt.plot(x, qs[1], label='Q2')
    plt.plot(x, qs[2], label='Q3')
    plt.plot(x, qs[3], label='Q4')
    plt.xlabel('Attributes')
    plt.ylabel(index.split('_')[1])
    plt.title(title)
    plt.legend()
    plt.show()