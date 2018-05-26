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

index_sjz = ['sjz_co2', 'sjz_no2', 'sjz_o3', 'sjz_pm2', 'sjz_pm10', 'sjz_so2']
index_xt = ['sjz_co2', 'sjz_no2', 'sjz_o3', 'sjz_pm2', 'sjz_pm10', 'sjz_so2']

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
#                     ['PM2.5', 'PM10', 'So2', 'No2', 'CO', 'O3']
#                     ):
#     plt.figure(i)
#     i += 1
#     qs = [df.loc[index] for df in sjz]
#     plt.plot(x, qs[0], label='sjzSpring')
#     plt.plot(x, qs[1], label='sjzSummer')
#     plt.plot(x, qs[2], label='sjzAutumn')
#     plt.plot(x, qs[3], label='sjz_Winter')
#     plt.xlabel('the ith attributes of all factors')
#     plt.ylabel('The coefficient of Information Gain')
#     plt.title(title)
#     plt.legend()
#     plt.show()

# 邢台
# for index, title in zip(
#                     ['xt_pm2', 'xt_pm10', 'xt_so2', 'xt_no2', 'xt_co2', 'xt_o3'],
#                     ['PM2.5', 'PM10', 'So2', 'No2', 'CO', 'O3']
#                     ):
#     plt.figure(i)
#     i += 1
#     qs = [df.loc[index] for df in xt]
#     # 画四条线
#     plt.plot(x, qs[0], 'o--', label='xtSpring')
#     plt.plot(x, qs[1], 'v-', label='xtSummer')
#     plt.plot(x, qs[2], '*-.', label='xtAutumn')
#     plt.plot(x, qs[3], 'x:',  label='xtWinter')
#     # X 轴坐标
#     plt.xlabel('the ith attributes of all factors')
#     # Y 轴坐标
#     plt.ylabel('The coefficient of Information Gain')
#     # 标题
#     plt.title(title)
#     # 显示图例
#     plt.legend()
#     # 显示图片
#     plt.show()