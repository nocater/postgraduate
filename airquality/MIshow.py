import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def showFig():
    pass

BATH_PATH = r'C:\Users\chenshuai\Documents\airquality\mu'
files = os.listdir(BATH_PATH)
# print(files)

COLS = ['y','ground_wind', 'high', 'humidity', 'max_temperature',
                                 'max_wind', 'min_humidity',
                                 'min_temperature', 'pressure', 'staticstability', 'sunshine',
                                 'temperature', 'water', 'wind']

pd_sjz = []
pd_xt = []
for f in files:
    path = BATH_PATH+'\\'+f
    df = pd.read_csv(path)
    df = df.drop(columns=['Unnamed: 0'])
    df.columns = COLS
    df = df.set_index('y')
    if 'sjz' in f:
        pd_sjz.append(df)
    if 'xt' in f:
        pd_xt.append(df)

x = range(1, 14)
i = 1

# 石家庄
for index, title in zip(
                    ['pm2', 'pm10', 'so2', 'no2', 'co2', 'o3'],
                    ['PM2.5', 'PM10', 'So2', 'No2', 'CO', 'O3']
                    ):
    plt.figure(i)
    i += 1
    qs = [df.loc[index] for df in pd_xt]
    # 画四条线
    plt.plot(x, qs[0], 'o--', label='sjzSpring')
    plt.plot(x, qs[1], 'v-', label='sjzSummer')
    plt.plot(x, qs[2], '*-.', label='sjzAutumn')
    plt.plot(x, qs[3], 'x:',  label='sjzWinter')
    # X 轴坐标
    plt.xlabel('the ith attributes of all factors')
    # Y 轴坐标
    plt.ylabel('The coefficient of Information Gain')
    # 标题
    plt.title(title)
    # 显示图例
    plt.legend()
    # 显示图片
    plt.show()

# 邢台
for index, title in zip(
                    ['pm2', 'pm10', 'so2', 'no2', 'co2', 'o3'],
                    ['PM2.5', 'PM10', 'So2', 'No2', 'CO', 'O3']
                    ):
    plt.figure(i)
    i += 1
    qs = [df.loc[index] for df in pd_xt]
    # 画四条线
    plt.plot(x, qs[0], 'o--', label='xtSpring')
    plt.plot(x, qs[1], 'v-', label='xtSummer')
    plt.plot(x, qs[2], '*-.', label='xtAutumn')
    plt.plot(x, qs[3], 'x:',  label='xtWinter')
    # X 轴坐标
    plt.xlabel('the ith attributes of all factors')
    # Y 轴坐标
    plt.ylabel('The coefficient of Information Gain')
    # 标题
    plt.title(title)
    # 显示图例
    plt.legend()
    # 显示图片
    plt.show()