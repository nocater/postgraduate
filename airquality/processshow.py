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
for f in files:
    path = BATH_PATH+'\\'+f
    df = pd.read_csv(path)
    df = df.drop(columns=['Unnamed: 0'])
    df.columns = COLS
    df = df.set_index('y')
    # df.info()
    if 'sjz' in f:
        pd_sjz.append(df)
        pass


pm2 = [i.loc['pm2', :] for i in pd_sjz]
pm2 = np.array(pm2)
# pm2.reshape(-1, 13)
# # 获取到了一个城市的四个季度的pm2值
# pm2 = np.mean(pm2, axis=0)
# print(np.shape(pm2))

plt.plot([i for i in range(13)], pm2[0], 'r--', label='Q1')
plt.plot([i for i in range(13)], pm2[1], 'g--', label='Q2')
plt.plot([i for i in range(13)], pm2[2], 'b--', label='Q3')
plt.plot([i for i in range(13)], pm2[3], 'p--', label='Q4')
plt.title('SJZ_PM2.5_MU')
plt.legend()
plt.xticks(np.arange(0, 13), tuple(COLS)[1:], rotation=60)
plt.xlabel('Attribute')
plt.ylabel('MU')
plt.show()