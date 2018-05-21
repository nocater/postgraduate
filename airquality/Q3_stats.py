import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


BASE_PATH = r'C:\Users\chenshuai\Documents\airquality\\'
df = pd.read_csv(BASE_PATH+r'chen2014-2017.csv')
df.date = pd.to_datetime(df.date)
df = df.set_index('date')

# Y_sjz = df[['']]
# 按月统计
# 按季度
df_q_mean = df.resample('Q').mean()
df_q_sum = df.resample('Q').sum()
print(df_q_mean.head())

# 石家庄的pm2.5
sjz = df_q_mean.sjz_pm2.values
sjz = sjz[:-1]
sjz = sjz.reshape(-1, 4)
# sjz = sjz.T
names = ['Spring', 'Summer', 'August', 'Winter']
# names = ['2014', '2015', '2016']
x = range(len(names))
plt.plot(x, sjz[0], '*-', label='2014')
plt.plot(x, sjz[1], 'v-', label='2015')
plt.plot(x, sjz[2], 'x-', label='2016')
# plt.plot(x, sjz[0], '*-', label='Spring')
# plt.plot(x, sjz[1], '+-', label='Summer')
# plt.plot(x, sjz[2], 'x-', label='August')
# plt.plot(x, sjz[3], 'v-', label='August')
plt.xticks(x, names)
plt.ylabel('PM2.5')
plt.title('SJZ_Mean_PM2.5(2014-2016)')
plt.legend()
plt.show()

# 邢台的pm2.5
xt = df_q_mean.xt_pm2.values
xt = xt[:-1]
xt = xt.reshape(-1, 4)
names = ['Spring', 'Summer', 'August', 'Winter']
x = range(len(names))
plt.plot(x, xt[0], '*-', label='2014')
plt.plot(x, xt[1], 'v-', label='2015')
plt.plot(x, xt[2], 'x-', label='2016')
plt.xticks(x, names)
plt.ylabel('PM2.5')
plt.title('XT_Mean_PM2.5(2014-2016)')
plt.legend()
plt.show()


df_q_sum = df.resample('Q').sum()
# 石家庄的pm2.5
sjz = df_q_mean.sjz_co2.values
sjz = sjz[:-1]
sjz = sjz.reshape(-1, 4)
names = ['Spring', 'Summer', 'Autumn', 'Winter']
x = range(len(names))
plt.plot(x, sjz[0], '*-', label='2014')
plt.plot(x, sjz[1], 'v-', label='2015')
plt.plot(x, sjz[2], 'x-', label='2016')
plt.xticks(x, names)
plt.ylabel('PM2.5')
plt.title('SJZ_Mean_CO(2014-2016)')
plt.legend()
plt.show()

# 邢台的pm2.5
xt = df_q_mean.xt_co2.values
xt = xt[:-1]
xt = xt.reshape(-1, 4)
names = ['Spring', 'Summer', 'Autumn', 'Winter']
x = range(len(names))
plt.plot(x, xt[0], '*-', label='2014')
plt.plot(x, xt[1], 'v-', label='2015')
plt.plot(x, xt[2], 'x-', label='2016')
plt.xticks(x, names)
plt.ylabel('PM2.5')
plt.title('XT_Mean_CO(2014-2016)')
plt.legend()
plt.show()


# plt.figure(1)
# for col in df.columns[:1]:
#     y = np.array(df[[col]].values)
#     x = range(len(y))
#     plt.plot(x, y, label=col)
# plt.legend()
# plt.show()