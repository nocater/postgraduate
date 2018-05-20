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
df_q = df.resample('Q').mean()
print(df_q.head())

# 石家庄的pm2.5
sjz = df_q.sjz_pm2.values
sjz = sjz[:-1]
sjz = sjz.reshape(-1, 4)
names = ['Spring', 'Summer', 'August', 'Winter']
x = range(len(names))
plt.plot(x, sjz[0], '*-', label='2014')
plt.plot(x, sjz[1], '+-', label='2015')
plt.plot(x, sjz[2], 'x-', label='2016')
plt.xticks(x, names)
plt.ylabel('Pm2.5')
plt.title('SJZ_Mean_Pm2.5(2014-2016)')
plt.legend()
plt.show()

# 邢台的pm2.5
xt = df_q.xt_pm2.values
xt = xt[:-1]
xt = xt.reshape(-1, 4)
names = ['Spring', 'Summer', 'August', 'Winter']
x = range(len(names))
plt.plot(x, xt[0], '*-', label='2014')
plt.plot(x, xt[1], '+-', label='2015')
plt.plot(x, xt[2], 'x-', label='2016')
plt.xticks(x, names)
plt.ylabel('Pm2.5')
plt.title('XT_Mean_Pm2.5(2014-2016)')
plt.legend()
plt.show()


# plt.figure(1)
# for col in df.columns[:1]:
#     y = np.array(df[[col]].values)
#     x = range(len(y))
#     plt.plot(x, y, label=col)
# plt.legend()
# plt.show()