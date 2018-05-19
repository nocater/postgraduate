import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.ensemble import RandomForestRegressor


BASE_PATH = r'C:\Users\chenshuai\Documents\airquality\\'
df = pd.read_csv(BASE_PATH+r'chen2014-2017.csv')
df.date = pd.to_datetime(df.date)
df = df.set_index('date')

# 问题二 逐日浓度数据
def compute(l):
    """
    计算差值数据
    :param l:
    :return:
    """
    r = []
    for i in range(1, len(l)):
        r.append(l[i]-l[i-1])
    return r


def compute_humidity(l):
    """
    计算差值数据
    :param l:
    :return:
    """
    r = []
    for i in range(1, len(l)):
        if(l[i]>=48 and l[i-1]>=48):
            r.append(1)
        else:
            r.append(0)
    return r


def compute_temperature(l):
    r = []
    for i in range(1, len(l)):
        if (l[i] >= 20 and l[i - 1] >= 20):
            r.append(1)
        else:
            r.append(0)
    return r
    pass


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
# 取石家庄逐日数据
df_q2 = df.filter(regex='sjz_*')
df_q2 = df_q2['2014-1-1':'2017-1-1']
df_q2.drop(columns=['sjz_water'])

Y_pm2 = df_q2.sjz_co2.values                   # Y 为当日浓度
Y_pm2 = Y_pm2[1:]
# Y_pm2 = compute(df_q2.sjz_co2.values) # Y 为逐日差值类型
# 风速增加
wind_add = compute(df_q2.sjz_wind)
# 温度增加
# temperate_add = compute(df_q2.sjz_temperature)            # 直接差值法
temperate_add = compute_temperature(df_q2.sjz_temperature)  # 稳定法
# 湿度处理
humidity_add = compute_humidity(df_q2.sjz_humidity)
# print(np.max(wind_add))
df_q2_add = pd.DataFrame({
                          'wind_add': wind_add,
                          'temperate_add': temperate_add,
                          'humidity_add': humidity_add
                          })
# df_q2_add.info()

df_q2 = df_q2.iloc[1:, :]
df_q2 = df_q2[X_SJZ]
df_q2_add.index = df_q2.index
X_q2 = pd.concat([df_q2, df_q2_add], axis=1)  # 加三维度
# X_q2 = df_q2                                    # 不加三个维度
# X_q2 = df_q2_add                              # 只是用add三个维度
# X_q2.info()

if False:
    pass
    # # 邢台
    # df_q2 = df.filter(regex='xt_*')
    # df_q2 = df_q2['2014-1-1':'2017-1-1']
    #
    # Y_pm2_xt = compute(df_q2.xt_co2.values)
    # # 风速增加
    # wind_add = compute(df_q2.xt_wind)
    # # 温度增加
    # temperate_add = compute(df_q2.xt_temperature)
    # humidity_add = compute_humidity(df_q2.xt_humidity)
    # df_q2_add = pd.DataFrame({'wind_add': wind_add, 'temperate_add': temperate_add, 'humidity_add': humidity_add})
    # # df_q2_add.info()
    #
    # df_q2 = df_q2.iloc[1:, :]
    # df_q2 = df_q2[X_XT]
    # df_q2_add.index = df_q2.index
    # X_q2_xt = pd.concat([df_q2, df_q2_add], axis=1)  # 加三维度
    # # X_q2 = df_q2                                   # 不加三个维度
    # # X_q2.info()
    #
    # X_q2 = np.append(X_q2.as_matrix(), X_q2_xt.as_matrix())
    # X_q2 = X_q2.reshape(-1, 16)
    # Y_pm2 = np.append(Y_pm2, Y_pm2_xt)
    # Y_pm2 = Y_pm2.ravel()

X_train, X_test, y_train, y_test = train_test_split(X_q2, Y_pm2, test_size=0.2, shuffle=True, random_state=1)
# from sklearn.ensemble import GradientBoostingRegressor
# gbr = GradientBoostingRegressor(random_state=0)
# gbr.fit(X_train, y_train)
# y_pred = gbr.predict(X_test)
# print(f'R2_train:{gbr.score(X_train, y_train):.2f}')
# print(f'R2:{gbr.score(X_test, y_test):.2f}')
# print(gbr.feature_importances_)
#
print()
rfr = RandomForestRegressor(random_state=0, n_estimators=50)
rfr.fit(X_train, y_train)
y_pred = rfr.predict(X_test)
print(f'R2_train:{rfr.score(X_train, y_train):.2f}')
print(f'R2:{rfr.score(X_test, y_test):.2f}')


# from sklearn.model_selection import cross_val_score
# rfr = RandomForestRegressor(random_state=0, n_estimators=50)
# scores = cross_val_score(rfr, X_q2, Y_pm2, cv=3, scoring='r2')            #5-fold cv
# print(scores.mean())
# 问题三 季节分析
# X_COLS = ['sjz_staticstability', 'sjz_temperature',
#           # 'sjz_high', 'sjz_ground_wind',
#        'sjz_max_temperature', 'sjz_min_temperature',
#        'sjz_water','sjz_pressure',
#        'sjz_humidity', 'sjz_min_humidity',
#        'sjz_wind', 'sjz_max_wind',
#        'sjz_sunshine']
#
# # 取2014 - 2016 数据
# df = df['2014-1-1':'2017-1-1']
# df = df.filter(regex='date|sjz_*')
# Y = df.sjz_pm2
# X = df[X_COLS]
# # X.to_csv(BASE_PATH+'\X.csv')
#
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=1)
# from sklearn.ensemble import GradientBoostingRegressor
# gbr = GradientBoostingRegressor(random_state=10)
# gbr.fit(X_train, y_train)
# y_pred = gbr.predict(X_test)
# print(f'R2:{gbr.score(X_test, y_test):.2f}')
# print(gbr.feature_importances_)