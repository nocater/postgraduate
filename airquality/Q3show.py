import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


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


def getAppendix(df_origin, type='sjz'):
    """
    添加额外三个维度
    :param df_origin:
    :return:
    """
    if type == 'sjz':
        # 风速增加
        wind_add = compute(df_origin.sjz_wind)
        # 温度增加
        # temperate_add = compute(df_q2.sjz_temperature)            # 直接差值法
        temperate_add = compute_temperature(df_origin.sjz_temperature)  # 稳定法
        # 湿度处理
        humidity_add = compute_humidity(df_origin.sjz_humidity)
        # print(np.max(wind_add))
        df_appendix = pd.DataFrame({
            'wind_add': wind_add,
            'temperate_add': temperate_add,
            'humidity_add': humidity_add
        })
        return df_appendix
    else:
        # 风速增加
        wind_add = compute(df_origin.xt_wind)
        # 温度增加
        # temperate_add = compute(df_q2.xt_temperature)            # 直接差值法
        temperate_add = compute_temperature(df_origin.xt_temperature)  # 稳定法
        # 湿度处理
        humidity_add = compute_humidity(df_origin.xt_humidity)
        # print(np.max(wind_add))
        df_appendix = pd.DataFrame({
            'wind_add': wind_add,
            'temperate_add': temperate_add,
            'humidity_add': humidity_add
        })
        return df_appendix


def getQ3Data(file_sjz, file_xt):
    BASE_PATH = r'C:\Users\chenshuai\Documents\airquality\\'
    X_COLS = ['sjz_staticstability', 'sjz_temperature',
              # 'sjz_high', 'sjz_ground_wind',
              'sjz_max_temperature', 'sjz_min_temperature',
              'sjz_water', 'sjz_pressure',
              'sjz_humidity', 'sjz_min_humidity',
              'sjz_wind', 'sjz_max_wind',
              'sjz_sunshine',
              # 'wind_add', 'temperate_add', 'humidity_add'
              ]
    df_sjz = pd.read_csv(BASE_PATH + r'\\'+file_sjz)
    df_xt = pd.read_csv(BASE_PATH + r'\\'+file_xt)
    df_sjz = df_sjz.set_index('date')
    df_xt = df_xt.set_index('date')

    # 加三个维度
    # df_sjz_add = getAppendix(df_sjz)
    # df_xt_add = getAppendix(df_xt, type='xt')
    #
    # df_sjz = df_sjz.iloc[1:, :]
    # df_sjz_add.index = df_sjz.index
    # df_sjz = pd.concat([df_sjz, df_sjz_add], axis=1)
    #
    # df_xt = df_xt.iloc[1:, :]
    # df_xt_add.index = df_xt.index
    # df_xt = pd.concat([df_xt, df_xt_add], axis=1)


    df_xt.columns = df_sjz.columns
    df = pd.concat([df_sjz, df_xt])
    Y = df.sjz_pm2
    X = df[X_COLS]
    print('均值：', np.mean(Y), '方差：', np.sqrt(np.var(Y)))
    return (X, Y)


def plotQ3(data, title, paprams, seed):
    (X, Y) = data
          # RandomForestRegressor(random_state=42, **CV_rfc.best_params_)
    rfg = RandomForestRegressor(random_state=42,**paprams)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=seed)
    print(X.shape)
    rfg.fit(X_train, y_train)
    y_pred = rfg.predict(X_test)
    print(f'R2_train:{rfg.score(X_train, y_train):.2}')
    print(f'R2_test:{rfg.score(X_test, y_test):.2}')
    print(f'MAE:{int(mean_absolute_error(y_test, y_pred))}')
    print(f'MAE:{int(mean_squared_error(y_test, y_pred))}')
    print(f'RMSE:{int(np.sqrt(mean_squared_error(y_test, y_pred)))}')

    # indices = np.argsort(y_test)
    # y_test = y_test[indices]
    # y_pred = y_pred[indices]
    # 画图
    x = range(1, X_test.shape[0]+1)
    plt.plot(x, y_pred, '-', label='prediction')
    plt.plot(x, y_test, '-.', label='real')
    plt.xlabel('test')
    plt.ylabel('PM2.5')
    plt.title(title)
    plt.legend()
    plt.show()
    pass


if __name__ == '__main__':
    params_Q1 = {'max_depth': 15, 'max_features': 'sqrt', 'n_estimators': 540}
    params_Q2 = {'max_depth': 16, 'max_features': 'sqrt', 'n_estimators': 540}
    params_Q3 = {'max_depth': 14, 'max_features': 'sqrt', 'n_estimators': 475}
    params_Q4 = {'max_depth': 13, 'max_features': 'sqrt', 'n_estimators': 490}
    data = []
    for i in ['q1', 'q2', 'q3', 'q4']:
        data.append(getQ3Data('sjz_'+i+'.csv', 'xt_'+i+'.csv'))
    titles = ['Spring_RFModel', 'Summer_RFModel', 'Autumn_RFModel', 'Winter_RFModel']
    plotQ3(data[0], titles[0], params_Q1, 13, )   # 0.93 0.63 63
    # print(data[1][0].iloc[:3, :3])
    plotQ3(data[1], titles[1], params_Q2, 13)   # 0.92 0.42 27
    plotQ3(data[2], titles[2], params_Q3, 13)   # 0.91 0.43 28
    plotQ3(data[3], titles[3], params_Q4, 71)   # 0.94 0.60 67