import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor


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


def evaluate_regression(y, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    y = y.ravel()
    y_pred = y_pred.ravel()
    # print(f'MAE:{int(mean_absolute_error(y, y_pred))}')
    # print(f'MSE:{int(mean_squared_error(y, y_pred))}')
    print(f'RMSE:{int(np.sqrt(mean_squared_error(y, y_pred)))}')
    # print('抽样随机结果对比：')
    # index = np.random.randint(1, 20)
    # result = pd.DataFrame([y[index:index+5], y_pred[index:index+5]], index=['y', 'y_pred'])
    # print(result)
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


def Q2(df):
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
    # df_q2 = df.filter(regex='sjz_*')
    # df_q2 = df_q2['2014-1-1':'2017-1-1']
    df_q2 = df['2014-1-1':'2017-1-1']
    df_q2 = df_q2.filter(regex='sjz_*')
    # df_q2.drop(columns=['sjz_water'])

    Y_pm2 = df_q2.sjz_pm2.values                   # Y 为当日浓度
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
    # X_q2 = pd.concat([df_q2, df_q2_add], axis=1)  # 加三维度
    X_q2 = df_q2                                    # 不加三个维度
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
    print('RF')
    rfr = RandomForestRegressor(random_state=0, n_estimators=50)
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_test)
    print(f'R2_train:{rfr.score(X_train, y_train):.2f}')
    print(f'R2:{rfr.score(X_test, y_test):.2f}')
    evaluate_regression(y_pred, y_test)
    # print(X_q2.columns)
    # print(rfr.feature_importances_)

    print('GBR:')
    gbr = GradientBoostingRegressor(random_state=10)
    gbr.fit(X_train, y_train)
    y_pred = gbr.predict(X_test)
    print(f'R2:{gbr.score(X_test, y_test):.2f}')


def Q2_BEST(df, col):
    # 问题三 季节分析
    X_COLS = ['sjz_staticstability', 'sjz_temperature',
              # 'sjz_high', 'sjz_ground_wind',
           'sjz_max_temperature', 'sjz_min_temperature',
           'sjz_water','sjz_pressure',
           'sjz_humidity', 'sjz_min_humidity',
           'sjz_wind', 'sjz_max_wind',
           'sjz_sunshine']
    X_COLS = ['sjz_staticstability', 'sjz_temperature',
              'sjz_high', 'sjz_ground_wind',
              'sjz_max_temperature', 'sjz_min_temperature',
              'sjz_water', 'sjz_pressure',
              'sjz_humidity', 'sjz_min_humidity',
              'sjz_wind', 'sjz_max_wind',
              'sjz_sunshine']

    # 取2014 - 2016 数据
    df = df['2014-1-1':'2017-1-1']
    df = df.filter(regex='date|sjz_*')
    Y = df[[col]].values.ravel()
    X = df[X_COLS]
    # X.to_csv(BASE_PATH+'\X.csv')

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=1)
    rfr = RandomForestRegressor(random_state=0, n_estimators=50)
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_test)
    print(f'RF:;R2_train:{rfr.score(X_train, y_train):.2f}')
    print(f'R2:{rfr.score(X_test, y_test):.2f}')
    from sklearn.ensemble import GradientBoostingRegressor
    gbr = GradientBoostingRegressor(random_state=10)
    gbr.fit(X_train, y_train)
    y_pred = gbr.predict(X_test)
    print(f'R2:{gbr.score(X_test, y_test):.2f}')
    print(X_COLS)
    print(gbr.feature_importances_)
    return gbr.feature_importances_


def searchQ2(df):
    from sklearn.model_selection import GridSearchCV
    param_grid = {
        'n_estimators': [475, 490, 510, 540],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [13,14,15,16,17],
        # 'criterion': ['gini', 'entropy']
    }
    rfr = RandomForestRegressor(random_state=0, n_jobs=-1)
    from sklearn.ensemble import GradientBoostingRegressor
    CV_rfc = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=10, n_jobs=4)

    # 数据
    X_COLS = ['sjz_staticstability', 'sjz_temperature',
              # 'sjz_high', 'sjz_ground_wind',
              'sjz_max_temperature', 'sjz_min_temperature',
              'sjz_water', 'sjz_pressure',
              'sjz_humidity', 'sjz_min_humidity',
              'sjz_wind', 'sjz_max_wind',
              'sjz_sunshine']

    # 取2014 - 2016 数据
    df = df['2014-1-1':'2017-1-1']
    df = df.filter(regex='date|sjz_*')
    Y = df.sjz_pm2
    X = df[X_COLS]

    df_q2_add = getAppendix(X)

    # X = X.iloc[1:, :]
    # df_q2_add.index = X.index
    # X = pd.concat([X, df_q2_add], axis=1)
    # Y = Y[1:]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=0)
    print('----------')
    CV_rfc.fit(X_train, y_train)
    print(CV_rfc.best_params_)
    print(f'Score:{CV_rfc.best_score_}')
    rfc1 = RandomForestRegressor(random_state=42, **CV_rfc.best_params_)
    rfc1.fit(X_train, y_train)
    y_predic = rfc1.predict(X_test)
    evaluate_regression(y_predic, y_test)
    print(f'R2:{rfc1.score(X_test, y_test):.2}')


def Q3():
    print('Question3:')
    from sklearn.model_selection import GridSearchCV
    param_grid = {
        'n_estimators': [475, 490, 510, 540],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [13, 14, 15, 16, 17],
        # 'criterion': ['gini', 'entropy']
    }
    rfr = RandomForestRegressor(random_state=0, n_jobs=-1)
    CV_rfc = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=10, n_jobs=4)

    # 数据
    X_COLS = ['sjz_staticstability', 'sjz_temperature',
              # 'sjz_high', 'sjz_ground_wind',
              'sjz_max_temperature', 'sjz_min_temperature',
              'sjz_water', 'sjz_pressure',
              'sjz_humidity', 'sjz_min_humidity',
              'sjz_wind', 'sjz_max_wind',
              'sjz_sunshine',
              'wind_add', 'temperate_add', 'humidity_add'
              ]
    df1 = pd.read_csv(BASE_PATH + r'\sjz_q1.csv')
    df1_2 = pd.read_csv(BASE_PATH + r'\xt_q1.csv')
    df1 = df1.set_index('date')
    df1_2 = df1_2.set_index('date')

    # 加三个维度
    df1_add = getAppendix(df1)
    df1_2_add = getAppendix(df1_2, type='xt')
    df1 = df1.iloc[1:, :]
    df1_add.index = df1.index
    df1 = pd.concat([df1, df1_add], axis=1)

    df1_2 = df1_2.iloc[1:, :]
    df1_2_add.index = df1_2.index
    df1_2 = pd.concat([df1_2, df1_2_add], axis=1)

    df1_2.columns = df1.columns
    df1 = pd.concat([df1, df1_2])
    Y = df1.sjz_pm2
    X = df1[X_COLS]

    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=0)
    CV_rfc.fit(X_train, y_train)
    print(CV_rfc.best_params_)
    print(f'Score:{CV_rfc.best_score_}')
    # p = {'max_depth': 13, 'max_features': 'sqrt', 'n_estimators': 490}
    rfc1 = RandomForestRegressor(random_state=42, **CV_rfc.best_params_) # **CV_rfc.best_params_
    rfc1.fit(X_train, y_train)
    y_predict = rfc1.predict(X_test)
    evaluate_regression(y_predict, y_test)
    print(f'R2:{rfc1.score(X_test, y_test):.2}')
    result = pd.DataFrame({'real': y_test, 'y_predict': y_predict})
    result.to_csv(BASE_PATH+r'\result\Question3_Q1.csv')


def Q1_IG(df, col):
    # 问题1 IG
    X_COLS = ['sjz_staticstability', 'sjz_temperature',
              'sjz_high', 'sjz_ground_wind',
              'sjz_max_temperature', 'sjz_min_temperature',
              'sjz_water', 'sjz_pressure',
              'sjz_humidity', 'sjz_min_humidity',
              'sjz_wind', 'sjz_max_wind',
              'sjz_sunshine']

    # 取2014 - 2016 数据
    df = df['2014-1-1':'2017-1-1']
    df = df.filter(regex='date|sjz_*')
    Y = df[[col]].values.ravel()
    X = df[X_COLS]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=1)
    from sklearn.ensemble import GradientBoostingRegressor
    gbr = GradientBoostingRegressor(random_state=10)
    gbr.fit(X_train, y_train)
    y_pred = gbr.predict(X_test)
    print(f'R2:{gbr.score(X_test, y_test):.2f}')
    print(X_COLS)
    print(gbr.feature_importances_)
    return gbr.feature_importances_


if __name__ == '__main__':
    BASE_PATH = r'C:\Users\chenshuai\Documents\airquality\\'
    df = pd.read_csv(BASE_PATH+r'chen2014-2017.csv')
    df.date = pd.to_datetime(df.date)
    df = df.set_index('date')

    # Q2(df)
    # Q3(df)
    # Q2_BEST(df)
    # searchQ2(df)
    # Q3()

    sjz = []
    Ys = ['sjz_pm2',  'sjz_pm10', 'sjz_so2', 'sjz_co2', 'sjz_o3']
    for y in Ys:
        sjz.append(Q2_BEST(df, y))
    sjz = np.reshape(sjz, (-1, 13))
    cols = ['sjz_staticstability', 'sjz_temperature', 'sjz_high', 'sjz_ground_wind', 'sjz_max_temperature', 'sjz_min_temperature', 'sjz_water', 'sjz_pressure', 'sjz_humidity', 'sjz_min_humidity', 'sjz_wind', 'sjz_max_wind', 'sjz_sunshine']
    shz_ig = pd.DataFrame(sjz, columns=cols, index=Ys)
    shz_ig.to_csv(BASE_PATH+r'\result\xt_IG.csv')
    # from sklearn.model_selection import cross_val_score
    # rfr = RandomForestRegressor(random_state=0, n_estimators=50)
    # scores = cross_val_score(rfr, X_q2, Y_pm2, cv=3, scoring='r2')            #5-fold cv
    # print(scores.mean())
