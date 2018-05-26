from sklearn.feature_selection import chi2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics as mr
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def evaluate_regression(y, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    y = y.ravel()
    y_pred = y_pred.ravel()
    print(f'MAE:{int(mean_absolute_error(y, y_pred))}')
    print(f'MSE:{int(mean_squared_error(y, y_pred))}')
    print(f'RMSE:{int(np.sqrt(mean_squared_error(y, y_pred)))}')
    # print('抽样随机结果对比：')
    # index = np.random.randint(1, 100)
    # result = pd.DataFrame([y[index:index+5], y_pred[index:index+5]], index=['y', 'y_pred'])
    # print(result)
    pass


BASE_PATH = r'C:\Users\chenshuai\Documents\airquality'


def Q1_IG(df, col):
    # 问题1 IG
    X_COLS = ['sjz_staticstability', 'sjz_temperature',
              'sjz_high', 'sjz_ground_wind',
              'sjz_max_temperature', 'sjz_min_temperature',
              'sjz_water', 'sjz_pressure',
              'sjz_humidity', 'sjz_min_humidity',
              'sjz_wind', 'sjz_max_wind',
              'sjz_sunshine']
    COLUMNS_SJZ = ['sjz_ground_wind', 'sjz_high', 'sjz_humidity', 'sjz_max_temperature',
                   'sjz_max_wind', 'sjz_min_humidity',
                   'sjz_min_temperature', 'sjz_pressure', 'sjz_staticstability', 'sjz_sunshine',
                   'sjz_temperature', 'sjz_water', 'sjz_wind']
    # 取2014 - 2016 数据
    df.date = pd.to_datetime(df.date)
    df = df.set_index('date')
    df = df['2014-1-1':'2017-1-1']
    df = df.filter(regex='date|sjz_*')
    Y = df[[col]].values.ravel()
    X = df[COLUMNS_SJZ]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=1)
    from sklearn.ensemble import GradientBoostingRegressor
    gbr = GradientBoostingRegressor(random_state=10)
    gbr.fit(X_train, y_train)
    y_pred = gbr.predict(X_test)
    print(f'R2:{gbr.score(X_test, y_test):.2f}')
    print(X_COLS)
    print(gbr.feature_importances_)
    return gbr.feature_importances_


def Q1_IG_XT(df, col):
    # 问题1 IG
    X_COLS = ['xt_staticstability', 'xt_temperature',
              'xt_high', 'xt_ground_wind',
              'xt_max_temperature', 'xt_min_temperature',
              'xt_water', 'xt_pressure',
              'xt_humidity', 'xt_min_humidity',
              'xt_wind', 'xt_max_wind',
              'xt_sunshine']

    COLUMNS_XT = ['xt_ground_wind', 'xt_high', 'xt_humidity', 'xt_max_temperature',
                  'xt_max_wind', 'xt_min_humidity',
                  'xt_min_temperature', 'xt_pressure', 'xt_staticstability', 'xt_sunshine',
                  'xt_temperature', 'xt_water', 'xt_wind']
    # 取2014 - 2016 数据
    df.date = pd.to_datetime(df.date)
    df = df.set_index('date')
    df = df['2014-1-1':'2017-1-1']
    df = df.filter(regex='date|xt_*')
    Y = df[[col]].values.ravel()
    X = df[COLUMNS_XT]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=1)
    from sklearn.ensemble import GradientBoostingRegressor
    gbr = GradientBoostingRegressor(random_state=10)
    gbr.fit(X_train, y_train)
    y_pred = gbr.predict(X_test)
    print(f'R2:{gbr.score(X_test, y_test):.2f}')
    print(X_COLS)
    print(gbr.feature_importances_)
    return gbr.feature_importances_


def generateQ():
    """
    生成季度数据表格
    :return:
    """
    df = pd.read_csv(BASE_PATH+r'\chen2014-2017.csv')
    df.date = pd.to_datetime(df.date)
    first= None

    for city in ['xt_', 'sjz_']:
        datas = df.filter(regex='date|'+city+'*')
        datas = datas.set_index('date')

        # 第一季度
        first = datas['2014-1-1':'2014-3-31']
        for year in ['2015', '2016', '2017']:
            tmp = datas[year + '-1-1':year + '-3-31']
            first = pd.concat([first, tmp])
        first.to_csv(BASE_PATH + r'\\'+city+'q1.csv')

        # 第二季度
        second = datas['2014-4-1':'2014-6-30']
        for year in ['2015', '2016']:
            tmp = datas[year + '-4-1':year + '-6-30']
            second = pd.concat([second, tmp])
        second.to_csv(BASE_PATH + r'\\' + city + 'q2.csv')

        # 第三季度
        third = datas['2014-7-1':'2014-9-30']
        for year in ['2015', '2016']:
            tmp = datas[year + '-7-1':year + '-9-30']
            third = pd.concat([third, tmp])
        third.to_csv(BASE_PATH + r'\\' + city + 'q3.csv')

        # 第四季度
        fourth = datas['2014-10-1':'2014-12-31']
        for year in ['2015', '2016']:
            tmp = datas[year + '-10-1':year + '-12-31']
            fourth = pd.concat([fourth, tmp])
        fourth.to_csv(BASE_PATH + r'\\' + city + 'q4.csv')

        print(first.shape, second.shape, third.shape, fourth.shape)

# generateQ()
first = pd.read_csv(BASE_PATH+r'\sjz_q1.csv')
first.date = pd.to_datetime(first.date)
first = first.set_index('date')

#
print(len(first[first.sjz_wind==0]))


first = first['2014-1-1':'2017-1-1']
Y = first.sjz_no2
X = first.loc[:, 'sjz_staticstability':]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=0)

dtr = DecisionTreeRegressor(random_state=3, max_features='sqrt', criterion='mse')
dtr.fit(X_train, y_train)
y_pred = dtr.predict(X_test)
evaluate_regression(y_test, y_pred)
print(f'R2:{dtr.score(X_test, y_test):.2f}')
print(dtr.feature_importances_)

from sklearn.linear_model import LinearRegression
print()
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
evaluate_regression(y_test, y_pred)
print(f'R2:{lr.score(X_test, y_test):.2f}')


from sklearn.ensemble import GradientBoostingRegressor
print()
gbr = GradientBoostingRegressor(random_state=10)
gbr.fit(X_train, y_train)
y_pred = gbr.predict(X_test)
evaluate_regression(y_test, y_pred)
print(f'R2:{gbr.score(X_test, y_test):.2f}')

from xgboost import XGBRegressor
print()
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
evaluate_regression(y_test, y_pred)
print(f'R2:{xgb.score(X_test, y_test):.2f}')



# IG
COLUMNS_SJZ = ['sjz_ground_wind', 'sjz_high', 'sjz_humidity', 'sjz_max_temperature',
              'sjz_max_wind', 'sjz_min_humidity',
            'sjz_min_temperature', 'sjz_pressure', 'sjz_staticstability', 'sjz_sunshine',
            'sjz_temperature', 'sjz_water', 'sjz_wind']
COLUMNS_XT = ['xt_ground_wind', 'xt_high', 'xt_humidity', 'xt_max_temperature',
              'xt_max_wind', 'xt_min_humidity',
            'xt_min_temperature', 'xt_pressure', 'xt_staticstability', 'xt_sunshine',
            'xt_temperature', 'xt_water', 'xt_wind']
# 计算石家庄的IG
sjz_files = ['sjz_q4.csv', 'sjz_q3.csv', 'sjz_q2.csv', 'sjz_q1.csv']
for f in sjz_files:
    df = pd.read_csv(BASE_PATH+'\\'+f)
    Ys = ['sjz_co2', 'sjz_no2', 'sjz_o3', 'sjz_pm2', 'sjz_pm10', 'sjz_so2']
    # Ys = ['sjz_pm2', 'sjz_pm10', 'sjz_so2', 'sjz_no2', 'sjz_co2', 'sjz_o3']
    sjz = []
    for y in Ys:
        sjz.append(Q1_IG(df, y))
    sjz = np.reshape(sjz, (-1, 13))
    # cols = ['sjz_staticstability', 'sjz_temperature', 'sjz_high', 'sjz_ground_wind', 'sjz_max_temperature',
    #         'sjz_min_temperature', 'sjz_water', 'sjz_pressure', 'sjz_humidity', 'sjz_min_humidity', 'sjz_wind',
    #         'sjz_max_wind', 'sjz_sunshine']
    shz_ig = pd.DataFrame(sjz, columns=COLUMNS_SJZ, index=Ys)
    shz_ig.to_csv(BASE_PATH + r'\result\\'+f.split('.')[0]+'_IG.csv')



# 计算邢台的IG
xt_files = ['xt_q4.csv', 'xt_q3.csv', 'xt_q2.csv', 'xt_q1.csv']
for f in xt_files:
    df = pd.read_csv(BASE_PATH+'\\'+f)
    Ys = ['xt_co2', 'xt_no2', 'xt_o3', 'xt_pm2', 'xt_pm10', 'xt_so2']
    # Ys = ['xt_pm2', 'xt_pm10', 'xt_so2', 'xt_no2', 'xt_co2', 'xt_o3']
    sjz = []
    for y in Ys:
        sjz.append(Q1_IG_XT(df, y))
    sjz = np.reshape(sjz, (-1, 13))
    # cols = ['xt_staticstability', 'xt_temperature', 'xt_high', 'xt_ground_wind', 'xt_max_temperature',
    #         'xt_min_temperature', 'xt_water', 'xt_pressure', 'xt_humidity', 'xt_min_humidity', 'xt_wind',
    #         'xt_max_wind', 'xt_sunshine']
    shz_ig = pd.DataFrame(sjz, columns=COLUMNS_XT, index=Ys)
    shz_ig.to_csv(BASE_PATH + r'\result\\'+f.split('.')[0]+'_IG.csv')