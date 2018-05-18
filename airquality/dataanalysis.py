from sklearn.feature_selection import chi2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics as mr
import numpy as np


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


first = first['2014-1-1':'2017-1-1']
Y = first.sjz_pm2
X = first.loc[:, 'sjz_staticstability':]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=0)

dtr = DecisionTreeRegressor(random_state=3, max_features='sqrt', criterion='mse')
dtr.fit(X_train, y_train)
y_pred = dtr.predict(X_test)
evaluate_regression(y_test, y_pred)
print(dtr.score(X_test, y_test))
print(dtr.feature_importances_)


