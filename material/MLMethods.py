import numpy as np
from material.datahelper import feature_energing
import pandas as pd
from sklearn.model_selection import cross_val_predict, cross_val_score
import random


def evaluate_classifier(y, y_pred):
    """
    评估
    :param y:
    :param y_predt:
    :return:
    """
    # 计算F1 Recall support
    from sklearn.metrics import precision_recall_fscore_support as score
    precision, recall, fscore, support = score(y, y_pred)
    table = pd.DataFrame({'precision': precision, 'recall': recall, 'fscore': fscore, 'support': support})
    print(table)
    return table


def evaluate_regression(y, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    y = y.ravel()
    y_pred = y_pred.ravel()
    MAE = int(mean_absolute_error(y, y_pred))
    MSE = int(mean_squared_error(y, y_pred))
    RMSE = int(np.sqrt(mean_squared_error(y, y_pred)))
    # print(f'MAE:{int(mean_absolute_error(y, y_pred))}')
    # print(f'MSE:{int(mean_squared_error(y, y_pred))}')
    # print(f'RMSE:{int(np.sqrt(mean_squared_error(y, y_pred)))}')
    index = np.random.randint(1, 100)
    result = pd.DataFrame([y[index:index+5], y_pred[index:index+5]], index=['y', 'y_pred'])
    # print('抽样随机结果对比：')
    # print(result)
    return MAE,MSE,RMSE


def generate_models():
    regressors = dict()
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    regressors['lr'] = lr
    from sklearn.tree import DecisionTreeRegressor
    dtr = DecisionTreeRegressor(random_state=3, max_features='sqrt', criterion='mae')
    regressors['dtr'] = dtr
    from sklearn.ensemble import GradientBoostingRegressor
    gbr = GradientBoostingRegressor(random_state=10)
    regressors['gbr'] = gbr
    from xgboost import XGBRegressor
    xgb = XGBRegressor()
    regressors['xgb'] = xgb

    return regressors


if __name__=='__main__':
    BASE_PATH = r'C:\Users\chenshuai\Documents\材料学院\data\贝氏体钢数据统计-总2018'
    files = ['0421', '0502']
    # 所有的回归模型
    regressions = generate_models()
    # 每次数据集的结果
    results = []

    for file in files:
        print('数据集：',file)
        fe = feature_energing(BASE_PATH + file + '.xlsx', info=False, regression=True)
        X_train, X_test, y_train, y_test = fe.preprocess()
        X = fe.X
        Y = fe.Y
        # 打乱数据集
        xy = list(zip(X, Y))
        random.shuffle(xy)
        X[:], Y[:] = zip(*xy)

        df_r2 = []
        df_eva = []
        for k,v in regressions.items():

            scores = cross_val_score(v, X, Y, cv=2, scoring='r2')
            print(k,scores.mean())
            v.fit(X_train, y_train)
            train_r2 = v.score(X_train, y_train)
            test_r2 = v.score(X_test, y_test)
            df_r2.append([train_r2, test_r2])

            train_pred = v.predict(X_train)
            test_pred = v.predict(X_test)
            MAE1, MSE1, RMSE1 = evaluate_regression(y_train, train_pred)
            MAE2, MSE2, RMSE2 = evaluate_regression(y_test, test_pred)
            df_eva.append([MAE1, MAE2, MSE1, MSE2, RMSE1, RMSE2])
            pass

        # R2结果处理
        df_r2 = np.reshape(df_r2, (-1, 2))
        df_r2 = df_r2 * 100
        df_r2 = df_r2.astype(int)
        df_r2 = df_r2.astype(float)
        df_r2 = df_r2 / 100
        df_r2 = pd.DataFrame(df_r2, columns=['Train_r2', 'Test_r2'], index=regressions.keys())
        print(df_r2)

        # MAE
        df_eva = np.reshape(df_eva, (-1, 6))
        df_eva = df_eva.astype(int)
        df_eva = pd.DataFrame(df_eva, columns=['train_MAE','test_MAE','train_MSE',
                                               'test_MSE','train_RMSE','test_RMSE'],
                                      index=regressions.keys())
        print(df_eva.T)
        tf_result = pd.concat([df_eva, df_r2], axis=1)
        print(tf_result)
        results.append(tf_result)
        # tf_result.to_csv(r'C:\Users\chenshuai\Documents\材料学院\data\mothods_result.csv')