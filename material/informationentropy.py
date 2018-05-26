from material.datahelper import feature_energing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

if __name__ == '__main__':
    """
    概率模型 数据缩放与否没有影响 依赖于数据的分布
    """
    path = r'C:\Users\chenshuai\Documents\材料学院\data\贝氏体钢数据统计-总20180502.xlsx'
    # 查看第三次仅仅增加数据的信息
    fe = feature_energing(file=path, regression=True, info=False)
    fe.preprocess()
    df = fe.target_df
    X = fe.X
    Y = df.Y

    # 数据缩放
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    print(np.max(X, axis=0))

    dtr = DecisionTreeRegressor()
    rfr = RandomForestRegressor(random_state=0, n_jobs=-1)
    param_grid = {
        'n_estimators': [10, 50, 100, 200, 300, 400, 500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [3,5,9,13,19,26,40],
    }
    # gs = GridSearchCV(n_jobs=-1, estimator=rfr, param_grid=param_grid, cv=5)
    # gs.fit(fe.X, fe.Y)
    # print(gs.best_score_)
    # print(gs.best_params_)
    # dtr = DecisionTreeRegressor(**gs.best_params_)
    # dtr.fit(fe.X, fe.Y)
    rfr = RandomForestRegressor(random_state=0, **{'max_depth': 13, 'max_features': 'sqrt', 'n_estimators': 200})
    rfr.fit(X, Y)
    print(fe.columns)
    print(rfr.feature_importances_)

    # IE
    r = pd.DataFrame({'col':['C', 'Si', 'Mn', 'Ni', 'Cr', 'Mo', 'Al', 'Co', 'T2', 'T3'],
                      'IG':[0.28326462, 0.10479349, 0.07890757, 0.0313342, 0.11564087,
                            0.0483494, 0.03508706, 0.0043995, 0.27546628, 0.02275701]
                      })
    r = r.sort_values('IG')
    print(r)
    # ['C', 'Si', 'Mn', 'Ni', 'Cr', 'Mo', 'Al', 'Co', 'T2', 'T3', '抗拉强度']
    # [0.28326462 0.10479349 0.07890757 0.0313342  0.11564087 0.0483494
    #  0.03508706 0.0043995  0.27546628 0.02275701]

