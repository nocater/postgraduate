from material.datahelper import feature_energing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

path = r'C:\Users\chenshuai\Documents\材料学院\data\贝氏体钢数据统计-总20180502.xlsx'
fe = feature_energing(file=path, regression=True, info=True)
fe.preprocess()
df = fe.target_df

columns = ['C', 'Si', 'Mn', 'Ni', 'Cr', 'Mo', 'Al', 'Co', 'T2', 'T3']

# 获取每一列的Y平均值并替换
for col in columns:
    col_list = df[col]
    df_col = df.groupby(col)['Y'].mean()
    col_list = [df_col.loc[x] for x in col_list]
    df[col] = col_list

#验证数据#
mins = np.min(df, axis=0)
maxs = np.max(df, axis=0)
print(pd.DataFrame([mins, maxs], index=['min:', 'max'], columns=df.columns))


X = df.iloc[:, :-1].as_matrix()
y = df.Y.as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_ = lr.predict(X_test)
print(lr.score(X_test, y_test))
print(f"MAE:{mean_absolute_error(y_test, y_)}")
print(f"RMSE:{np.sqrt(mean_squared_error(y_test, y_))}")
print(lr.coef_)
print(lr.intercept_)
result = pd.DataFrame({'coef':np.append(lr.coef_,lr.intercept_)}, index=df.columns)
result.rename()
print(result)

