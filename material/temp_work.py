import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

NEED_COLUMS = ['T','theta','G','W']

df = pd.read_excel(r'C:\Users\chenshuai\Documents\材料学院\贝氏体板条宽度W公式拟合.xlsx')
# print(df.columns)
df = df.rename(columns=dict(zip(df.columns,['C','T','theta','G','W','from','other'])))
df = df[NEED_COLUMS]
df.loc[:, 'W'] = [x*1000 for x in df.loc[:, 'W']]

x = df.iloc[:, :-1]
y = df.W

print(np.shape(x),np.shape(y))

lr = LinearRegression()
lr.fit(x,y)
print(f'{lr.score(x, y):.2}')
y_pred = lr.predict(x)
print(f'MAE:{int(mean_absolute_error(y, y_pred))}')
print(f'MSE:{int(mean_squared_error(y, y_pred))}')
print(f'RMSE:{int(np.sqrt(mean_squared_error(y, y_pred)))}')

parameters = pd.DataFrame(np.append(np.array(lr.coef_), [lr.intercept_]), index=['T','σγ','ΔGγ','intercept'])
print(parameters)


result = pd.DataFrame([y.values, y_pred], index=['y', 'y_pred'])
result = result.astype(int)
result.T.to_csv(r'C:\Users\chenshuai\Documents\材料学院\贝氏体板条宽度W公式拟合结果.csv')

from sklearn.metrics import mean_absolute_error, mean_squared_error
print(f'MAE:{int(mean_absolute_error(y, y_pred))}')
print(f'MSE:{int(mean_squared_error(y, y_pred))}')
print(f'RMSE:{int(np.sqrt(mean_squared_error(y, y_pred)))}')