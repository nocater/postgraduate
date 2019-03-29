import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_excel(r'D:\Documents\材料学院\data\贝氏体板条宽度W公式拟合.xlsx')
# df = df.rename(columns=dict(zip(df.columns,['C','T','theta','G','W','from','other'])))
df = df.rename(columns={'C（wt.%)':'C','T(℃)':'T','σγ(MPa)':'sigma','ΔGγ→α(J)':'dG','Wαβ(μm)':'W'})
df = df.loc[:, 'T':'W']
df.W = df.W*1000

x_train, x_test, y_train, y_test= train_test_split(df.iloc[:, :-1], df.W, test_size=0.3, random_state=3)

print('线性分类无测试集 W = 0.406801 T + -0.314194σγ + -0.005893ΔGγ+ 18.746198 R2 0.81  MAE:13 MSE:301 RMSE:17')
# 线性分类器
lr = LinearRegression()
lr.fit(x_train, y_train)
print(f'线性回归:{lr.score(x_test, y_test):.2}')
y_pred = lr.predict(x_test)
print(f'MAE:{int(mean_absolute_error(y_test, y_pred))}')
print(f'MSE:{int(mean_squared_error(y_test, y_pred))}')
print(f'RMSE:{int(np.sqrt(mean_squared_error(y_test, y_pred)))}')

# 决策树
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state=3, max_features='sqrt', criterion='mae')
dtr.fit(x_train, y_train)
print(f'CART:{lr.score(x_test, y_test):.2}')
y_pred = dtr.predict(x_test)
print(f'MAE:{int(mean_absolute_error(y_test, y_pred))}')
print(f'MSE:{int(mean_squared_error(y_test, y_pred))}')
print(f'RMSE:{int(np.sqrt(mean_squared_error(y_test, y_pred)))}')

parameters = pd.DataFrame(np.append(np.array(lr.coef_), [lr.intercept_]), index=['T','σγ','ΔGγ','intercept'])
print(parameters)

# Save result
# result = pd.DataFrame([y.values, y_pred], index=['y', 'y_pred'])
# result = result.astype(int)