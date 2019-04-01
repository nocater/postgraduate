import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv(r'D:\Documents\材料学院\data\贝氏体钢数据-fixedT3(320).csv')

# 对元素进行round
round_columns = ['Si', 'Mn', 'Ni', 'Cr', 'Mo', 'Al']
decimals = pd.Series([1]*len(round_columns), index=round_columns)
df = df.round(decimals)
df.loc[df.C == 0.71, 'C'] = 0.7

# 对元素进行分组
means = df.groupby(['C', 'Si', 'Mn', 'Ni', 'Cr', 'Mo', 'Al', 'T2', 'T3']).mean()
print(means)

# 分析结论： T3=320, C=0.7下 T2与Y负相关
df = pd.read_csv(r'/mnt/d/Documents/材料学院/data/贝氏体钢数据-fixedT3(320).csv')
df = df[df.T2 > 0]
df = df[df.C == 0.7]

plt.scatter(df.T2, df.Y)
plt.show()