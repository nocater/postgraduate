#coding=utf-8
import pandas as pd
import numpy as np
from pandas import Series, DataFrame

BASE_PATH = r'C:\Users\chenshuai\Documents\Kaggle\Titanic'
data_train = pd.read_csv(BASE_PATH+r'\train.csv')


def set_missing_ages(df):
    from sklearn.ensemble import RandomForestRegressor
    """
    使用RandomForestClassifier填补年龄属性
    :param df: 
    :return: 
    """

    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]

    # 乘客年龄已知和未知
    know_age = age_df[age_df.Age.notnull()].as_matrix()
    unknow_age = age_df[age_df.Age.isnull()].as_matrix()

    # y就是目标年龄
    y = know_age[:, 0]

    # x 就是特征属性
    X = know_age[:, 1:]
    # 训练模型
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 拟合未知年龄
    predictedAges = rfr.predict(unknow_age[:, 1:])

    # 填充未知年龄
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df, rfr


def set_Carbin_type(df):
    # 必须使用df.loc[(), ''] ()不能丢
    df.loc[(df.Cabin.notnull()), 'Cabin'] = 'Yes'
    df.loc[(df.Cabin.isnull()), 'Cabin'] = 'No'
    return df


data_train, rfr = set_missing_ages(data_train)
data_train = set_Carbin_type(data_train)

# 将一些列变成onehot表示形式 pandas.get_dummies 完成onehot 表示
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
dummies_Plcass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Plcass, dummies_Sex], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

# 对数据进行缩放
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
age_param = scaler.fit(np.array(df['Age']).reshape(-1,1))
df['Age_scaled'] = scaler.fit_transform(np.array(df['Age']).reshape(-1,1), age_param)
fare_param = scaler.fit(np.array(df['Fare']).reshape(-1,1))
df['Fare_scaled'] = scaler.fit_transform(np.array(df['Fare']).reshape(-1,1), fare_param)
# print(df[['Age_scaled', 'Fare_scaled']].head(1))

from sklearn.linear_model import LogisticRegression
# 用正则取出所需要的属性值
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
print(train_df.columns)
train_np = train_df.as_matrix()

y = train_np[:, 0]
X = train_np[:, 1:]

# 训练模型
lr = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
lr.fit(X, y)

# 测试集
data_test = pd.read_csv(BASE_PATH+r'\test.csv')
data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0
tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAges

data_test = set_Carbin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')

df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(np.array(df_test['Age']).reshape(-1, 1), age_param)
df_test['Fare_scaled'] = scaler.fit_transform(np.array(df_test['Fare']).reshape(-1, 1), fare_param)
# print(df_test[['Age_scaled', 'Fare_scaled']].head(4))

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = lr.predict(test)
result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
result.to_csv(BASE_PATH+r'\lr_predictions.csv', index=False)

# 相关系数
print(pd.DataFrame({"columns": list(train_df.columns)[1:], "coef": list(lr.coef_.T)}))

# 交叉验证
from sklearn.model_selection import cross_validate
 #简单看看打分情况
clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
X = all_data.as_matrix()[:,1:]
y = all_data.as_matrix()[:,0]
print(cross_validate(clf, X, y, cv=5))