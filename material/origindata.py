import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

class feature_energing:
    def __init__(self, file,
                    colomns:'选取的数据维度' = ['C', 'Si', 'Mn', 'Ni', 'Cr', 'Mo', 'Al', '等温温度T2', '回火温度T3', '抗拉强度'],
                    scaler:'是否采用数据缩放' = StandardScaler,
                    regression = True,
                    shuffle:''= True,
                    random_seed:''= 1,
                    categories = 4,
                    onehot = False,
                    info = True,
                 ):
        self.colomns = colomns
        self.sclaer = scaler
        self.regression = regression
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.categories = categories
        self.onehot = onehot
        self.info = info
        self.origin_df = pd.read_excel(file)
        pass

    def preprocess(self):
        """
        数据预处理 默认打乱 4分类 不使用onehot
        :param df:
        :param shuffle:
        :param categories:
        :param onehot:
        :return:
        """

        if self.info:print('对数据进行回归处理' if self.regression else '对数据进行分类处理')

        df = self.origin_df[self.colomns]
        if self.info:
            print('原始数据：', np.shape(df))
            print('列名:', df.columns)
            df.info

        # 重命名列名
        df = df.rename(columns={'等温温度T2': 'T2', '回火温度T3': 'T3', '抗拉强度': 'Y'})

        ##
        print(df.loc[(df.C>3)])
        ##
        # 把所有列转为float 错误的置为N
        df = df.loc[:, :].apply(pd.to_numeric, errors='coerce')

        # 删除强度为空的行
        df = df[df.Y.notnull()]
        if self.info:
            print('删除强度为空:', np.shape(df))

        # 成分元素空值用0填充
        df.loc[:, 'C':'Al'] = df.loc[:, 'C':'Al'].fillna(value=0)

        # T3 填充室温 25
        df.loc[(df.T3.isnull()), 'T3'] = 25

        # T2 转为Float 错误的先删除
        # df.T2 = pd.to_numeric(df.T2, errors='coerce')
        df = df[df.T2.notnull()]

        # 各维度最大最小值
        mins = np.min(df, axis=0)
        maxs = np.max(df, axis=0)
        self.origin_range = pd.DataFrame([mins, maxs], index=['min:', 'max'], columns=self.colomns)
        # if self.info: print('原始数据范围:', self.origin_range, sep='\n')


        from sklearn.model_selection import train_test_split
        X = df.iloc[:, :-1].as_matrix()
        y = df.Y.as_matrix()
        if not self.regression:     # 非回归处理label
            y = self.deal_labels(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=self.shuffle, random_state=self.random_seed)

        # 数据缩放
        if self.regression:
            scaler = self.sclaer()
            scaler = scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        # 各维度最大最小值
        mins = np.min(df, axis=0)
        maxs = np.max(df, axis=0)
        self.target_range = pd.DataFrame([mins, maxs], index=['min:', 'max'], columns=self.colomns)
        # if self.info:print('处理数据范围:', self.target_range, sep='\n')

        self.X_train = X_train
        self.y_trian = y_train
        self.X_test = X_test
        self.y_test = y_test
        return X_train, X_test, y_train, y_test

    def deal_labels(self, y):
        """
        将类别进行离散表示
        """
        min_ = np.min(y)
        max_ = np.max(y)

        # 设置分类边界
        bounds = np.linspace(min_ - 1, max_, self.categories + 1)

        if self.info: print('类别统计:', np.histogram(y, bounds)[0])

        # 画图
        if False:
            from matplotlib import pyplot as plt
            plt.hist(y, bounds)
            plt.title('Split Data with 4 categories')
            plt.xlabel('Tensile Strength')
            plt.ylabel('Count')
            plt.show()

        # 转换类别编号
        y = [(bounds >= i).nonzero()[0][0] for i in y]
        # 编号下标从0开始
        y = y - np.array([1])
        y = y.ravel()

        # 是否使用onehot表示类别
        if self.onehot:
            y = np.array(y).reshape(-1, 1)
            enc = OneHotEncoder()
            y = enc.fit_transform(y).toarray()
        return y


def load_data(path):
    """
    加载excel文档并进行数据处理
    :param path:
    :return:
    """
    # 读取Excel文件
    df = pd.read_excel(path)

    df = df[['C', 'Si', 'Mn', 'Ni', 'Cr', 'Mo', 'Al', '等温温度T2', '回火温度T3', '抗拉强度']]
    # print('Origin shape:', np.shape(df))
    # print(df.shape[0] - df.count())
    # 重命名列名
    df = df.rename(columns={'等温温度T2': 'T2', '回火温度T3': 'T3', '抗拉强度': 'Y'})

    # 把所有列转为float 错误的置为N
    df = df.loc[:, :].apply(pd.to_numeric, errors='coerce')

    # 删除强度为空的行
    df = df[df.Y.notnull()]
    print('After delete NULL Y:', np.shape(df))
    # df.info()

    # 将列转为float errors={‘ignore’, ‘raise’, ‘coerce’} 返回数值 异常 置Nan
    # df.loc[:, 'C':'Al'] = df.loc[:, 'C':'Al'].apply(pd.to_numeric, errors='coerce')

    # 成分元素空值用0填充
    df.loc[:, 'C':'Al'] = df.loc[:, 'C':'Al'].fillna(value=0)
    """df.fillna({'C':0,'Si':0}) """

    # T3 填充室温 25
    df.loc[(df.T3.isnull()), 'T3'] = 25

    # T2 转为Float 错误的先删除
    # df.T2 = pd.to_numeric(df.T2, errors='coerce')
    df = df[df.T2.notnull()]
    # df.loc[:, 'T2'] = df.loc[:, 'T2'].fillna(value=25)

    return df


def compute_pearson(x, y):
    """
    计算pearson系数
    :param x:
    :param y:
    :return:
    """
    pearson = []
    for i in range(x.shape[1]):
        x_ = np.hstack(x[:, i])
        y_ = np.hstack(y)
        pearson.append(np.corrcoef(x_, y_)[1, 0])
    return pearson
    pass


def drawpic(df):
    from matplotlib import pyplot as plt
    plt.scatter(df.C, df.Y)
    plt.title('Carbon and tensile strength relationship scatter plot')
    plt.xlabel('Carbon content (wt.%)')
    plt.ylabel('Tensile Strength (MPa)')
    plt.show()


def evaluate_classifier(y, y_pred):
    """
    评估
    :param y:
    :param y_predt:
    :return:
    """
    # 计算F1 Recall support
    from sklearn.metrics import precision_recall_fscore_support as score
    precision, recall, fscore, support = score(y_test, y_pred)
    table = pd.DataFrame({'precision': precision, 'recall': recall, 'fscore': fscore, 'support': support})
    print(table)
    return table


def evaluate_regression(y, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    y = y.ravel()
    y_pred = y_pred.ravel()
    print(y[:10])
    print(y_pred[:10])
    print(f'MAE:{mean_absolute_error(y, y_pred)}')
    print(f'MSE:{mean_squared_error(y, y_pred)}')
    print(f'RMSE:{np.sqrt(mean_squared_error(y, y_pred))}')
    pass


if __name__ == "__main__":
    # path = r'C:\Users\chenshuai\Documents\材料学院\贝氏体钢数据统计-总20180421_pd.xlsx'
    path = r'C:\Users\chenshuai\Documents\材料学院\贝氏体钢数据统计-chenshuai_pd.xlsx'

    fe = feature_energing(file=path, regression=False, info=True)
    X_train, X_test, y_train, y_test = fe.preprocess()

    # Logist回归
    # from sklearn.linear_model import LogisticRegression
    # lr = LogisticRegression(C=1, penalty='l2', tol=1e-6)
    # lr.fit(X_train, y_train)
    # y_pred = lr.predict(X_test)
    # print(f'Logist: Train Acc: {lr.score(X_train, y_train):.2}  Test Acc:{lr.score(X_test, y_test):.2}')
    # evaluate_classifier(y_test, y_pred)

    from sklearn.tree import DecisionTreeRegressor
    fe.regression=True
    X_train, X_test, y_train, y_test = fe.preprocess()
    # dtr = DecisionTreeRegressor(random_state=1, max_features='sqrt')
    # dtr.fit(X_train, y_train)
    # y_pred = dtr.predict(X_test)
    # train_acc = dtr.score(X_train, y_train)
    # test_acc = dtr.score(X_test, y_test)
    # print(f'CART: Train Score: {train_acc:.2}  Test Score:{test_acc:.2}')
    # evaluate_regression(y_test, y_pred)
    # print(dtr.tree_.max_depth)