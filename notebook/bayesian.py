# 朴素贝叶斯
import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target
from collections import namedtuple
from pandas.compat import StringIO, BytesIO


def train(x, y):
    """
    朴素贝叶斯训练
    :param x:
    :param y:
    :return:
    """
    m, n = x.shape

    # 拉普拉斯平滑
    p1 = (len(y[y == '是']) + 1) / (m + 2)

    p1_list = []
    p0_list = []

    x1 = x[y == '是']
    x0 = x[y == '否']

    m1, _ = x1.shape
    m0, _ = x0.shape

    for i in range(n):
        xi = x.iloc[:, i]
        p_xi = namedtuple(x.columns[i], ['is_continuous', 'conditional_pro'])

        is_continuous = type_of_target(xi) == 'continuous'
        xi1 = x1.iloc[:, i]
        xi0 = x0.iloc[:, i]

        if is_continuous:  # 连续值时候，contional_pro储存的是[mean,var]
            xi1_mean = np.mean(xi1)
            xi1_var = np.var(xi1)
            xi0_mean = np.mean(xi0)
            xi0_var = np.var(xi0)

            p1_list.append(p_xi(is_continuous, [xi1_mean, xi1_var]))
            p0_list.append(p_xi(is_continuous, [xi0_mean, xi0_var]))
        else:  # 离散直接计算各类别的条件概率
            unique_value = xi.unique()
            nvalue = len(unique_value)

            xi1_value_count = pd.value_counts(xi1)[unique_value].fillna(0) + 1  # 拉普拉斯平滑
            xi0_value_count = pd.value_counts(xi0)[unique_value].fillna(0) + 1
            p1_list.append(p_xi(is_continuous, np.log(xi1_value_count / (m1 + nvalue))))
            p0_list.append(p_xi(is_continuous, np.log(xi0_value_count / (m0 + nvalue))))

    return p1, p1_list, p0_list


def predict(x, p1, p1_list, p0_list):
    """
    朴素贝叶斯预测
    :param x:
    :param p1:
    :param p1_list:
    :param p0_list:
    :return:
    """
    n = len(x)
    x_p1 = np.log(p1)
    x_p0 = np.log(1 - p1)

    for i in range(n):
        if p1_list[i].is_continuous:
            mean1, var1 = p1_list[i].conditional_pro
            mean0, var0 = p0_list[i].conditional_pro

            x_p1 += np.log(1 / (np.sqrt(2 * np.pi) * var1) * np.exp(- (x[i] - mean1) ** 2 / (2 * var1 ** 2)))
            x_p0 += np.log(1 / (np.sqrt(2 * np.pi) * var0) * np.exp(- (x[i] - mean0) ** 2 / (2 * var0 ** 2)))
        else:
            x_p1 += p1_list[i].conditional_pro[x[i]]
            x_p0 += p0_list[i].conditional_pro[x[i]]

    if x_p1 > x_p0:
        return '是'
    else:
        return '否'

if __name__ == '__main__':
    data_string = \
        """
        编号,色泽,根蒂,敲声,纹理,脐部,触感,密度,含糖率,好瓜
        1,青绿,蜷缩,浊响,清晰,凹陷,硬滑,0.697,0.46,是
        2,乌黑,蜷缩,沉闷,清晰,凹陷,硬滑,0.774,0.376,是
        3,乌黑,蜷缩,浊响,清晰,凹陷,硬滑,0.634,0.264,是
        4,青绿,蜷缩,沉闷,清晰,凹陷,硬滑,0.608,0.318,是
        5,浅白,蜷缩,浊响,清晰,凹陷,硬滑,0.556,0.215,是
        6,青绿,稍蜷,浊响,清晰,稍凹,软粘,0.403,0.237,是
        7,乌黑,稍蜷,浊响,稍糊,稍凹,软粘,0.481,0.149,是
        8,乌黑,稍蜷,浊响,清晰,稍凹,硬滑,0.437,0.211,是
        9,乌黑,稍蜷,沉闷,稍糊,稍凹,硬滑,0.666,0.091,否
        10,青绿,硬挺,清脆,清晰,平坦,软粘,0.243,0.267,否
        11,浅白,硬挺,清脆,模糊,平坦,硬滑,0.245,0.057,否
        12,浅白,蜷缩,浊响,模糊,平坦,软粘,0.343,0.099,否
        13,青绿,稍蜷,浊响,稍糊,凹陷,硬滑,0.639,0.161,否
        14,浅白,稍蜷,沉闷,稍糊,凹陷,硬滑,0.657,0.198,否
        15,乌黑,稍蜷,浊响,清晰,稍凹,软粘,0.36,0.37,否
        16,浅白,蜷缩,浊响,模糊,平坦,硬滑,0.593,0.042,否
        17,青绿,蜷缩,沉闷,稍糊,稍凹,硬滑,0.719,0.103,否
        """
    data = pd.read_csv(StringIO(data_string), index_col=0)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    p1, p1_list, p0_list = train(X, y)
    p1, p1_list, p0_list
    x_test = X.iloc[0, :]  # 书中测1 其实就是第一个数据
    print(predict(x_test, p1, p1_list, p0_list))