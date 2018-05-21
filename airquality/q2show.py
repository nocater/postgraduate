import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

if __name__=='__main__':
    """
    Question 2
    石家庄全年数据预测模型
    测试集合108
    """
    df = pd.read_csv(r'C:\Users\chenshuai\Documents\airquality\result\Question2_xt.csv')
    df = shuffle(df, random_state=3)
    y1 = df.predict.values
    y2 = df.real.values

    # y1, y2 = shuffle(y1, y2, random_state=3)
    x = range(1, len(y1)+1)
    plt.plot(x, y1, '-', label='pridection')  # 'o-',
    plt.plot(x, y2, '-.', label='real')       # '*-',
    plt.legend()
    plt.xlabel('test')
    plt.ylabel('PM2.5')
    plt.title('XT_RFModel')
    plt.show()
    #  R2:0.82/0.69  MAE:34  MSE:1946  RMSE:44