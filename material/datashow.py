import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from material.datahelper import feature_energing

def show_distribute(path):
    # path = r'C:\Users\chenshuai\Documents\材料学院\贝氏体钢数据统计-总20180421_pd.xlsx'
    # path = r'C:\Users\chenshuai\Documents\材料学院\贝氏体钢数据统计-chenshuai_pd.xlsx'
    fe = feature_energing(file=path, regression=True, info=False)
    fe.preprocess()
    df = fe.target_df

    imgs = len(df.columns)
    index = 0
    for col in df.columns:
        index += 1
        plt.figure(index)
        plt.xlabel(str(col)+' element')
        plt.ylabel('Count')
        plt.hist(df[col].values, bins=50)
    plt.show()


if __name__ == '__main__':
    path = r'C:\Users\chenshuai\Documents\材料学院\data\贝氏体钢数据统计-总20180502.xlsx'
    # path = r'C:\Users\chenshuai\Documents\材料学院\贝氏体钢数据统计-chenshuai_pd.xlsx'
    show_distribute(path)