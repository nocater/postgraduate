import os
#数据处理
import pandas as pd
import numpy as np
import random
import sklearn.preprocessing as preprocessing
#可视化
import matplotlib.pyplot as plt
import seaborn as sns

#ML
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve

#完成deeplearing.ai 第一课第二周作业
#完成deeplearing.ai 第一课第三周作业
