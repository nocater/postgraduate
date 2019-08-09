1. # 数据分析流程

   从给定的、有限的、用于学习的训练数据(training data)集合出发，假设数据是独立同分布产生的；并且假设要学习的模型属于某个函数集合，成为假设空间(hypothesis space)；应用某个评价准则(evaluation criterion)，从假设空间中选取一个最后的模型，是他对一直训练数据及位置测试数据（test data)在给定的评价准则下有最优的预测；最优模型的选择是由算法实现。统计学习的三要素：模型(model)，策略(strategy)，算法(algorithm)。

   模型：就是索要学习的条件概率分布或决策函数。

   策略：损失函数

   算法：求解最优化问题的算法，如最小二乘法，梯度下降法

   机器学习的方法步骤：

   1. 得到一个有限的训练数据集合
   2. 确定包含所有可能的模型的假设空间，即学习模型的集合
   3. 确定模型的选择的准则，即学习的策略
   4. 求解最优模型的算法，即学习的算法
   5. 通过学习方法选择最优模型
   6. 利用学习的最优模型对新数据进行预测或分析

   

   从实践角度出发，方法步骤：

   1. 问题定义：监督(分类/回归/标注）
   
   2. 特征工程(通用)：对原始数据进行清洗。
   
   3. 特征工程(与模型弱相关)：数据预处理(离散化，缺失值，特征缩放)
   
      > 通常使用[`sklearn.preprocessing`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing)中的方法，比如Minmax_scale,StandardScaler.
   
   4. 特征工程(与模型强相关)：
      1. 降维：PCA/LDA
   
         > 通常使用[`sklearn.decomposition`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition)中的方法，包括PCA等
   
      2. 特征选择：
   
         > 通常使用[`sklearn.feature_selection`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection)中的方法，如chi2,SelectKBest
   
         1. Filter:相关系数，CHI，信息增益，互信息
         2. Wrapper:
         3. Embedding:学习器自动选择特征
   
   5. 选择模型
   
      1. 线性模型：[`sklearn.linear_model`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model)
      2. 集成模型：[`sklearn.ensemble`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble)，包括AdaBoost,GBDT,RandomForest等
      3. 树模型：[`sklearn.tree`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree)，决策树
      4. SVM：[`sklearn.svm`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm)
   
   6. 模型学习于验证
   
      > 主要属于[`sklearn.model_selection`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection)，包含四个部分：
      >
      > 1. 类别切分：GroupKFold，KFold
      > 2. 训练验证集切分：[`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split)
      > 3. 超参数优化：[`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV)
      > 4. 模型验证：[`cross_validate`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate)，[`validation_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html#sklearn.model_selection.validation_curve)。
   
   7. 评估指标
   
      > 分类评估指标：[`auc`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html#sklearn.metrics.auc), [`f1_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score),[`roc_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve)等
      >
      > 回归评估指标：[`mean_absolute_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error)，[`r2_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score)等
      >
      > 多类别排名指标
      >
      > 聚类评估值指标