#coding: utf-8
import numpy as np
'''
    计算信息增益
    powerd by ayonel
'''
class InformationGain:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.totalSampleCount = X.shape[0]      # 样本总数
        self.totalSystemEntropy = 0             # 系统总熵
        self.totalClassCountDict = {}           # 存储每个类别的样本数量是多少
        self.nonzeroPosition = X.T.nonzero()    # 将X转置之后输出非零值的位置
        self.igResult = []                      # 保存结果的list
        self.wordExistSampleCount = 0
        self.wordExistClassCountDict = {}
        self.iter()


    # 将结果列表排序输出
    def get_result(self):
        return self.igResult

    # 计算系统总熵
    def cal_total_system_entropy(self):
        # 计算每个类别各有多少个
        for label in self.y:
            if label not in self.totalClassCountDict:
                self.totalClassCountDict[label] = 1
            else:
                self.totalClassCountDict[label] += 1
        for cls in self.totalClassCountDict:
            probs = self.totalClassCountDict[cls] / float(self.totalSampleCount)
            self.totalSystemEntropy -= probs * np.log(probs)


    # 遍历nonzeroPosition时，逐步计算出每个word的信息增益
    def iter(self):
        self.cal_total_system_entropy()

        pre = 0
        for i in range(len(self.nonzeroPosition[0])):
            if i != 0 and self.nonzeroPosition[0][i] != pre:
                for notappear in range(pre+1, self.nonzeroPosition[0][i]):  # 如果一个词在整个样本集中都未出现，则直接赋为0
                    self.igResult.append(0.0)
                ig = self.cal_information_gain()
                self.igResult.append(ig)
                self.wordExistSampleCount = 0
                self.wordExistClassCountDict = {}
                pre = self.nonzeroPosition[0][i]
            self.wordExistSampleCount += 1
            yclass = self.y[self.nonzeroPosition[1][i]]  # 求得当前样本的标签
            if yclass not in self.wordExistClassCountDict:
                self.wordExistClassCountDict[yclass] = 1
            else:
                self.wordExistClassCountDict[yclass] += 1
        # 计算最后一个单词的ig
        ig = self.cal_information_gain()
        self.igResult.append(ig)

    # 计算ig的主要函数
    def cal_information_gain(self):
        x_exist_entropy = 0
        x_nonexist_entropy = 0

        for cls in self.wordExistClassCountDict:
            probs = self.wordExistClassCountDict[cls] / float(self.wordExistSampleCount)
            x_exist_entropy -= probs * np.log(probs)

            probs = (self.totalClassCountDict[cls] - self.wordExistClassCountDict[cls]) / float(self.totalSampleCount - self.wordExistSampleCount)
            if probs == 0: #该单词在每条样本中都出现了，虽然该几率很小
                x_nonexist_entropy = 0
            else:
                x_nonexist_entropy -= probs*np.log(probs)

        for cls in self.totalClassCountDict:
            if cls not in self.wordExistClassCountDict:
                probs = self.totalClassCountDict[cls] / float(self.totalSampleCount - self.wordExistSampleCount)
                x_nonexist_entropy -= probs*np.log(probs)

        # 合并两项，计算出ig
        ig = self.totalSystemEntropy - ((self.wordExistSampleCount/float(self.totalSampleCount))*x_exist_entropy +
                                        ((self.totalSampleCount-self.wordExistSampleCount)/float(self.totalSampleCount)*x_nonexist_entropy))
        return ig
if __name__ == '__main__':
    X = np.array([[1,0,0,1],[0,1,1,1],[0,0,1,0]])
    y = [0,0,1]
    ig = InformationGain(X, y)
    print(ig.get_result())