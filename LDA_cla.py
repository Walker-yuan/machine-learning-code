import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from eval_fun import cross_validation
from perfermance_measure import accuracy
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from perfermance_measure import PerMeasure
import pandas as pd

class LDA():
    def __init__(self):
        pass

    def prepare_data(self):
        ## 数据准备
        data, target = make_classification(n_samples = 200, n_features = 2, n_redundant = 0, n_informative = 2, n_classes = 2)
        X, y = shuffle(data, target, random_state = 42)
        X = X.astype(np.float32)
        y = y.reshape(-1, 1)
        data = np.concatenate((X, y), axis = 1)
        return data


    def LDA_train(self, X, y):
        """
function: 线性判别分析训练参数
        :param X: array， 输入的特征数据集
        :param y: array, 输入的类标集
        :return: w, array, 投影参数； threshold, float, 分类阈值
        """
        y = y.flatten()
        ## 数据分类
        X0 = X[y == 0, :]
        X1 = X[y == 1, :]

        ## 计算各个类别的均值和协方差矩阵
        mu0 = np.mean(X0, axis=0)
        mu1 = np.mean(X1, axis=0)
        Y0 = X0 - mu0
        Y1 = X1 - mu1
        sigma0 = np.dot(Y0.T, Y0)
        sigma1 = np.dot(Y1.T, Y1)
        ## 类内散度矩阵
        Sw = sigma0 + sigma1
        ## 利用矩阵奇异值分解来求类内散度矩阵的逆
        U, sigma, VT = np.linalg.svd(Sw)
        inv_Sw = np.dot(np.dot(VT.T, np.linalg.pinv(np.diag(sigma))), U.T)
        ## 根据w = Sw**(-1)*(mu1-mu0)来求参数
        w = np.dot(inv_Sw, mu1-mu0)# mu0 - mu1可能等于0？np.atleast_1d?
        ## 求解LDA模型的分类阈值
        threshold = np.dot((mu0+mu1)/2, w)
        return w, threshold


    def LDA_predict(self, X, w, threshold):
        """
function：LDA预测
        :param X: array, 待预测的特征集
        :param w: array, 训练得到的投影参数
        :param threshold: float, 分类阈值
        :return: array, 预测类别以及预测得分(注意size，一般是n*1)
        """
        y_predicet_score = np.dot(X, w)
        y_predict = (y_predicet_score > threshold) * 1
        return y_predict.reshape(-1, 1), y_predicet_score.reshape(-1, 1)

    def LDA_acc(self, y_test, y_predict):
        """
function: 调用函数accuracy函数求解精度
        :param y_test: array, 测试集
        :param y_predict: array, 预测类标
        :return: float, 精度
        """
        return accuracy(y_test, y_predict)

    def LDA_crossvalidation(self, X, y, K):
        """
function：使用交叉验证来验证模型
        :param X: array, 样本集合
        :param y: array， 样本相应的类标集
        :param K: int, 交叉验证的折数
        """
        ## 调用cross_validation函数划分测试集和训练集
        X_train, y_train, X_test, y_test = cross_validation(X, y, K)
        acc_totol = 0
        ## 对于每一折数据求解模型的精度
        for i in range(K):
            w, threshold = self.LDA_train(X_train[i], y_train[i])
            y_predict, y_predicet_score = self.LDA_predict(X_test[i], w, threshold)
            acc = accuracy(y_test[i], y_predict)
            acc_totol += acc
            print('%d折交叉验证, 第%d次精度：%f'%(K, i+1, acc))
        ## 打印模型交叉验证的平均精度
        print('%d折交叉验证平均精度：%f'%(K, acc_totol/K))

    def LDA_2classification_plot(self, X_train, y_train, title, w=None, threshold=None):
        """画出数据分类图以及拟合的直线"""
        n = X_train.shape[0]
        xcord1 = []
        ycord1 = []
        xcord2 = []
        ycord2 = []
        for i in range(n):
            if y_train[i] == 1:
                xcord1.append(X_train[i, 0])
                ycord1.append(X_train[i, 1])
            else:
                xcord2.append(X_train[i, 0])
                ycord2.append(X_train[i, 1])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xcord1, ycord1, s = 32, c = 'red')
        ax.scatter(xcord2, ycord2, s = 32, c = 'green')
        if w is not None:
            x = np.arange(-5, 5, 0.1)
            y = (threshold - w[0] * x) / w[1]
            ax.plot(x, y)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title(title)
        plt.show()



if __name__ == '__main__':
    ## 类实例化
    lda = LDA()

    # ## 数据准备
    # data = lda.prepare_data()
    # X = data[:, :-1]
    # y = data[:, -1]
    # print('——————————————————交叉验证——————————————————')
    # lda.LDA_crossvalidation(X, y, 5)
    #
    # ## 利用后面写的评估函数进行测试
    # ## 数据准备
    # print('——————————————————重新划分测试数据集进行验证————————————————')
    # X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1].reshape(-1, 1), test_size=0.3)
    # w, threshold = lda.LDA_train(X_train, y_train)
    # y_predict, y_predict_score = lda.LDA_predict(X_test, w, threshold)
    # acc_lda = lda.LDA_acc(y_test, y_predict)
    # print('lda测试集上的精度为：%f'%(acc_lda))
    # lda.LDA_2classification_plot(X_train, y_train,'lda训练集分类图及lda分类边界',  w, threshold)
    # lda.LDA_2classification_plot(X_test, y_predict, title='lda验证集分类图')



    # ## 朴素贝叶斯
    from Naive_Bayes import Naive_bayes

    # bayes_X_train = pd.DataFrame(X_train, columns=['X1', 'X2'])
    # bayes_X_test = pd.DataFrame(X_test, columns=['X1', 'X2'])
    # bayes_y_train = pd.Series(y_train.flatten())
    # bayes_y_test = pd.Series(y_test.flatten())
    #
    # naive_bayes = Naive_bayes(bayes_X_train, bayes_y_train)
    # classes, class_prior, prior = naive_bayes.Naive_bayes_train()
    # bayes_y_predict = naive_bayes.predict(bayes_X_test)
    # acc_nb = accuracy(bayes_y_test.values, bayes_y_predict)
    # print('naive_bayes测试集上的精度为：%f'%(acc_nb))
    # lda.LDA_2classification_plot(X_test, bayes_y_predict, title='naive_bayes验证集分类图')



    ## iris数据集
    from sklearn.datasets import load_iris
    iris = load_iris()
    columns = ['sepal length', 'sepal width', 'petal length', 'petal width']
    X = pd.DataFrame(iris.data, columns=columns)
    y = pd.Series(iris.target)
    bayes_X_train, bayes_X_test, bayes_y_train, bayes_y_test = train_test_split(X, y, test_size=0.3)
    X_train, X_test, y_train, y_test = bayes_X_train.values, bayes_X_test.values, bayes_y_train.values, bayes_y_test.values

    print('——————————————iris数据集上的测试——————————————')
    naive_bayes = Naive_bayes(bayes_X_train, bayes_y_train)
    classes, class_prior, prior = naive_bayes.Naive_bayes_train()
    bayes_y_predict = naive_bayes.predict(bayes_X_test)
    acc_nb = accuracy(bayes_y_test.values, bayes_y_predict)
    # print(bayes_y_predict, bayes_y_test)
    print('naive_bayes测试集上的精度为：%f'%(acc_nb))

    w, threshold = lda.LDA_train(X_train, y_train)
    y_predict, y_predict_score = lda.LDA_predict(X_test, w, threshold)
    acc_lda = lda.LDA_acc(y_test, y_predict.flatten())
    # print(y_predict, y_test)
    print('lda测试集上的精度为：%f'%(acc_lda))



    # ## 利用perfermance_measure中的函数进行评估
    # eval = PerMeasure()
    #
    # ## 计算混淆矩阵及相应指标
    # con_mat = eval.confusion_mat(y_test, y_predict)  # 混淆矩阵
    # pre = eval.precision(con_mat)  # 查准率
    # rec = eval.recall(con_mat)  # 查全率
    # tpr = eval.TPR(con_mat)  # 真正例率
    # fpr = eval.FPR(con_mat)  # 假正例率
    # f_beta = eval.F_beta(con_mat, 1)  # F_beta度量
    # print("混淆矩阵：\n", con_mat)
    # print('查准率：%f,\n 查全率：%f, \n真正例率：%f, \n假正例率：%f, \nF_beta度量：%f' % (pre, rec, tpr, fpr, f_beta))
    #
    # ## 计算AUC和AUPRC并画出P-R曲线和ROC曲线
    # pre_list, rec_list, tpr_list, fpr_list = eval.confusion_list(y_test, y_predict_score)
    # # print(tpr_list)
    # # print(fpr_list)
    # AUC, AUPRC = eval.AUC2AUPRC(y_test, y_predict_score)
    # print('AUC值：%f, AUPRC值：%f' % (AUC, AUPRC))
    # eval.Plot_Pr_Roc(y_test, y_predict_score)


