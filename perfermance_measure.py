import numpy as np
import matplotlib.pyplot as plt

#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class PerMeasure():
    def __init__(self):
        pass

        # self.y = y
        # self.y_predict = y_predict

        ## 判断输入的列标是否是(-1, 1)， 如果是的话就转化成(0, 1)
        # if -1 in set(self.y):
        #     self.y = (self.y+1)/2



    def inverse10201(self, y):
        """
    function: 将（0, 1）标签转化成（1, 0）标签
        input:
        :param y: array, 输入的集合
        :return: array, 转化后的标签
        """
        return ~(y.astype(np.int)) + 2


    def confusion_mat(self, y, y_predict):
        """
    function: 计算模型的混淆矩阵
        input:
        :param y: array, 测试集的标签
        :param y_predict: array, 预测标签
        :return:Pre: array, 返回模型的混淆矩阵
        """
        ## 分别计算混淆矩阵中的真正例(TP)、假正例(FP)、假反例(FN)、真反例(TN)
        TP = np.sum(y*y_predict)
        FN = np.sum(y*self.inverse10201(y_predict))
        FP = np.sum(self.inverse10201(y)*y_predict)
        TN = np.sum(self.inverse10201(y)*self.inverse10201(y_predict))

        ## 混淆矩阵
        confusion_mat = np.array([[TP, FN], [FP, TN]])
        return confusion_mat

    def precision(self, confusion_mat):
        """
    function: 根据混淆矩阵计算查准率
    input:
        :confusion: array, 混淆矩阵
        :return: float, 查准率
        """
        Pre = confusion_mat[0, 0]/(confusion_mat[0, 0] + confusion_mat[1, 0]+10**(-8))
        return Pre

    def recall(self, confusion_mat):
        """
    function: 根据混淆矩阵计算查全率
        input:
        :confusion: array, 混淆矩阵
        :return: float, 查全率
        """
        Rec = confusion_mat[0, 0]/(confusion_mat[0, 0] + confusion_mat[0, 1]+10**(-8))
        return Rec

    def FPR(self, confusion_mat):
        """
    function: 根据混淆矩阵计算假正例率
        input:
        :confusion: array, 混淆矩阵
        :return: float, 假正例率
        """
        Fpr = confusion_mat[1, 0]/(confusion_mat[1, 1] + confusion_mat[1, 0]+10**(-8))
        return Fpr

    def TPR(self, confusion_mat):
        """
funtion: 计算该模型的真正例率
    input:
        :confusion: array, 混淆矩阵
        :return: float, 真正例率
        """
        Tpr = confusion_mat[0, 0]/(confusion_mat[0, 0] + confusion_mat[0, 1]+10**(-8))
        return Tpr

    def F_beta(self, confusion_mat, beta):
        """
function: 模型度量或者评估的一种方式, beta>0, beta衡量了查全率Recall对查准率Precision的相对重要性,
            beta = 1时退化为F1度量, beta>1时，查全率有更大影响; beta<1时，查准率有更大影响.
            1/f_beta = 1/(1+beta**2) * (1/Pre + beta**2/Rec)
input:
        :confusion: array, 混淆矩阵
        :param beta: float, 衡量查全率和查准率相对重要性的指标
        :return: f_beta, float, 模型评估度量值
        """
        Pre = self.precision(confusion_mat)
        Rec = self.recall(confusion_mat)
        f_beta = (1+beta**2) * Pre * Rec/((beta**2)*Pre + Rec)
        return  f_beta

    def confusion_list(self, y, y_predict_score):
        """
function: 根据不同的阈值计算查准率、查全率、真正例子、假正例率
        :param y: array, 输入的数据标签
        :param y_predict_score: array, 更具模型计算得出的预测得分
        :return: array, 不同阈值下的查准率、查全率、真正例子、假正例率，
        """
        sort_y_predict_score = np.sort(y_predict_score)
        y_predict = y_predict_score.copy()
        Pre_list = []
        Rec_list = []
        Fpr_list = []
        Tpr_list = []
        for i in sort_y_predict_score:
            y_predict = y_predict_score.copy()
            y_predict[np.where(y_predict > i)] = 1
            y_predict[np.where(y_predict <= i)] = 0
            confusion_mat = self.confusion_mat(y, y_predict)
            Pre_list.append(self.precision(confusion_mat))
            Rec_list.append(self.recall(confusion_mat))
            Fpr_list.append(self.FPR(confusion_mat))
            Tpr_list.append(self.TPR(confusion_mat))

        ## 排序
        Pre_list = sorted(Pre_list, reverse=True)
        Rec_list = sorted(Rec_list)
        Fpr_list = sorted(Fpr_list)
        Tpr_list = sorted(Tpr_list)
        # print(Pre_list)
        # print(Rec_list)
        return np.array(Pre_list), np.array(Rec_list), np.array(Fpr_list), np.array(Tpr_list)


    def AUC2AUPRC(self, y, y_predict_score):
        """
function: 计算模型的AUC值和AUPRC值
        :param y: array, 输入的数据标签
        :param y_predict_score: array, 更具模型计算得出的预测得分
        :return: float， 计算的AUC值和AUPRC值
        """
        Pre_list, Rec_list, Fpr_list, Tpr_list = self.confusion_list(y, y_predict_score)
        num = len(Tpr_list)
        AUC = 0.5 * Tpr_list[0] * Fpr_list[0]
        AUPRC = 0.5 * (1+Pre_list[0]) * Rec_list[0]
        for i in range(1, num):
            # print(AUC, AUPRC)
            AUC += 0.5 * (Tpr_list[i] + Tpr_list[i-1]) * abs(Fpr_list[i]-Fpr_list[i-1])
            AUPRC += 0.5 * (Pre_list[i]+Pre_list[i-1]) * abs(Rec_list[i]-Rec_list[i-1])
        return AUC, AUPRC

    def Plot_Pr_Roc(self, y, y_predict_score):
        """
function: 画出P-R曲线和ROC曲线
        :param y: array, 输入的数据标签
        :param y_predict_score: array, 更具模型计算得出的预测得分
        """
        Pre_list, Rec_list, Fpr_list, Tpr_list = self.confusion_list(y, y_predict_score)
        # print(Tpr_list)
        # print(Fpr_list)
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        ax1.plot(Rec_list, Pre_list)
        ax1.set_title('P-R曲线')
        ax1.set_xlabel('Rec(查全率)')
        ax1.set_ylabel('Pre(查准率)')

        ax2 = fig.add_subplot(122)
        ax2.plot(Fpr_list, Tpr_list)
        ax2.set_title('ROC曲线')
        ax2.set_xlabel('FPR(假正例率)')
        ax2.set_ylabel('TPR(真正例率)')

        fig.show()


def accuracy(y, y_predict):
    """
funtion: 计算模型的精度， 二分类及多分类
input:
    :param y: array, 测试集的标签
    :param y_predict: array, 预测标签
    :return:acc: float, 输出模型的精度
    """
    # y = y.flatten()
    # y_predict = y_predict.flatten()
    num_y = y.shape[0]
    acc = np.sum(y == y_predict) / num_y
    return acc



# if __name__ == '__main__':
#     ## 数据准备
#     y = np.concatenate((np.ones(50), np.zeros(50)))
#     y_predict = np.concatenate((np.ones(60), np.zeros(40)))
#     y_predict_score = np.random.rand(100)
#
#     ## 测试
#     eval = PerMeasure()
#     con_mat = eval.confusion_mat(y, y_predict)
#     pre = eval.precision(con_mat)
#     rec = eval.recall(con_mat)
#     tpr = eval.TPR(con_mat)
#     fpr = eval.FPR(con_mat)
#     f_beta = eval.F_beta(con_mat, 1)
#     print(con_mat, pre, rec, tpr, fpr, f_beta)
#     pre_list, rec_list, tpr_list, fpr_list = eval.confusion_list(y, y_predict_score)
#     print(pre_list)
#     AUC, AUPRC = eval.AUC2AUPRC(y, y_predict_score)
#     print('/n')
#     print(AUC, AUPRC)
#     eval.Plot_Pr_Roc(y, y_predict_score




# y = np.array([0, 1, 0, 1, 0, 1])
# y_predict = np.array([0, 1, 1, 0, 0, 1])
# a = PerMeasure(y, y_predict)
# b = a.confusion_mat()
# print(b)

