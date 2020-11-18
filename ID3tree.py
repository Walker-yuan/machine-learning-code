
import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split



def cross_entropy(cla):
    """
function: 计算信息熵（信息的不确定程度，值越大，代表所含信息越多，同时也代表数据纯度越高）
input:
    :param cla: series, 表示数据类别的列表
output:
    :entropy:交叉熵
entropy = -sum(p * log(p)), p为概率值
    """
    probs = [cla.tolist().count(i)/len(cla) for i in set(cla)]
    entropy = -sum([prob * math.log(prob, 2) for prob in probs])
    return entropy



def Split_data(dataset, feature):
    """
function:依据特征feature划分数据集
input:
    :param dataset: dataframe, 输入的数据集
    :param feature: Index, 需要被划分的特征
output:
    :split_dataset: dic, 划分后的数据集
    """
    split_dataset = {elem : pd.DataFrame for elem in set(dataset[feature])}
    for key in split_dataset.keys():
        split_dataset[key] = dataset[:][dataset[feature] == key]
    return split_dataset




def infogain(dataset, feature, label):
    """
function:计算特征features的信息增益
input:
    :param dataset: dataframe, 输入的数据集
    :param feature: str, 需要被计算的信息增益所对应的特征
    :param label: str, 类别所对应的列名
output:
    :gain_D_A: float, 特征feature的信息增益
H(D) = -sum(len(Ck)/len(D)*log2(len(Ck)/len(D))) , len(Ck)为类Ck的样本个数
H(D|A) = sum(len(Di)/len(D)*H(Di)) ,Di为特征feature的第i个特征值所包含的样本的集合
gain(D, A) = H(D) - H(D|A)
    """
    num_dataset = len(dataset)
    entropy_D = cross_entropy(dataset[label])
    split_dataset = Split_data(dataset, feature)
    entropy_D_A = 0
    for key in split_dataset.keys():
        entropy_D_A += len(split_dataset[key])/num_dataset * cross_entropy(split_dataset[key][label])
    gain_D_A = entropy_D - entropy_D_A
    return gain_D_A




def choose_best_feature(dataset, features, label):
    """
function:从所有特征中选出信息增益最大的特征作为节点的划分
input:
    :param dataset: dataframe, 输入的数据集
    :param features: Index, 特征的集合
output:
    :max_value: float, 信息增益的最大值
    :best_feature: str, 最大信息增益对应的特征
    :split_data: dic, 以最佳特征划分的数据集
    """
    features = [i for i in features if i != label]
    max_value = -999
    best_feature = None
    for feature in features:
        gain_value = infogain(dataset, feature, label)
        if gain_value > max_value:
            max_value = gain_value
            best_feature = feature
    split_data = Split_data(dataset, best_feature)
    return np.float(max_value), best_feature, split_data




def creat_ID3tree(dataset, features, label, alpha = 0.00001, strRoot="-", strRootAttri="-"):
    """
function:利用数据生成一棵决策树
input:
    :param dataset: dataframe, 输入的数据集
    :param features:series, 所有特征
    :param label:str, 类别名称
    :param alpha:float, 信息增益划分的阈值
    :param strRoot:str, 根节点名称
    :param strRootAttri:str, 子节点名称
output:一棵决策树
注意：
（1）若D中所有实例属于同一类别Ck， 则T为单节点树，并将类Ck作为该节点的类标记，返回T
（2）若A(特征集)=空集， 则T为单节点树，并将D中实例数最大的类Ck作为该节点的类标记，并返回T
（3）当前节点所包含的样本集合为空时，把父节点的样本分布作为当前节点的先验分布
    """
    ## 计算各个类别的频率
    freq_label = np.array([[i, dataset[label].tolist().count(i)/len(dataset)] for i in set(dataset[label])]).reshape(-1, 2)

    ## 列出需要划分的特征
    features = [i for i in features if i != label]

    ## 若D中所有实例属于同一类别Ck， 则T为单节点树，并将类Ck作为该节点的类标记，返回T
    if max(freq_label[:, 1]) == 1:
        print(strRoot, strRootAttri, freq_label[np.where(freq_label == max(freq_label[:, 1]))[0], 0][0], 1)
        return freq_label[np.where(freq_label == max(freq_label[:, 1]))[0], 0][0]


    ##若A(特征集)=空集， 则T为单节点树，并将D中实例数最大的类Ck作为该节点的类标记，并返回T
    if len(features) == 0:
        print(strRoot, strRootAttri, freq_label[np.where(freq_label == max(freq_label[:, 1]))[0], 0][0], max(freq_label[:, 1]))
        return freq_label[np.where(freq_label == max(freq_label[:, 1]))[0], 0][0]

    ## 计算得到最优划分特征best_feature以及以最优划分特征划分的数据集split_data以及相应的信息增益max_value
    max_value, best_feature, split_data = choose_best_feature(dataset, features, label)


    ## 如果信息增益小于设定的阈值alpha，则返回T
    if max_value < alpha:
        print(strRoot, strRootAttri, freq_label[np.where(freq_label == max(freq_label[:, 1]))[0], 0][0], max(freq_label[:, 1]))
        return freq_label[np.where(freq_label == max(freq_label[:, 1]))[0], 0][0]

    myTree = {best_feature: {}}  # 这里很巧妙，以此来创建嵌套字典

    ## 否则对于每一个最优划分特征的特征值，得到相应的子数据集进行递归
    for feature in set(dataset[best_feature]):
        ## 当前节点所包含的样本集合为空时，把父节点的样本分布作为当前节点的先验分布
        if len(split_data[feature]) == 0:
            print(strRoot, strRootAttri, freq_label[np.where(freq_label == max(freq_label[:, 1]))[0], 0][0],
                  max(freq_label[:, 1]))
            return freq_label[np.where(freq_label == max(freq_label[:, 1]))[0], 0][0]
        else:
            print(strRoot, strRootAttri, best_feature, feature)
            next_features = [a for a in features if a != best_feature]
            myTree[best_feature][feature] = creat_ID3tree(split_data[feature], next_features, label, alpha, strRoot=best_feature, strRootAttri=feature)
    return myTree


def classify(inputTree, pre_data):
    """
function:利用生成的决策树对数据进行预测
input:
    :param inputTree: dict, 生成的决策树
    :param pre_data: series, 要预测的数据
    :return: class_label: str, 预测数据的类别
    """
    ## 查看决策树字典的第一个键值也即第一个被选出的最优特征
    firstKey = list(inputTree.keys())[0]

    ## 下一个字典是在上一个特征的基础上划分的
    next_dict = inputTree[firstKey]
    class_label = None
    for key in next_dict.keys():
        if pre_data[firstKey] == key:
            if type(next_dict[key]).__name__ == 'dict':
                class_label = classify(next_dict[key], pre_data)
            else:
                class_label = next_dict[key]
    return  class_label


if __name__ == '__main__':
    ## 导入数据并处理无关特征
    df_data = pd.read_table('xigua.txt', sep=',', encoding='gbk')
    dataset = df_data.drop(columns=['编号', '密度', '含糖率'])

    ## 划分训练数据集和测试数据集
    X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:, :-1], dataset['好瓜'], test_size=0.2)

    ## 生成决策树
    train_dataset = pd.concat([X_train, y_train], axis=1)
    my_tree = creat_ID3tree(train_dataset, train_dataset.columns, '好瓜', alpha = 0.001, strRoot="-", strRootAttri="-")
    print(my_tree)

    ## 利用生成的决策树来测试数据
    class_label = []
    for i in range(len(X_test)):
        class_label.append(classify(my_tree, X_test.iloc[i, :]))
    print(class_label)
    print(list(y_test))
    print('测试精度：', len(np.where(np.array(class_label) == np.array(list(y_test)))[0])/len(class_label))