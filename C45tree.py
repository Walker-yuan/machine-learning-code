import pandas as pd
import math
import numpy as np



def cross_entropy(cla):
    """
function: 计算信息熵（信息的不确定程度，值越大，代表所含信息越多，同时也代表数据纯度越高）
input:
    :param cla:一个表示数据类别的列表
output:交叉熵
entropy = -sum(p * log(p)), p为概率值
    """
    probs = [cla.tolist().count(i)/len(cla) for i in set(cla)]##dataframe or series有自身的计数函数value_counts()
    entropy = -sum([prob * math.log(prob, 2) for prob in probs])
    return entropy



def Split_data(dataset, feature, split_point =  None):
    """
function:根据特征feature划分数据集dataset
input:
    :param dataset: dataframe, 需要划分的数据集
    :param feature: str, 用于划分数据集的特征
    :split_point : float, 连续属性的划分点
output:
    :split_data: dic, 划分后的数据集
    """
    ## 当特征feature是连续型特征时
    if type(dataset[feature].tolist()[0]).__name__ in ['int', 'int32', 'int64', 'float', 'float32', 'float64']:
        split_dataset = {'> %f'%(split_point): dataset[:][dataset[feature] > split_point],
                         '<= %f'%(split_point): dataset[:][dataset[feature] <= split_point]}
    else:
    ## 当特征feature是离散型特征时
        split_dataset = {elem: pd.DataFrame for elem in set(dataset[feature])}
        for key in split_dataset.keys():
            split_dataset[key] = dataset[:][dataset[feature] == key]
    return split_dataset

# def Split_discrete_data(dataset, feature):
#     """
# function:根据特征feature划分数据集dataset
# input:
#     :param dataset: dataframe, 需要划分的数据集
#     :param feature: str, 用于划分数据集的特征
# output:
#     :split_data: dic, 划分后的数据集
#     """
#     split_dataset = {elem: pd.DataFrame for elem in set(dataset[feature])}
#     for key in split_dataset.keys():
#         split_dataset[key] = dataset[:][dataset[feature] == key]
#     return split_dataset
#
# def Split_continuous_data(dataset, feature, split_point):
#     """
# function:根据特征feature划分数据集dataset
# input:
#     :param dataset: dataframe, 需要划分的数据集
#     :param feature: str, 用于划分数据集的特征
#     :split_point : float, 连续属性的划分点
# output:
#     :split_data: dic, 划分后的数据集
#     """
#     split_dataset = {'> %f'%(split_point): dataset[:][dataset[feature] > split_point],
#                         '<= %f'%(split_point): dataset[:][dataset[feature] <= split_point]}
#     return split_dataset



def infogain_ratio(dataset, feature, label, split_point =  None):
    """
function: 计算特征feature的信息增益率
input:
    :param dataset: dataframe, 输入的数据集
    :param feature: str, 需要被计算的信息增益率所对应的特征
    :param label: str, 类别所对应的列名
    :param split_point: float, 连续属性的划分点
output:
    gain_ratio: float, 特征feature的信息增益率
    """
    num_dataset = len(dataset)
    entropy_D = cross_entropy(dataset[label])
    split_dataset = Split_data(dataset, feature, split_point)
    entropy_D_A = 0
    intrinsic_value = 0
    for key in split_dataset.keys():
        intrinsic_value -= len(split_dataset[key])/num_dataset * math.log(len(split_dataset[key])/num_dataset, 2)
        entropy_D_A += len(split_dataset[key])/num_dataset * cross_entropy(split_dataset[key][label])
    gain_D_A = entropy_D - entropy_D_A
    gain_ratio = gain_D_A/(intrinsic_value+0.1)
    return gain_ratio
#
# def infogain_ratio_discrete(dataset, feature, label):
#     """
# function: 计算特征离散型特征feature的信息增益率
# input:
#     :param dataset: dataframe, 输入的数据集
#     :param feature: str, 需要被计算的信息增益率所对应的特征
#     :param label: str, 类别所对应的列名
# output:
#     :param gain_ratio: float, 特征feature的信息增益率
#     """
#     num_dataset = len(dataset)
#     entropy_D = cross_entropy(dataset[label])
#     split_dataset = Split_discrete_data(dataset, feature)
#     entropy_D_A = 0
#     intrinsic_value = 0
#     for key in split_dataset.keys():
#         intrinsic_value -= len(split_dataset[key])/num_dataset * math.log(len(split_dataset[key])/num_dataset, 2)
#         entropy_D_A += len(split_dataset[key])/num_dataset * cross_entropy(split_dataset[key][label])
#     gain_D_A = entropy_D - entropy_D_A
#     gain_ratio = gain_D_A/intrinsic_value
#     return gain_ratio
#
#
#
# def infogain_ratio_continuous(dataset, feature, label, split_point):
#     """
# function: 计算连续型特征feature的信息增益率
# input:
#     :param dataset: dataframe, 输入的数据集
#     :param feature: str, 需要被计算的信息增益率所对应的特征
#     :param label: str, 类别所对应的列名
#     :param split_point: float, 连续属性的划分点
# output:
#     :param gain_ratio: float, 特征feature的信息增益率
#     """
#     num_dataset = len(dataset)
#     entropy_D = cross_entropy(dataset[label])
#     split_dataset = Split_continuous_data(dataset, feature, split_point)
#     entropy_D_A = 0
#     intrinsic_value = 0
#     for key in split_dataset.keys():
#         intrinsic_value -= len(split_dataset[key])/num_dataset * math.log(len(split_dataset[key])/num_dataset, 2)
#         entropy_D_A += len(split_dataset[key])/num_dataset * cross_entropy(split_dataset[key][label])
#     gain_D_A = entropy_D - entropy_D_A
#     gain_ratio = gain_D_A/intrinsic_value
#     return gain_ratio



def choose_best_feature(dataset, features, label):
    """
function:从所有特征中选出信息增益最大的特征作为节点的划分
input:
    :param dataset: dataframe, 输入的数据集
    :param features: Index, 特征的集合
    :param label: str, 类别所对应的列名
output:
    :max_value: float, 信息增益率的最大值
    :best_feature: str, 最大信息增益对应的特征
    :split_data: dic, 以最佳特征划分的数据集
    """
    features = [i for i in features if i != label]
    max_value = -999
    best_feature = None
    best_split_point = 0
    for feature in features:
        ## 如果是连续型属性
        if type(dataset[feature].tolist()[0]).__name__ in ['int', 'int32', 'int64', 'float', 'float32', 'float64']:
            sort_feaValue = np.sort(np.unique(dataset[feature]))
            split_point_list = [np.float(sort_feaValue[i]+sort_feaValue[i+1])/2.0 for i in range(len(sort_feaValue)-1)]
            for split_point in split_point_list:
                gain_ratio = infogain_ratio(dataset, feature, label, split_point = split_point)
                if gain_ratio > max_value:
                    max_value = gain_ratio
                    best_feature = feature
                    best_split_point = split_point
        ## 如果是离散型属性
        else:
            gain_value = infogain_ratio(dataset, feature, label)
            if gain_value > max_value:
                max_value = gain_value
                best_feature = feature

    split_data = Split_data(dataset, best_feature, split_point = best_split_point)
    return np.float(max_value), best_feature, split_data, np.float(best_split_point)

def creat_C45tree(dataset, features, label, alpha = 0.00001, strRoot="-", strRootAttri="-"):
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
    ##计算各个类别的频率
    freq_label = np.array([[i, dataset[label].tolist().count(i)/len(dataset)] for i in set(dataset[label])]).reshape(-1, 2)

    ##列出需要划分的特征
    features = [i for i in features if i != label]

    ##若D中所有实例属于同一类别Ck， 则T为单节点树，并将类Ck作为该节点的类标记，返回T
    if max(freq_label[:, 1]) == 1:
        print(strRoot, strRootAttri, freq_label[np.where(freq_label == max(freq_label[:, 1]))[0], 0][0], 1)
        return freq_label[np.where(freq_label == max(freq_label[:, 1]))[0], 0][0]


    ##若A(特征集)=空集， 则T为单节点树，并将D中实例数最大的类Ck作为该节点的类标记，并返回T
    if len(features) == 0:
        print(strRoot, strRootAttri, freq_label[np.where(freq_label == max(freq_label[:, 1]))[0], 0][0], max(freq_label[:, 1]))
        return freq_label[np.where(freq_label == max(freq_label[:, 1]))[0], 0][0]

    ##计算得到最优划分特征best_feature以及以最优划分特征划分的数据集split_data以及相应的信息增益max_value
    max_value, best_feature, split_data, best_split_point= choose_best_feature(dataset, features, label)


    ##如果信息增益小于设定的阈值alpha，则返回T
    if max_value < alpha:
        print(strRoot, strRootAttri, freq_label[np.where(freq_label == max(freq_label[:, 1]))[0], 0][0], max(freq_label[:, 1]))
        return freq_label[np.where(freq_label == max(freq_label[:, 1]))[0], 0][0]

    myTree = {best_feature: {}}  # 这里很巧妙，以此来创建嵌套字典

    ##否则对于每一个最优划分特征的特征值，得到相应的子数据集进行递归
    for feature in split_data.keys():
        # print(feature)
        # print(split_data[feature])
        ##当前节点所包含的样本集合为空时，把父节点的样本分布作为当前节点的先验分布
        if len(split_data[feature]) == 0:
            print(strRoot, strRootAttri, freq_label[np.where(freq_label == max(freq_label[:, 1]))[0], 0][0],
                  max(freq_label[:, 1]))
            return freq_label[np.where(freq_label == max(freq_label[:, 1]))[0], 0][0]
        else:
            print(strRoot, strRootAttri, best_feature, feature)
            next_features = [a for a in features if a != best_feature]
            # print(next_features)
            myTree[best_feature][feature] = creat_C45tree(split_data[feature], next_features, label, alpha, strRoot=best_feature, strRootAttri=feature)
    return myTree





if __name__ == '__main__':
    df_data = pd.read_table('xigua.txt', sep=',', encoding='gbk')
    dataset = df_data.drop(columns=['编号'])
    # max_value, best_feature, split_data, best_split_point = choose_best_feature(dataset, dataset.columns, '好瓜')
    # print(max_value, best_feature, split_data, best_split_point)
    my_tree = creat_C45tree(dataset, dataset.columns, '好瓜', alpha = 0.001, strRoot="-", strRootAttri="-")
    print(my_tree)