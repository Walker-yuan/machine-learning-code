import pandas as pd
import numpy as np



df_data = pd.read_table('xigua.txt', sep=',', encoding='gbk')
dataset = df_data.drop(columns=['编号', '含糖率', '密度'])

def Gini_index(cla):
    """
function:计算基尼指数(基尼指数越小, 代表着数据集的纯度越高)
input:
    :param cla: series, 表示数据类别的系列
    :return:gini_index, 基尼指数
    """
    probs = cla.value_counts()/len(cla)
    gini_index = sum([prob*(1-prob) for prob in probs])
    return gini_index

# print(Gini_index(dataset['好瓜']))



def Split_data(dataset, feature, split_point):
    """
function:根据特征feature划分数据集dataset
input:
    :param dataset: dataframe, 需要划分的数据集
    :param feature: str, 用于划分数据集的特征
    :split_point : float or str, 连续属性的划分点
output:
    :split_data: dic, 划分后的数据集
    """
    if type(split_point).__name__ == 'str':
        split_data = {split_point: dataset[dataset[feature] == split_point],
                         'others': dataset[dataset[feature] != split_point]}
    else:
        split_data = {'> %f'%(split_point): dataset[:][dataset[feature] > split_point],
                         '<= %f'%(split_point): dataset[:][dataset[feature] <= split_point]}
    return  split_data

# print(Split_data(dataset, '含糖率', 0.25))


def Mean_square(dataset, feature, label, split_point):
    """
function: 计算特征feature基于划分点split_point的基尼指数
    :param dataset: dataframe, 输入的数据集
    :param feature: str, 需要计算基尼指数的特征
    :param label: str, 类别所对应的列名
    :param split_point: float or str, 特征feature的划分点
    :return: pre_label， float, 特征feature基于划分点split_point的基尼指数
    """
    num_data = len(dataset)
    split_data = Split_data(dataset, feature, split_point)
    mean_square = 0
    for key in split_data.keys():
        mean_square += np.sum((split_data[key][label] - np.mean(split_data[key][label]))**2)
    return mean_square



def Gini_index_DA(dataset, feature, label, split_point):
    """
function: 计算特征feature基于划分点split_point的基尼指数
    :param dataset: dataframe, 输入的数据集
    :param feature: str, 需要计算基尼指数的特征
    :param label: str, 类别所对应的列名
    :param split_point: float or str, 特征feature的划分点
    :return: Gini_index_DA， float, 特征feature基于划分点split_point的基尼指数
    """
    num_data = len(dataset)
    split_data = Split_data(dataset, feature, split_point)
    gini_index_DA = 0
    for key in split_data.keys():
        gini_index_DA += len(split_data[key])/num_data * Gini_index(split_data[key][label])
    return gini_index_DA

# print(Gini_index_DA(dataset, '含糖率', '好瓜', 0.25))



def choose_best_split(dataset, feature, label, typeof = 'classification'):
    """
function:基于基尼指数最小从特征feature中选择最优的划分点
input:
    :param dataset: dataframe, 输入的数据集
    :param feature: Index, 特征的集合
    :param label: str, 类别对应的列名
    :param typeof: str, default: 'classification', 选择是做回归还是做分类
output:
    :min_value: float, 特征feature基于最优划分点的基尼指数或者均方误差
    :best_split: float or str, 特征feature的最优划分点
    """
    best_split = None
    min_value = np.inf # 基尼指数的最大值为1, 所以这个值设的比1大就行
    if type(dataset[feature].tolist()[0]).__name__ == 'str':
        for point in set(dataset[feature]):
            if typeof == 'classification':
                gini_value = Gini_index_DA(dataset, feature, label, point)
                if gini_value < min_value:
                    min_value, best_split = gini_value, point
            else:
                mean_value = Mean_square(dataset, feature, label, point)
                if mean_value < min_value:
                    min_value, best_split = mean_value, point
    else:
        sort_feaValue = np.sort(np.unique(dataset[feature]))
        split_point_list = [np.float(sort_feaValue[i] + sort_feaValue[i + 1]) / 2.0 for i in
                            range(len(sort_feaValue) - 1)]
        for point in split_point_list:
            if typeof == 'classification':
                gini_value = Gini_index_DA(dataset, feature, label, point)
                if gini_value < min_value:
                    min_value, best_split = gini_value, point
            else:
                mean_value = Mean_square(dataset, feature, label, point)
                if mean_value < min_value:
                    min_value, best_split = mean_value, point
    return best_split, min_value

# best_split, min_value = choose_best_split(dataset, '色泽', '好瓜')
# print(best_split, min_value)



def choose_best_feature(dataset, features, label, typeof = 'classification'):
    """
function:从所有特征中以及对应特征的所有划分点中找到基尼指数最小的特征及其相应的划分点
input:
    :param dataset: dataframe, 输入的数据集
    :param features: Index, 特征的集合
    :param label: str, 类别所对应的列名
    :param typeof: str, default: 'classification', 选择是做回归还是做分类
output:
    :min_value: float, 最优特征及其相应最有划分点所对应的基尼指数或者均方误差
    :best_feature: str, 最小基尼指数或均方误差对应的特征
    :split_data: dic, 以最佳特征划分的数据集
    ：best_split: float or str, 最优特征对应的最有划分点
    """
    features = [feature for feature in features if feature != label]
    min_value = np.inf
    best_feature = None
    best_split = None
    for feature in features:
        split, feature_value= choose_best_split(dataset, feature, label, typeof)
        if feature_value < min_value:
            min_value = feature_value
            best_feature, best_split = feature, split
    split_data = Split_data(dataset, best_feature, best_split)
    return split_data, best_feature, best_split, min_value

# split_data, best_feature, best_split, min_value = choose_best_feature(dataset, dataset.columns, '好瓜')
# print(split_data,'\n', best_feature, best_split, min_value)




def creat_Carttree(dataset, features, label, sample_threshold, gini_threshold, strRoot="-", strRootAttri="-", typeof = 'classification'):
    """
function: 利用数据集dataset以及特征集features产生一颗决策树
input:
    :param dataset: dataframe, 输入的数据集
    :param features: series, 输入的特征集
    :param label: str, 类别所对应的列名
    :param sample_threshold: int, 样本数对应的阈值(节点划分所对应的最小样本数)
    :param gini_threshold: float, 划分数据集对应的最小的基尼指数，小于这个阈值，说明数据集的纯度很高
    :param strRoot:str, 根节点名称
    :param strRootAttri:str, 子节点名称
    :return:一颗决策树
注意：
（1）若D中所有实例属于同一类别Ck， 则T为单节点树，并将类Ck作为该节点的类标记，返回T
（2）若A(特征集)=空集， 则T为单节点树，并将D中实例数最大的类Ck作为该节点的类标记，并返回T
（3）当前节点所包含的样本集合为空时，把父节点的样本分布作为当前节点的先验分布
    """
    ## 去除掉类别所对应的那个列名
    features = [i for i in features if i != label]

    if typeof == 'classification':
        ## 基于类别label, 计算数据集的频率
        freq_label = dataset[label].value_counts()/len(dataset)

        ## 结束标准1：若D中所有实例属于同一类别Ck， 则T为单节点树，并将类Ck作为该节点的类标记，返回T
        if max(freq_label) == 1:
            print(strRoot, strRootAttri, freq_label.index[freq_label == max(freq_label)][0], 1)
            return freq_label.index[freq_label == max(freq_label)][0]

        ## 结束标准2：若A(特征集)=空集， 则T为单节点树，并将D中实例数最大的类Ck作为该节点的类标记，并返回T
        if len(features) == 0:
            print(strRoot, strRootAttri, freq_label.index[freq_label == max(freq_label)][0], max(freq_label))
            return freq_label.index[freq_label == max(freq_label)][0]

        ## 结束标准3：若该节点的样本个数小于给定的阈值, 将D中实例数最大的类Ck作为该节点的类标记，并返回T
        if len(dataset) <= sample_threshold:
            print(strRoot, strRootAttri, freq_label.index[freq_label == max(freq_label)][0], max(freq_label))
            return freq_label.index[freq_label == max(freq_label)][0]

        ## 结束标准4：若该数据集的基尼指数小于给定的阈值(说明该节点的数据纯度已经很高)，将D中实例数最大的类Ck作为该节点的类标记，并返回T
        gini_index_D = Gini_index(dataset[label])
        if gini_index_D <= gini_threshold:
            print(strRoot, strRootAttri, freq_label.index[freq_label == max(freq_label)][0], max(freq_label))
            return freq_label.index[freq_label == max(freq_label)][0]
    else:
        mean_label = np.mean(dataset[label])
        if len(features) == 0 or len(dataset) <= sample_threshold:
            print(strRoot, strRootAttri, label, mean_label)
            return mean_label

    ## 选择最优特征以及最优划分点
    split_data, best_feature, best_split, min_value = choose_best_feature(dataset, features, label, typeof)
    # print(best_feature, best_split, min_value,split_data)
    ## 结束标准4：若该数据集的基尼指数小于给定的阈值(说明该节点的数据纯度已经很高)，将D中实例数最大的类Ck作为该节点的类标记，并返回T
    # if min_value < gini_threshold:
    #     print(strRoot, strRootAttri, freq_label.index[freq_label == max(freq_label)][0], max(freq_label))
    #     return
    myTree = {best_feature: {}}  # 这里很巧妙，以此来创建嵌套字典

    ## 对于每一个最优划分特征的最优划分点，得到相应的子数据集进行递归
    for key in split_data.keys():

        ## 结束标准5：当前节点所包含的样本集合为空时，把父节点的样本分布作为当前节点的先验分布
        if len(split_data[key]) == 0:
            if typeof == 'classification':
                print(strRoot, strRootAttri, freq_label.index[freq_label == max(freq_label)][0], max(freq_label))
                return freq_label.index[freq_label == max(freq_label)][0]
            else:
                print(strRoot, strRootAttri, label, mean_label)
                return mean_label
        else:
            print(strRoot, strRootAttri, best_feature, key)
            next_features = [feature for feature in features if feature != best_feature]
            myTree[best_feature][key] = creat_Carttree(split_data[key], next_features, label, sample_threshold, gini_threshold, typeof = typeof, strRoot=best_feature, strRootAttri=key)
    return myTree


if __name__ == '__main__':
    df_data = pd.read_table('xigua.txt', sep=',', encoding='gbk')
    dataset = df_data.drop(columns=['编号', '好瓜'])
    # split_data, best_feature, best_split, min_value = choose_best_feature(dataset, dataset.columns, '好瓜')
    # print(split_data,'\n', best_feature, best_split, min_value)
    my_tree = creat_Carttree(dataset, dataset.columns, '密度', sample_threshold = 4, gini_threshold = 0, typeof = 'regression', strRoot="-", strRootAttri="-")
    print(my_tree)


