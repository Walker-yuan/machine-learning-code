import numpy as np
import sklearn


def cross_validation(X, y, K):
    """
function: 实现数据的交叉验证
input:
    :param X: array, 输入的特征数据集
    :param y: array， 数据集对应的标签集
    :param K: int, 交叉验证的折数
    :return:
    X_train_folds, dic, 每一个元素为K折交叉验证的第i次的训练集数据
    y_train_folds, dic, 每一个元素为K折交叉验证的i次训练数据集所对应的标签集
    X_test_folds, dic, 每一个元素为K折交叉验证的第i次的测试集数据
    y_test_folds, dic, 每一个元素为K折交叉验证的i次测试数据集所对应的标签集
    """
    ## 先把数据的顺序打乱
    X_shuffle, y_shuffle = sklearn.utils.shuffle(X, y)

    ## 把数据分为K份， np.array_split可以对数据进行不均等分割
    X_Kfolds = np.array_split(X_shuffle, K)
    y_Kfolds = np.array_split(y_shuffle, K)

    ## 把训练集、测试集以及相应的标签集保存到下述字典中
    X_train_folds = {i: np.array for i in range(K)}
    X_test_folds = {i: np.array for i in range(K)}
    y_train_folds = {i: np.array for i in range(K)}
    y_test_folds = {i: np.array for i in range(K)}
    for k in range(K):
        X_test_folds[k] = X_Kfolds[k]
        X_train_folds[k] = np.concatenate(X_Kfolds[:k] + X_Kfolds[k+1:], axis=0)
        y_test_folds[k] = y_Kfolds[k]
        y_train_folds[k] = np.concatenate((y_Kfolds[:k] + y_Kfolds[k+1:]), axis=0)
    return X_train_folds, y_train_folds, X_test_folds, y_test_folds


def hold_out(X, y, train_size = 0.7):
    """
function: 实现数据的留出法， 保证训练集和测试集的类别平衡
    :param X: array, 输入的特征数据集
    :param y: array, 数据集对应的标签集
    :param train_size: float, 训练数据集占样本集的比例
    :return:
    X_train, array， 训练特征数据集
    X_test, array, 测试特征数据集
    y_train, array, 训练标签集
    y_test, array, 测试标签集
    """
    ## 样本个数
    num_data = np.shape(X)[0]

    ## 打乱数据集
    X_shuffle, y_shuffle = sklearn.utils.shuffle(X, y)

    ## 训练集的索引
    train_index = []

    ## 利用循环针对数据类别把训练集的索引取出
    for i in set(y_shuffle):
        data_index = np.where(y_shuffle == i)[0]
        len_index = len(data_index)
        train_index = np.append(train_index, data_index[:int(len_index*train_size)]).astype(int).tolist()

    ## 测试集索引
    test_index = [i for i in range(num_data) if i not in train_index]

    ## 取出训练集和测试集以及相应的标签集
    X_train = X_shuffle[train_index, :]
    y_train = y_shuffle[train_index]
    X_test = X_shuffle[test_index]
    y_test = y_shuffle[test_index]
    return X_train, X_test, y_train, y_test


def bootstrapping(X, y):
    """
function: 实现数据的自助法， 通过有放回地取数据将数据集划分为训练数据集和测试数据集
    :param X: array, 输入的特征数据集
    :param y: array, 数据集对应的标签集
    :return:
    X_train, array， 训练特征数据集
    X_test, array, 测试特征数据集
    y_train, array, 训练标签集
    y_test, array, 测试标签集
    """
    ## 样本个数
    num_data = np.shape(X)[0]

    ## 打乱数据集
    X_shuffle, y_shuffle = sklearn.utils.shuffle(X, y)

    ## 训练集索引
    train_index = np.unique(np.random.choice(num_data, num_data))

    ## 测试集索引
    test_index = [i for i in range(num_data) if i not in train_index]

    ## 取出训练集和测试集以及相应的标签集
    X_train = X_shuffle[train_index, :]
    y_train = y_shuffle[train_index]
    X_test = X_shuffle[test_index]
    y_test = y_shuffle[test_index]
    return X_train, X_test, y_train, y_test




# X = np.array(range(100)).reshape(50, 2)
# y = np.concatenate((np.zeros(25), np.ones(25)))
# X_train,  X_test, y_train, y_test = bootstrapping(X, y)
# print(X_train)
# print(y_train)