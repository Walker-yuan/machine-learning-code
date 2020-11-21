import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


def perceptron(X, y, learning_rate):
    """
function: 感知机模型， f(x) = sign(w*x + b)
    :param X: ndarray, n*m, 其中n为数据个数， m为特征个数
    :param y: naarray, n*1, 数据标签
    :param learning_rate: float,  0 < learning_rate <= 1, 学习率， 控制学习的速度
    :return: w, b, 模型f(x)中的超参数
    """
    w = np.zeros(X.shape[1])
    b = 0
    num_data = len(X)
    wrong = False
    while not wrong:
        count_wrong = 0
        for i in range(num_data):
            if y[i] * (X[i].dot(w) + b) <= 0:
                w += learning_rate * y[i] * X[i]
                b += learning_rate * y[i]
                count_wrong += 1
        print(count_wrong)
        if count_wrong == 0:
            wrong = True
    params = {
        'w': w,
        'b':b
    }
    return params


def perceptron_dual(X, y, learning_rate):
    """
function: 实现感知机的对偶形式
    :param X: ndarray, n*m, 其中n为数据个数， m为特征个数
    :param y: naarray, n*1, 数据标签
    :param learning_rate: float,  0 < learning_rate <= 1, 学习率， 控制学习的速度
    :return: w, b, 模型f(x)中的超参数
    """
    Gram_mat = np.dot(X, X.T)
    num_data, num_fea = X.shape[0], X.shape[1]
    alpha, b = np.zeros(num_data), 0
    while 1:
        count_wrong = 0
        for i in range(num_data):
            alpha_y = y * alpha
            if y[i] * (np.sum(alpha_y * Gram_mat[i]) + b) <= 0:
                alpha[i] += learning_rate
                b += learning_rate * y[i]
                count_wrong += 1
        if count_wrong == 0:
            break
    # print(y.shape)
    # print(alpha.shape)
    # print(np.tile((y * alpha).reshape(-1, 1), [1, num_data]).shape)
    w = (np.tile((y * alpha).reshape(-1, 1), [1, num_fea]) * X).sum(axis=0)
    params = {
        'w': w,
        'b': b
    }
    return params


if __name__ == '__main__':

    ## 数据准备
    iris = load_iris()
    dataset_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
    dataset_iris['label'] = iris.target

    df = np.array(dataset_iris)
    df_X = df[:100, [0, 1]]
    df_Y = df[:100, -1]
    df_y = np.array([1 if i == 1 else -1 for i in df_Y])


    ## 数据可视化
    plt.scatter(df_X[:50, 0], df_X[:50, 1], color = 'red', label = '0')
    plt.scatter(df_X[50:, 0], df_X[50:, 1], color = 'green', label = '1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()


    ## 训练参数
    params = perceptron_dual(df_X, df_y, learning_rate=0.01)
    print(params)
    w1 = params['w'][0]
    w2 = params['w'][1]
    b = params['b']

    ## 在图中画出分界线
    X1 = np.linspace(4, 7, 100)
    X2 = -w1*X1/w2 - b/w2
    plt.plot(X1, X2)
    plt.show()