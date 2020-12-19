import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.datasets.samples_generator import make_classification

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def mul_Norm(x, mu, sigma):
    var = multivariate_normal(mean=mu, cov=sigma)
    return var.pdf(x)


def post_prob(x, parmas):
    num_cla = parmas['alpha'].shape[0]
    pre_x_prob = []
    for i in range(num_cla):
        prob = mul_Norm(x, parmas['mu'][i, :], parmas['sigma'][i, :])
        pre_x_prob.append(prob)
    prob_sum = np.sum(parmas['alpha']*np.array(pre_x_prob))
    x_post_prob = np.array([parmas['alpha'][i]*pre_x_prob[i]/prob_sum for i in range(num_cla)])
    return x_post_prob



def initial_params(k, num_fea):
    params = {
        'alpha': np.ones(k)/k,
        'mu':np.random.rand(k, num_fea),
        'sigma':np.tile(np.diag(np.ones(num_fea)), (k, 1)).reshape(k, num_fea, num_fea)
    }
    return params

def plot_cla(X_after_cla):
    #画出聚类后的效果图，针对特征数为2的数据
    num_cla = len(X_after_cla)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(num_cla):
        ax.scatter(X_after_cla[i][:, 0], X_after_cla[i][:, 1])
    plt.show()

def Gauss_cla_train(X, k, epoches):
    num_X = X.shape[0]
    num_fea = X.shape[1]
    parmas = initial_params(k, num_fea)
    cla_mat = np.zeros((num_X, k))
    for epoch in range(epoches):
        ## E步
        for j in range(num_X):
                cla_mat[j, :] = post_prob(X[j], parmas)
        ## M步
        for i in range(k):
            parmas['alpha'][i] = np.mean(cla_mat[:, i])
            # print(X.shape)
            # print(np.tile(cla_mat[:, i].reshape(-1, 1), (1, num_fea)).shape)
            mui = np.sum(X*np.tile(cla_mat[:, i].reshape(-1, 1), (1, num_fea)), axis=0)/np.sum(cla_mat[:, i])
            parmas['mu'][i] = mui
            sigmai = np.zeros((num_fea, num_fea))
            for j in range(num_X):
                sigmai += cla_mat[j, i] * np.dot((X[j, :]-mui).reshape(-1, 1), (X[j, :]-mui).reshape(-1, 1).T)
            sigmai = sigmai/np.sum(cla_mat[:, i])
            parmas['sigma'][i] = sigmai
        X_cla = []
        for j in range(num_X):
            X_cla.append(np.argmax(cla_mat[j, :]))
        if (epoch+1) % 10 == 0:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for i in set(X_cla):
                ax.scatter(X[np.where(np.array(X_cla) == i)][:, 0], X[np.where(np.array(X_cla) == i)][:, 1])
            plt.xlabel('X1')
            plt.ylabel('X2')
            plt.title('高斯混合聚类：第%d次迭代后聚类结果'%(epoch+1))
            plt.show()
    return X_cla







if __name__ == '__main__':
    # parmas = initial_params(3, 2)
    # print(parmas)
    # x_post_prob = post_prob([1, 1], parmas)
    # print(x_post_prob)

    # df_data = pd.read_table('xigua.txt', sep=',', encoding='gbk')
    # dataset = df_data.drop(columns=['编号'])
    # X = dataset[['密度', '含糖率']].values
    # X_cla = Gauss_cla_train(X, 3, 100)
    # print(X_cla)

    data, target = make_classification(
        n_samples=200, n_features=2, n_redundant=0, n_informative=2, n_classes=2)
    X, y = shuffle(data, target, random_state=42)
    X = X.astype(np.float32)
    y = y.reshape(-1, 1)

    X_cla = Gauss_cla_train(X, 3, 100)
    print(X_cla)