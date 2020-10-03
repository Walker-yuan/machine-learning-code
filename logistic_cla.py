# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:34:34 2020

@author: YRH
"""
import numpy as np
from sklearn.utils import shuffle
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_classification


class logistic_model():
    def __init__(self):
        pass
    
    def prepare_data(self):
        data, target = make_classification(n_samples = 200, n_features = 2, n_redundant = 0, n_informative = 2, n_classes = 2)
        X, y = shuffle(data, target, random_state = 42)
        X = X.astype(np.float32)
        y = y.reshape(-1, 1)
        data = np.concatenate((X, y), axis = 1)
        return data
 
    def sigmoid(self, X):
        y = 1 / (1 + np.exp(-X))
        return y
    
    
    def initialize_params(self, dims):
        w = np.zeros((dims, 1))
        b = 0
        return w, b
    
    
    def logistic_loss(self, X, y, w, b):
        num_train = X.shape[1]
        y_bar = np.dot(X, w) + b
        y_hat = self.sigmoid(y_bar)
        cost = -1/num_train *np.sum(y * np.log(y_hat) + (1-y)*np.log(1-y_hat))
        dw = np.dot(X.T, y_hat - y)/num_train
        db = np.sum(y_hat - y)/num_train
        return cost, dw, db
    
    
    def logistic_train(self, X, y, learning_rate, epochs):
        num_feature = X.shape[1]
        w, b = self.initialize_params(num_feature)
        for i in range(epochs):
            cost, dw, db = self.logistic_loss(X, y, w, b)
            w = w - learning_rate * dw
            b = b - learning_rate * db
            if i%1000 == 0:
                print("epoch:%d, cost:%f "%(i, cost))
        params = {'w': w, 'b': b}
        grads = {'dw': dw, 'db': db}
        return cost, params, grads
    
    def logistic_predict(self, X, params):
        w = params['w']
        b = params['b']
        y_predict = self.sigmoid(np.dot(X, w) + b)
        for i in range(len(y_predict)):
            if y_predict[i] >= 0.5:
                y_predict[i] = 1
            else:
                y_predict[i] = 0
        return y_predict

    def logistic_acc(self, y_test, y_predict):
        acc = np.sum(y_test == y_predict) / len(y_test)
        return acc
    
    def logistic_score(self, y_valid, y_predict):
        score = np.sum((y_valid - y_predict)**2)/ len(y_valid)
        return score
    
    def logistic_R2(self, y_valid, y_predict):
        R2 = 1 - (np.sum((y_predict - y_valid)**2)) / (np.sum((y_valid - y_valid.mean())**2))
        return R2
    
 
    def logistic_cross_validation(self, data, K, randomize = True):
        if randomize:
            data = list(data)
            shuffle(data)
        slices = [data[i::K] for i in range(K)]
        for i in range(K):
            validation = slices[i]
            train = [data for s in slices if s is not validation for data in s]
            train = np.array(train)
            validation = np.array(validation)
            yield train, validation #yield是一个生成器，每次迭代返回后面的元素
    def logistic_2classification_plot(self, X_train, y_train, params):
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
        x = np.arange(-5, 5, 0.1)
        y = (0.5 - params['b'] - params['w'][0] * x) / params['w'][1]
        ax.plot(x, y)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()


if __name__ == '__main__' :
    learning_rate = 0.001
    epochs = 100000
    K = 5  
    
    lr = logistic_model()
    data = lr.prepare_data()
 
    acc_list = []
    for train, validation in lr.logistic_cross_validation(data, K):
        X_train = train[:, :2]
        y_train = train[:, -1].reshape((-1, 1))
        X_valid = validation[:, :2]
        y_valid = validation[:, -1].reshape((-1, 1))
        loss, params, grads = lr.logistic_train(X_train, y_train, learning_rate, epochs)
        ytrain_predict = lr.logistic_predict(X_train, params)
        acc_train = lr.logistic_acc(y_train, ytrain_predict)
        acc_list.append(acc_train)
acc_list_mean = np.mean(acc_list)
print(" %d 次交叉验证的训练集上的平均精度为：%f"%(K, acc_list_mean))
ytest_predict = lr.logistic_predict(X_valid, params)
acc_test = lr.logistic_acc(y_valid, ytest_predict)
R2_test = lr.logistic_R2(y_valid, ytest_predict)
lr.logistic_2classification_plot(X_train, y_train, params)
print("测试集上的精度为：%f"%(acc_test))
print("测试集上的拟合优度R2为：%f"%(R2_test))

    
    