import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
import pandas as pd
from perfermance_measure import accuracy

def Norm(x, mu, sigma):
    return 1 / (np.sqrt(2*np.pi) * float(sigma)) * np.exp(-(float(x)-float(mu))**2/(2*float(sigma)**2))

class Naive_bayes():
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.classes = y.unique()
        self.class_count = y.value_counts()
        self.class_prior = y.value_counts()/len(y)
        self.prior = {}
    def Naive_bayes_train(self):
        for col in self.X.columns:
            if type(self.X[col].tolist()[0]).__name__ in ['int', 'int32', 'int64', 'float', 'float32', 'float64']:
                for j in self.classes:
                    # print(self.X[(self.y==j)][col].shape)
                    mu = np.mean(self.X[(self.y==j)][col])
                    sigma = np.std(self.X[(self.y==j)][col])
                    # print(mu, sigma)
                    self.prior[(col, '连续型', j)] = [mu, sigma]
            else:
                for j in self.classes:
                    p_xy = self.X[(self.y==j)][col].value_counts()
                    for i in p_xy.index:
                        self.prior[(col, i, j)] = p_xy[i]/self.class_count[j]
        return self.classes, self.class_prior, self.prior

    def pre_predict(self, X_test):
        pre_score = []
        for c in self.classes:
            p_xy = 1
            for col in X_test.index:
                if type(X_test[col]).__name__ in ['int', 'int32', 'int64', 'float', 'float32', 'float64']:
                    vec = self.prior[tuple([col]+['连续型']+[c])]
                    p_x_y = Norm(X_test[col], vec[0], vec[1])
                else:
                    p_x_y = self.prior[tuple([col]+[X_test[col]]+[c])]
                p_xy *= p_x_y
            pre_score.append(self.class_prior[c]*p_xy)
        return self.classes[np.argmax(pre_score)]

    def predict(self, X_test):
        num_test = X_test.shape[0]
        predict_label = []
        for i in range(num_test):
            label = self.pre_predict(X_test.iloc[i, :])
            predict_label.append(label)
        return predict_label







if __name__ == '__main__':

    ## iris数据集
    iris = load_iris()
    columns = ['sepal length', 'sepal width', 'petal length', 'petal width']
    X = pd.DataFrame(iris.data, columns=columns)
    y = pd.Series(iris.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


    ## 西瓜数据集
    # df_data = pd.read_table('xigua.txt', sep=',', encoding='gbk')
    # dataset = df_data.drop(columns=['编号'])
    # X = dataset.iloc[:11, :-1]
    # y = dataset.iloc[:11, -1]
    # X_test = dataset.iloc[11:, :-1]
    # y_test = dataset.iloc[11:, -1]

    naive_bayes = Naive_bayes(X_train, y_train)
    classes, class_prior, prior = naive_bayes.Naive_bayes_train()
    y_predict = naive_bayes.predict(X_test)
    acc = accuracy(y_test.values, y_predict)
    print(acc)
    print(classes, class_prior, prior)



    # def Naive_bayes_predict(self, X_test):

        




