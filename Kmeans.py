# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 18:47:22 2020

@author: YRH
"""
import numpy as np
from sklearn.utils import shuffle
from sklearn.datasets.samples_generator import make_classification
import matplotlib.pyplot as plt


class Kmeans():
    def __init__(self):
        pass

    def prepare_data(self):
        data, target = make_classification(
            n_samples=200, n_features=2, n_redundant=0, n_informative=2, n_classes=2)
        X, y = shuffle(data, target, random_state=42)
        X = X.astype(np.float32)
        y = y.reshape(-1, 1)
        data = np.concatenate((X, y), axis=1)
        return data

    def initialize_center(self, X, K):
        num_X = X.shape[0]
        choose_index = np.random.permutation(
            range(num_X))[0:K]  # 从0到(num_X-1)中随机选取K个数
        ##可以使用index = random.sample(list(range(len(dataset))), k)
        initial_center = X[choose_index, :]
        return initial_center

    def compute_distances(self, X, center):
        num_X = X.shape[0]
        num_center = center.shape[0]
        distance_matrix = np.zeros((num_X, num_center))
        for i in range(num_X):
            for j in range(num_center):
                distance_matrix[i, j] = np.sqrt(
                    np.sum((X[i, :] - center[j, :])**2))
        return distance_matrix

    def get_index1(self, lst, item):
        return [index for (index, value) in enumerate(lst) if value == item]

    def new_center(self, distance_matrix, X):
        X_pre_classification = []
        new_center = []
        X_after_cla = []
        for i in range(distance_matrix.shape[0]):
            X_pre_classification.append(distance_matrix[i, :].argmin())
        for j in range(distance_matrix.shape[1]):
            new_center.append(
                np.mean(X[self.get_index1(X_pre_classification, j), :], axis=0))
            X_after_cla.append(X[self.get_index1(X_pre_classification, j), :])
        return X_after_cla, X_pre_classification, np.array(new_center)

    def update_center(self, X, K, epochs):
        initial_center = self.initialize_center(X, K)
        next_center = initial_center
        for i in range(epochs):
            last_center = next_center
            distance_matrix = self.compute_distances(X, last_center)
            X_after_cla, X_pre_classification, next_center = self.new_center(
                distance_matrix, X)
            print('epochs:%d' % (i))
            if (next_center == last_center).all():
                break
            else:
                pass
        return X_after_cla, X_pre_classification, next_center

    def outsise_index_evaluate(self, y, X_pre_classification):
        a, b, c, d = 0, 0, 0, 0
        for i in range(len(y)):
            for j in range(i + 1, len(y)):
                if y[j] == y[i] and X_pre_classification[j] == X_pre_classification[i]:
                    a += 1
                elif y[j] != y[i] and X_pre_classification[j] == X_pre_classification[i]:
                    b += 1
                elif y[j] == y[i] and X_pre_classification[j] != X_pre_classification[i]:
                    c += 1
                else:
                    d += 1
        JC_coefficient = a / (a + b + c)
        FM_index = np.sqrt((a / (a + b)) * (a / (a + c)))
        Rand_index = 2 * (a + d) / (len(y) * (len(y) - 1))
        return JC_coefficient, FM_index, Rand_index

    def inside_index_evaluate(self, X_after_cla):
        avg_distance_cla = []
        diam_distance_cla = []
        dmin_distance_cla = np.zeros((len(X_after_cla), len(X_after_cla)))
        for i in range(len(X_after_cla)):
            distance_cla = self.compute_distances(
                X_after_cla[i], X_after_cla[i])
            avg_distance_cla.append(
                distance_cla.sum() / (X_after_cla[i].shape[0] * (X_after_cla[i].shape[0] - 1)))
            diam_distance_cla.append(distance_cla.max())
            for j in range(len(X_after_cla)):
                if j == i:
                    dmin_distance_cla[i, j] = 0
                else:
                    cla_to_cla_distance = self.compute_distances(
                        X_after_cla[i], X_after_cla[j])
                    dmin_distance_cla[i, j] = cla_to_cla_distance.min()
        center = np.array([np.mean(X_after_cla[i], axis=0)
                           for i in range(len(X_after_cla))])
        distance_center = self.compute_distances(center, center)

        DBI_list = np.zeros((len(X_after_cla), len(X_after_cla)))
        Dunn_list = np.zeros((len(X_after_cla), len(X_after_cla)))
        for i in range(len(X_after_cla)):
            for j in range(len(X_after_cla)):
                if j == i:
                    DBI_list[i, j] = 0
                    Dunn_list[i, j] = 0
                else:
                    DBI_list[i, j] = (
                        avg_distance_cla[i] + avg_distance_cla[j]) / distance_center[i, j]
                    Dunn_list[i, j] = dmin_distance_cla[i, j] / \
                        np.array(diam_distance_cla).max()
        DBI_index = np.mean(np.max(DBI_list, axis=0))
        Dunn_list[range(Dunn_list.shape[0]),
                  range(Dunn_list.shape[0])] = np.inf
        Dunn_index = np.min(np.min(Dunn_list, axis=0))
        return DBI_index, Dunn_index

    def plot_cla(self, X_after_cla):
        num_cla = len(X_after_cla)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(num_cla):
            ax.scatter(X_after_cla[i][:, 0], X_after_cla[i][:, 1])
        plt.show()



if __name__ == '__main__':
    K = 8
    epochs = 10000

    cla = Kmeans()
    data = cla.prepare_data()
    X = data[:, :-1]
    y = data[:, -1]
    X_after_cla, X_pre_classification, center = cla.update_center(X, K, epochs)
    JC_coefficient, FM_index, Rand_index = cla.outsise_index_evaluate(
        y, X_pre_classification)
    DBI_index, Dunn_index = cla.inside_index_evaluate(X_after_cla)
    cla.plot_cla(X_after_cla)
    print(JC_coefficient, FM_index, Rand_index)
    print(DBI_index, Dunn_index)
