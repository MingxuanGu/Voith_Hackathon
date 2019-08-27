# -*- coding: utf-8 -*-

import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def loadcsv(filename):

    tmp = np.loadtxt(filename, dtype=np.str, delimiter=",")
    data = tmp[1:, 2:].astype(np.double)
    return data


if __name__ == '__main__':
        filename = "data/assembly_training.csv"
        data = loadcsv(filename)
        print("data shape : ", data.shape)

        my_pca = PCA(n_components=2)
        my_pca.fit(data)
        print("n_components :", my_pca.n_components_)
        print("ratio : ", my_pca.explained_variance_ratio_)

        reduced_data = my_pca.transform(data)
        print("reduced data shape : ", reduced_data.shape)

        plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
        plt.show()

        my_pca = PCA(n_components=3)
        my_pca.fit(data)
        print("n_components :", my_pca.n_components_)
        print("ratio : ", my_pca.explained_variance_ratio_)

        reduced_data = my_pca.transform(data)
        print("reduced data shape : ", reduced_data.shape)

        fig = plt.figure()
        ax = Axes3D(fig)

        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2])
        plt.show()