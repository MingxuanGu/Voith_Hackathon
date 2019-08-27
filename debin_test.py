# -*- coding: utf-8 -*-

import numpy as np
import csv
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import KernelPCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
<<<<<<< HEAD
import csv
=======
import pickle
>>>>>>> 33d8327dc20b33bb05b9a2a2dcb8a7df80ee1c77


def loadcsv(filename,startx,starty):

    tmp = np.loadtxt(filename, dtype=np.str, delimiter=",")
    data = tmp[startx:, starty:].astype(np.double)
    return data


if __name__ == '__main__':
        filename = "data/assembly_training.csv"
        data = loadcsv(filename,1,2)
        print("data shape : ", data.shape)
        
        '''
        #For loading model file
        with open('my_pca.pkl', 'rb') as fid:
            my_pca = pickle.load(fid)
        '''
        my_pca = PCA(n_components=2)
        my_pca.fit(data)
        print("n_components :", my_pca.n_components_)
        print("ratio : ", my_pca.explained_variance_ratio_)

        reduced_data = my_pca.transform(data)
        print("reduced data shape : ", reduced_data.shape)

        plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
        plt.show()

        my_pca = KernelPCA(n_components=3, kernel='rbf')
        my_pca.fit(data)
        #print("n_components :", my_pca.n_components_)
        #print("ratio : ", my_pca.explained_variance_ratio_)

        reduced_data = my_pca.transform(data)
        print("reduced data shape : ", reduced_data.shape)

        fig = plt.figure()
        ax = Axes3D(fig)

        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2])
        plt.show()


        filename_cert = "data/neg.csv"
        sample_pos = loadcsv(filename_cert,0,0)
        print("pos shape :", sample_pos.shape)
        filename_defkt = "data/pos.csv"
        sample_neg = loadcsv(filename_defkt,0,0)
        print("neg shape :", sample_neg.shape)

        label_data = np.vstack((sample_pos,sample_neg))
        label = np.vstack((np.ones(sample_pos.shape[0]).reshape((-1, 1)),np.zeros(sample_neg.shape[0]).reshape((-1, 1))))
        print("label data shape:", label_data.shape)
        print("label shape:", label.shape)
        my_lda = LinearDiscriminantAnalysis(n_components=3)
        my_lda.fit(label_data, label)
        print("n components:", my_lda.n_components)
        reduced_pos = my_lda.transform(sample_pos)
        reduced_neg = my_lda.transform(sample_neg)
        print("reduced_pos shape:", reduced_pos.shape)
        
        #Now we separate points into different group according to their year
        data_transformed = my_pca.transform(data)
        #For saving model in file
        with open('my_pca.pkl', 'wb') as fid:
            pickle.dump(my_pca, fid)
        print(data_transformed.shape)
        figInYear = plt.figure()
        axInYear = Axes3D(figInYear)

        with open(filename, newline='') as csvfile:
            allSampleList = list(csv.reader(csvfile, delimiter=',', quotechar='|'))
        for i in range(len(allSampleList)-1):
            if allSampleList[i][0][0] is 'N':
                axInYear.scatter([data_transformed[i, 0]], data_transformed[i, 1], data_transformed[i, 2], c = 'lawngreen')
                
            elif allSampleList[i][0][3] is '4':
                axInYear.scatter(data_transformed[i, 0], data_transformed[i, 1], data_transformed[i, 2], c = 'deepskyblue', marker = 'v', s=10)
            elif allSampleList[i][0][3] is '5':
                axInYear.scatter(data_transformed[i, 0], data_transformed[i, 1], data_transformed[i, 2], c = 'lightgreen', marker = 'x')
            elif allSampleList[i][0][3] is '6':
                axInYear.scatter(data_transformed[i, 0], data_transformed[i, 1], data_transformed[i, 2], c = 'deepskyblue')
            elif allSampleList[i][0][3] is '7':
                axInYear.scatter(data_transformed[i, 0], data_transformed[i, 1], data_transformed[i, 2], c = 'lightsalmon')
            elif allSampleList[i][0][3] is '8':
                axInYear.scatter(data_transformed[i, 0], data_transformed[i, 1], data_transformed[i, 2], c = 'navy')
            else:
                axInYear.scatter(data_transformed[i, 0], data_transformed[i, 1], data_transformed[i, 2], c = 'lawngreen')
        plt.show()

        '''
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(reduced_pos[:, 0], reduced_pos[:, 1], reduced_pos[:, 2], marker='o', c='blue')
        ax.scatter(reduced_neg[:, 0], reduced_neg[:, 1], reduced_neg[:, 2], marker='x', c='crimson')
        plt.show()
        '''

        plt.scatter(reduced_pos,np.zeros(reduced_pos.shape[0]).reshape((-1,1)))
        plt.scatter(reduced_neg, np.zeros(reduced_neg.shape[0]).reshape((-1, 1)),marker='x',c = 'crimson')
        plt.show()
        print("End")
