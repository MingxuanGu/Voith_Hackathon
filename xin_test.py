# -*- coding: utf-8 -*-

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import KernelPCA
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def loadcsv(filename, startx, starty):

        tmp = np.loadtxt(filename, dtype=np.str, delimiter=",")
        data = tmp[startx:, starty:].astype(np.double)
        return data


def decom_pca_n2():

        filename = "data/assembly_training.csv"
        data = loadcsv(filename, 1, 2)
        print("data shape : ", data.shape)

        my_pca = PCA(n_components=2)
        my_pca.fit(data)
        print("n_components :", my_pca.n_components_)
        print("ratio : ", my_pca.explained_variance_ratio_)

        reduced_data = my_pca.transform(data)
        print("reduced data shape : ", reduced_data.shape)

        #plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
        #plt.show()

        return my_pca


def decom_kernel_pca_n3(data):

        print("data shape : ", data.shape)
        my_pca = KernelPCA(n_components=3, kernel='rbf')
        my_pca.fit(data)
        reduced_data = my_pca.transform(data)
        print("reduced data shape : ", reduced_data.shape)

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2])
        plt.show()

        return my_pca


def read_label_data():
        #print('bp1')
        filename_cert = "data/neg.csv"
        sample_pos = loadcsv(filename_cert, 0, 0)
        #print("pos shape :", sample_pos.shape)
        filename_defkt = "data/pos.csv"
        sample_neg = loadcsv(filename_defkt, 0, 0)
        #print("neg shape :", sample_neg.shape)

        X = np.vstack((sample_pos, sample_neg))
        Y = np.vstack((np.ones(sample_pos.shape[0]).reshape((-1, 1)),
                       np.zeros(sample_neg.shape[0]).reshape((-1, 1))))
        #print('X shape: ', X.shape)
        #print('Y shape: ', Y.shape)
        return sample_pos, sample_neg, X, Y


def read_label_validation():

        filename_cert = "data/pos_validate.csv"
        validate_pos = loadcsv(filename_cert, 0, 0)
        print("pos shape :", validate_pos.shape)
        filename_defkt = "data/neg_validate.csv"
        validate_neg = loadcsv(filename_defkt, 0, 0)
        print("neg shape :", validate_neg.shape)
        X = np.vstack((validate_pos, validate_neg))
        Y = np.vstack((np.ones(validate_pos.shape[0]).reshape((-1, 1)),
                       np.zeros(validate_neg.shape[0]).reshape((-1, 1))))
        return validate_pos, validate_neg, X, Y


def read_label_testbench():
    filename_cert = "data/pos_bench_training.csv"
    data_pos = loadcsv(filename_cert, 0, 0)
    print("pos shape :", data_pos.shape)
    filename_defkt = "data/neg_bench_training.csv"
    data_neg = loadcsv(filename_defkt, 0, 0)
    print("neg shape :", data_neg.shape)
    X = np.vstack((data_pos, data_neg))
    Y = np.vstack((np.ones(data_pos.shape[0]).reshape((-1, 1)),
                   np.zeros(data_neg.shape[0]).reshape((-1, 1))))
    return data_pos, data_neg, X, Y

def read_validate_testbench():
    filename_cert = "data/pos_bench_validate.csv"
    data_pos = loadcsv(filename_cert, 0, 0)
    print("pos shape :", data_pos.shape)
    filename_defkt = "data/neg_bench_validate.csv"
    data_neg = loadcsv(filename_defkt, 0, 0)
    print("neg shape :", data_neg.shape)
    X = np.vstack((data_pos, data_neg))
    Y = np.vstack((np.ones(data_pos.shape[0]).reshape((-1, 1)),
                   np.zeros(data_neg.shape[0]).reshape((-1, 1))))
    return data_pos, data_neg, X, Y

def label_decom_kernel_pca_n3(pca_model):

        sample_pos, sample_neg, X, Y = read_label_data()
        label_data = np.vstack((sample_pos, sample_neg))
        label = np.vstack((np.ones(sample_pos.shape[0]).reshape((-1, 1)),np.zeros(sample_neg.shape[0]).reshape((-1, 1))))
        my_lda = LinearDiscriminantAnalysis(n_components=10)
        my_lda.fit(label_data, label)
        reduced_pos = pca_model.transform(sample_pos)
        reduced_neg = pca_model.transform(sample_neg)
        '''
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(reduced_pos[:, 0], reduced_pos[:, 1], reduced_pos[:, 2], marker='o', c='blue')
        ax.scatter(reduced_neg[:, 0], reduced_neg[:, 1], reduced_neg[:, 2], marker='x', c='crimson')
        plt.show()
        '''

def GMM_clustering(data, num_clusters):

        my_gmm = GaussianMixture(n_components=num_clusters,
                                 covariance_type='diag',
                                 init_params='kmeans')
        my_gmm.fit(data)
        clustered_data = my_gmm.predict(data)
        #print("clustered_data shape : ", clustered_data.shape)
        return my_gmm,clustered_data


def pre_process():
        filename = "data/assembly_training.csv"
        data = loadcsv(filename, 1, 2)
        #print("data shape : ", data.shape)

        pos_testbench, neg_testbench, X_testbench, Y_testbench = read_label_data()
        kernel_pca_n3 = decom_kernel_pca_n3(X_testbench)
        #label_decom_kernel_pca_n3(kernel_pca_n3)
        reduced_data = kernel_pca_n3.transform(X_testbench)

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], marker='o', c='blue')
        plt.show()

        pos_testbench_re = kernel_pca_n3.transform(pos_testbench)
        neg_testbench_re = kernel_pca_n3.transform(neg_testbench)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(pos_testbench_re[:, 0], pos_testbench_re[:, 1], pos_testbench_re[:, 2], marker='o', c='crimson' ,label='Problem Breaks')
        ax.scatter(neg_testbench_re[:, 0], neg_testbench_re[:, 1], neg_testbench_re[:, 2], marker='x', c='blue', label='Functional Breaks')
        ax.legend()
        plt.show()


        gmm,cluster_label = GMM_clustering(reduced_data, 3)
        #print('label max: ', np.max(clustered_data))
        #print('label min: ', np.min(clustered_data))
        
        data_c1 = reduced_data[np.where(cluster_label == 0), :][0, :, :]
        data_c2 = reduced_data[np.where(cluster_label == 1), :][0, :, :]
        data_c3 = reduced_data[np.where(cluster_label == 2), :][0, :, :]

        '''
        print('data_c1 shape: ', data_c1.shape)
        print('data_c2 shape: ', data_c2.shape)
        print('data_c3 shape: ', data_c3.shape)
        '''

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(data_c1[:, 0], data_c1[:, 1], data_c1[:, 2], marker='o', c='skyblue', label='cluster1')
        ax.scatter(data_c2[:, 0], data_c2[:, 1], data_c2[:, 2], marker='x', c='crimson', label='cluster2')
        ax.scatter(data_c3[:, 0], data_c3[:, 1], data_c3[:, 2], marker='v', c='olive', label='cluster3')
        plt.legend()
        plt.show()

        
        return gmm, kernel_pca_n3


def merge_dataset(sample_pos, sample_neg, pos_label = 1, neg_label = 0):
        sample_merge = np.vstack((sample_pos,
                                  sample_neg))
        label_merge = np.vstack((pos_label * np.ones(sample_pos.shape[0]).reshape((-1,1)),
                                 neg_label * np.ones(sample_neg.shape[0]).reshape((-1,1))))
        return sample_merge, label_merge


def rf_assemble():

        assemble_pos, assemble_neg, assemble_X, assemble_Y = read_label_data()
        k_pca = KernelPCA(n_components=3, kernel='rbf')
        k_pca.fit(assemble_X)
        assemble_X_reduced = k_pca.transform(assemble_X)
        rfr = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight={0:80})
        rfr.fit(assemble_X_reduced, assemble_Y)

        validate_pos, validate_neg, validate_X, validate_Y = read_label_validation()
        validate_X_reduced = k_pca.transform(validate_X)
        rfr_res = rfr.predict(validate_X_reduced)
        print('precision : ', precision_score(rfr_res, validate_Y))
        validate_neg_reduced = k_pca.transform(validate_neg)
        print('miss classified : ', np.sum(np.abs(rfr.predict(validate_neg_reduced))))


def rf_test():

        test_pos, test_neg, test_X, test_Y = read_label_testbench()
        k_pca = KernelPCA(n_components=20, kernel='rbf')
        k_pca.fit(test_X)
        test_X_reduced = k_pca.transform(test_X)
        rfr = RandomForestClassifier(n_estimators=400, max_depth=8, class_weight={0: 90})
        rfr.fit(test_X_reduced, test_Y)
        rfr_res = rfr.predict(test_X_reduced)
        print('precision : ', precision_score(rfr_res, test_Y))
        test_neg_reduced = k_pca.transform(test_neg)
        print('miss classified : ', np.sum(np.abs(rfr.predict(test_neg_reduced))))


        validate_pos, validate_neg, validate_X, validate_Y = read_validate_testbench()
        print("test :   ", validate_X.shape)
        validate_X_reduced = k_pca.transform(validate_X)
        rfr_res = rfr.predict(validate_X_reduced)
        print('precision : ', precision_score(rfr_res, validate_Y))
        validate_neg_reduced = k_pca.transform(validate_neg)
        print('miss classified : ', np.sum(np.abs(rfr.predict(validate_neg_reduced))))


if __name__ == '__main__':


        pre_process()
        #rf_test()
        #gmm, kernel_pca = pre_process()
        '''
        sample_pos, sample_neg, X, Y = read_label_data()
        sample_pos_reduced = kernel_pca.transform(sample_pos)
        sample_neg_reduced = kernel_pca.transform(sample_neg)
        X_reduced = kernel_pca.transform(X)
        sample_pos_cluster = gmm.predict(sample_pos_reduced)
        sample_neg_cluster = gmm.predict(sample_neg_reduced)

        pos_c1 = sample_pos_reduced[np.where(sample_pos_cluster == 0), :][0, :, :]
        neg_c1 = sample_neg_reduced[np.where(sample_neg_cluster == 0), :][0, :, :]
        X_c1, Y_c1 = merge_dataset(pos_c1, neg_c1)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(pos_c1[:, 0], pos_c1[:, 1], pos_c1[:, 2], marker='o', c='skyblue')
        ax.scatter(neg_c1[:, 0], neg_c1[:, 1], neg_c1[:, 2], marker='x', c='crimson')
        plt.show()

        my_svm = SVC(kernel='linear', class_weight={0:5})
        my_svm.fit(X_c1, Y_c1)
        print('score : ', my_svm.score(X_c1, Y_c1))
        svm_label = my_svm.predict(X_c1).reshape((-1, 1))
        print('svm label shape : ', svm_label.shape)
        print('Y_c1 shape : ', Y_c1.shape)
        print('min label: ', np.min(svm_label))
        print('max label: ', np.max(svm_label))
        print('miss classified : ', np.sum(np.abs(svm_label - Y_c1)))


        validate_pos, validate_neg, validate_X, validate_Y = read_label_validation()
        validate_X_reduced = kernel_pca.transform(validate_X)
        validate_cluster = gmm.predict(validate_X_reduced)
        validate_c1 = validate_X_reduced[np.where(validate_cluster == 0), :][0, :, :]
        validate_label_c1 = validate_Y[np.where(validate_cluster == 0), :][0, :, :]
        validate_label = my_svm.predict(validate_c1).reshape((-1, 1))
        print('min label: ', np.min(validate_label))
        print('validate score : ', my_svm.score(validate_c1, validate_label_c1))
        print('miss classified : ', np.sum(np.abs(validate_label - validate_label_c1)))
        print('f1 score : ', f1_score(validate_label_c1, validate_label))
        print('precision score : ', precision_score(validate_label_c1, validate_label))
        #print('coef : ', my_svm.coef_)
        '''
