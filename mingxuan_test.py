# -*- coding: utf-8 -*-

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import KernelPCA
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from LinearLogisticRegression import LinearLogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from csvReader import getTextID

test_bench_path = "data/test_bench_training.csv"
def loadcsv(filename,startx,starty):

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


def decom_kernel_pca_n3(stage):
    data = []
    if stage=='assembly':
        data = loadcsv("data/assembly_training.csv", 1, 2)
    else:
        data = loadcsv("data/test_bench_training.csv", 1, 4)

    my_pca = KernelPCA(n_components=3, kernel='rbf')
    my_pca.fit(data)
    reduced_data = my_pca.transform(data)
    print("reduced data shape : ", reduced_data.shape)
    '''
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2])
    plt.show()
    '''
    return my_pca

def decom_kernel_pca_n20():

        filename = test_bench_path
        data = loadcsv(filename, 1, 4)
        print("data shape : ", data.shape)

        my_pca = KernelPCA(n_components=20, kernel='rbf')
        my_pca.fit(data)
        reduced_data = my_pca.transform(data)
        print("reduced data shape : ", reduced_data.shape)
        '''
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2])
        plt.show()
        '''
        return my_pca


def read_label_data(stage):
        sample_pos = []
        sample_neg = []
        if stage=='assembly':
            sample_pos = loadcsv("data/pos.csv", 0, 0)
            sample_neg = loadcsv("data/neg.csv", 0, 0)
        else:
            sample_pos = loadcsv("data/pos_bench_training.csv", 0, 0)
            sample_neg = loadcsv("data/neg_bench_training.csv", 0, 0)

        X = np.vstack((sample_pos, sample_neg))
        Y = np.vstack((np.ones(sample_pos.shape[0]).reshape((-1, 1)),
                       np.zeros(sample_neg.shape[0]).reshape((-1, 1))))
        return sample_pos, sample_neg, X, Y


def read_label_validation(stage):
    validate_pos = []
    validate_neg = []
    if stage == 'assembly':
        validate_pos = loadcsv("data/pos_validate.csv", 0, 0)
        validate_neg = loadcsv("data/neg_validate.csv", 0, 0)
    else:
        validate_pos = loadcsv("data/pos_bench_validate.csv", 0, 0)
        validate_neg = loadcsv("data/neg_bench_validate.csv", 0, 0)
    X = np.vstack((validate_pos, validate_neg))
    Y = np.vstack((np.ones(validate_pos.shape[0]).reshape((-1, 1)),
                   np.zeros(validate_neg.shape[0]).reshape((-1, 1))))
    return validate_pos, validate_neg, X, Y

def read_data_test(path, start, end):
    test_data = loadcsv(path, start, end)
    return test_data


def label_decom_kernel_pca_n3(pca_model, stage):

        sample_pos, sample_neg, X, Y = read_label_data(stage)
        label_data = np.vstack((sample_pos, sample_neg))
        label = np.vstack((np.ones(sample_pos.shape[0]).reshape((-1, 1)),np.zeros(sample_neg.shape[0]).reshape((-1, 1))))
        my_lda = LinearDiscriminantAnalysis(n_components=3)
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

def label_decom_kernel_pca_n20(pca_model):

        sample_pos, sample_neg, X, Y = read_label_data()
        label_data = np.vstack((sample_pos, sample_neg))
        label = np.vstack((np.ones(sample_pos.shape[0]).reshape((-1, 1)),np.zeros(sample_neg.shape[0]).reshape((-1, 1))))
        my_lda = LinearDiscriminantAnalysis(n_components=20)
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

def pre_process(stage):
    data = []
    kernel_pca_n3 = 0
    if stage == 'assembly':
        data = loadcsv("data/assembly_training.csv", 1, 2)
    else:
        data = loadcsv("data/test_bench_training.csv", 1, 4)

    kernel_pca_n3 = decom_kernel_pca_n3(stage)
    # label_decom_kernel_pca_n3(kernel_pca_n3, stage)
    reduced_data = kernel_pca_n3.transform(data)

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
    '''
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(data_c1[:, 0], data_c1[:, 1], data_c1[:, 2], marker='o', c='skyblue')
    ax.scatter(data_c2[:, 0], data_c2[:, 1], data_c2[:, 2], marker='x', c='crimson')
    ax.scatter(data_c3[:, 0], data_c3[:, 1], data_c3[:, 2], marker='v', c='olive')
    plt.show()
    '''
    return gmm, kernel_pca_n3


def merge_dataset(sample_pos, sample_neg, pos_label = 1, neg_label = 0):
        sample_merge = np.vstack((sample_pos,
                                  sample_neg))
        label_merge = np.vstack((pos_label * np.ones(sample_pos.shape[0]).reshape((-1,1)),
                                 neg_label * np.ones(sample_neg.shape[0]).reshape((-1,1))))
        return sample_merge, label_merge

class KNN():
    def __init__(self, knn, stage):
        self.knn = knn
        _, self.kernel_pca = pre_process(stage)
        sample_pos, sample_neg, X_c1, self.Y_c1 = read_label_data(stage)

        X_reduced = self.kernel_pca.transform(X_c1)

        self.knn1 = KNeighborsClassifier(n_neighbors=1, weights='uniform')
        self.y = []
        self.knn1.fit(X_reduced, self.Y_c1)
        self.knn2 = KNeighborsClassifier(n_neighbors=knn, weights='distance')
        self.knn2.fit(X_reduced, self.Y_c1)


    def predict(self, sample):
        prop = []
        X_reduced_valid = self.kernel_pca.transform(sample)
        for j in range(X_reduced_valid.shape[0]):
            kNeighbour = self.knn2.kneighbors([X_reduced_valid[j]], n_neighbors=self.knn)[1]
            if np.min(self.Y_c1[kNeighbour]) <= 0:
                self.y.append(0)
            else:
                self.y.append(1)
            prop.append(np.mean(self.Y_c1[kNeighbour]))
        y = np.array(self.y)
        self.y=[]
        return y, np.array(prop)
def writetocsv(list, path):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in list:
            writer.writerow(row)
from csvReader import getOrderedValidID
import csv
if __name__ == '__main__':
    # K1List = []
    # K2List = []
    # lossList = []
    # costList = []
    # bigList = []
    path_assembly = 'data/assembly_test.csv'
    path_bench = 'data/test_bench_test.csv'
    K1 = 16
    K2 = 8
    stage = 'assembly'
    knn_assembly = KNN(K1, stage)
    # valid_pos, valid_neg, X, Y = read_label_validation(stage)
    test_data_assembly = read_data_test(path_assembly, 1, 2)
    y_assembly, prop_assembly = knn_assembly.predict(test_data_assembly)
    # loss = np.mean(abs(y_assembly - Y))
    print("----------------Assembly---------------")
    # print('loss: ', loss)
    # false_negative = np.mean(abs(y_assembly[valid_pos.shape[0]:] - Y[valid_pos.shape[0]:]))
    # print('Mean False negative', false_negative)
    # false_positive = np.mean(abs(y_assembly[:valid_pos.shape[0]] - Y[:valid_pos.shape[0]]))
    # print('Mean False positive', false_positive)
    sample_index_to_bench = y_assembly == 0
    sample_to_bench = y_assembly[sample_index_to_bench]
    stage = 'test_bench'
    knn_bench = KNN(K2, stage)
    # valid_pos_bench, valid_neg_bench, X, Y_bench = read_label_validation(stage)
    test_data_bench = read_data_test(path_bench, 1, 4)
    assembly_ID = getTextID(path_assembly, 'assembly')
    assembly_ID_array = np.array(assembly_ID)[:]
    bench_ID = getTextID(path_bench, 'bench')
    bench_ID_array = np.array(bench_ID)[:-1]
    index_map = []
    for i in range(y_assembly.shape[0]):
        if sample_index_to_bench[i] == True:
            if assembly_ID[i] in bench_ID:
                index = bench_ID.index(assembly_ID[i])
                index_map.append([i, index])
    index_map_array = np.array(index_map)
    y_bench, prop_bench = knn_bench.predict(test_data_bench[index_map_array[:, 1]])

    y_full_bench = np.ones(test_data_bench.shape[0])
    y_full_bench[index_map_array[:, 1]] = y_bench
    y_full_prop = np.ones(test_data_bench.shape[0])
    y_full_prop[index_map_array[:, 1]] = prop_bench


    y_assembly[index_map_array[:, 0]] = y_bench
    print("----------------test_bench---------------")
    # loss = np.mean(abs(y_assembly - Y))
    # print('loss: ', loss)
    # false_negative = np.mean(abs(y_assembly[valid_pos.shape[0]:] - Y[valid_pos.shape[0]:]))
    # print('Mean False negative', false_negative)
    # false_positive = np.mean(abs(y_assembly[:valid_pos.shape[0]] - Y[:valid_pos.shape[0]]))
    # print('Mean False positive', false_positive)
    # cost = 0
    # cost += sample_to_bench.shape[0] * 100 + (y_assembly.shape[0] - np.sum(y_assembly)) * 500
    # all_pos_index = y_assembly == 1
    # cost += (Y[all_pos_index].shape[0] - np.sum(Y[all_pos_index])) * 25000
    # cost = cost / y_assembly.shape[0]
    # # cost += 1000
    # print('cost: ', cost)
    # K2List.append(q)
    # # lossList.append(loss)
    # costList.append(cost)
    # bigList = [K2List, costList]
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # # ax.scatter(bigList_array[0], bigList_array[1], bigList_array[2])
    # ax.scatter(bigList[0], bigList[1])
    # plt.show()
    assemly_data = []
    for i in range(y_assembly.shape[0]):
        assemly_data.append([assembly_ID_array[i], y_assembly[i], prop_assembly[i]])

    bench_data = []
    for i in range(bench_ID_array.shape[0]):
        bench_data.append([bench_ID_array[i], y_full_bench[i], y_full_prop[i]])

    writetocsv(assemly_data, 'data/results_test_bench_classifier_Team3.csv')
    writetocsv(bench_data, 'data/results_failure_processing_classifier_Team3.csv')


    # stage = 'test_bench'
    # gmm, kernel_pca = pre_process(stage)
    # sample_pos, sample_neg, X_c1, Y_c1 = read_label_data(stage)
    # # sample_pos_reduced = kernel_pca.transform(sample_pos)
    # # sample_neg_reduced = kernel_pca.transform(sample_neg)
    # X_reduced = kernel_pca.transform(X_c1)
    # # scorelist=[]
    # # scorelist1 = []
    # # cost = []
    # knn1 = KNeighborsClassifier(n_neighbors=1, weights='uniform')
    # y = []
    # knn1.fit(X_reduced, Y_c1)
    # knn = KNeighborsClassifier(n_neighbors=13, weights='distance')
    # knn.fit(X_reduced, Y_c1)
    # # valid_pos, valid_neg, X, Y = read_label_validation(stage)
    # X_reduced_valid = kernel_pca.transform(X)
    # for j in range(X_reduced_valid.shape[0]):
    #     kNeighbour = knn.kneighbors([X_reduced_valid[j]], n_neighbors=i)[1]
    #     if np.min(Y_c1[kNeighbour]) <= 0:
    #         y.append(0)
    #     else:
    #         y.append(1)
    # y = np.array(y)
    # score = knn.score(X_reduced_valid, Y)
    # scorelist.append(score)

    # print(score)
    # y = knn.predict(X_reduced_valid)
    # false_negative = np.mean(abs(y[valid_pos.shape[0]:] - Y[valid_pos.shape[0]:]))
    # scorelist.append(false_negative)
    # print('False negative', false_negative)
    # false_positive = np.mean(abs(y[:valid_pos.shape[0]] - Y[:valid_pos.shape[0]]))
    # print('False positive', false_positive)
    # scorelist1.append(false_positive)
    # cost.append(false_negative*25000+false_positive*1000)
    # print('min: ', np.min(y))
    # print('max: ', np.max(y))
    # y = []
    #     plt.plot(scorelist)
    #     plt.plot(scorelist1)
    #     # plt.plot(cost)
    #     plt.show()


        # gmm, kernel_pca = pre_process()
        # sample_pos, sample_neg, X_c1, Y_c1 = read_label_data()
        # valid_pos, valid_neg, X, Y = read_label_validation()
        # # sample_pos_reduced = kernel_pca.transform(sample_pos)
        # # sample_neg_reduced = kernel_pca.transform(sample_neg)
        # X_reduced = kernel_pca.transform(X_c1)
        # clf = AdaBoostClassifier(
        #     DecisionTreeClassifier(max_depth=4),
        #     n_estimators=600,
        #     learning_rate=1)
        # clf.fit(X_reduced, Y_c1)
        # # clf1 = clf1.fit(X_reduced, Y_c1)
        # # clf2 = clf2.fit(X_reduced, Y_c1)
        # # clf3 = clf3.fit(X_reduced, Y_c1)
        # # eclf = eclf.fit(X_reduced, Y_c1)
        # X_reduced_valid = kernel_pca.transform(X)
        # y = clf.predict(X_reduced_valid)



        # false_negative = np.mean(abs(y[valid_pos.shape[0]:] - Y[valid_pos.shape[0]:]))
        # # scorelist.append(false_negative)
        # print('False negative', false_negative)
        # false_positive = np.mean(abs(y[:valid_pos.shape[0]] - Y[:valid_pos.shape[0]]))
        # print('False positive', false_positive)
        # # scorelist1.append(false_positive)
        # cost = false_negative*25000+false_positive*1000
        # print('min: ', np.min(y))
        # print('max: ', np.max(y))
        # print('cost', cost)
        # plt.plot(scorelist)
        # plt.plot(scorelist1)
        # plt.plot(cost)
        # plt.show()

        # gmm,cluster_label = GMM_clustering(X_c1, 3)
        # sample_pos_cluster = gmm.predict(sample_pos)
        # sample_neg_cluster = gmm.predict(sample_neg)
        # pos_c1 = sample_pos[np.where(sample_pos_cluster == 0), :][0, :, :]
        # neg_c1 = sample_neg[np.where(sample_neg_cluster == 0), :][0, :, :]
        # X_c1, Y_c1 = merge_dataset(pos_c1, neg_c1)
        # my_svm = SVC(kernel='rbf')
        # my_svm.fit(X_c1, Y_c1)
        # print('score : ', my_svm.score(X_c1, Y_c1))
        # svm_label = my_svm.predict(X_c1).reshape((-1, 1))
        # print("min: ", np.min(svm_label))
        # print('svm label shape : ', svm_label.shape)
        # print('Y_c1 shape : ', Y_c1.shape)

        # LLR = LinearLogisticRegression()
        # LLR.fit(X_c1, Y_c1)
        # valid_pos, valid_neg, X, Y = read_label_validation()
        # y = LLR.predict(X)
        # loss = np.mean(abs(y - Y))

        #print('min label: ', np.min(svm_label))
        #print('max label: ', np.max(svm_label))

        # print('miss classified : ', np.sum(np.abs(svm_label - Y_c1)))


        # validate_pos, validate_neg, validate_X, validate_Y = read_label_validation()
        # validate_X_reduced = kernel_pca.transform(validate_X)
        # validate_cluster = gmm.predict(validate_X_reduced)
        # validate_c1 = validate_X_reduced[np.where(validate_cluster == 0), :][0, :, :]
        # validate_label_c1 = validate_Y[np.where(validate_cluster == 0), :][0, :, :]
        # validate_label = my_svm.predict(validate_c1).reshape((-1, 1))
        # print('validate score : ', my_svm.score(validate_c1, validate_label_c1))
        # print('miss classified : ', np.sum(np.abs(validate_label - validate_label_c1)))

