import numpy as np
import csvReader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
from data import data

def main():
    # my_data = csvReader.getarray('Data/test_bench_training.csv')[1:, 4:]
    # print(my_data.shape)
    # eig_val, eig_vec = np.linalg.eig(np.cov(my_data.T))
    # print(eig_vec.shape)
    # datasplit()
    fig = plt.figure()
    ax = Axes3D(fig)

    x_line = [13, 13, 10, 11, 9, 14, 13, 14, 16, 18, 16, 16, 16]
    y_line = [3, 5, 5, 3, 3, 3, 6, 6, 6, 6, 3, 5, 8]
    z_line = [15787.105083718463, 12563.413894992045, 13827.505871656942, 16335.385256458821, 17237.764224562467, 15604.345026138342, 11306.936131525114, 11126.714145010985, 10589.855292067581, 10605.08371846352, 15057.33388893098, 11842.525948935527, 8637.87029320403]
    ax.scatter(x_line, y_line, z_line)
    # ax.plot3D(x_line, y_line, z_line, 'gray')

    plt.show()
    # z_points = 15 * np.random.random(100)
    # x_points = np.cos(z_points) + 0.1 * np.random.randn(100)
    # y_points = np.sin(z_points) + 0.1 * np.random.randn(100)
    # ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='hsv');


    # listOfID = list(map(float, listOfID))
    # a = (float)listOfID[0]
    # print(listOfID)
    # plt.plot(eig_val)
    # plt.ylabel('eig_val')
    # plt.show()
    pass

def datasplit():
    good, bad = csvReader.splitLabelData()
    # print(np.array(bad).shape)
    with open('data/pos_90.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in good:
            writer.writerow(row)
    with open('data/neg_90.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in bad:
            writer.writerow(row)


main()
