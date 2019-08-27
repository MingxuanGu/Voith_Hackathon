import numpy as np
import csvReader


class data:
    def __init__(self, batch_size=-1):
        pos = csvReader.getarray('Data/pos.csv')
        neg = csvReader.getarray('Data/neg.csv')
        self.data = np.concatenate((pos, neg))
        mean = np.mean(self.data)
        self.data -= mean
        self.label = np.zeros((len(pos) + len(neg), 2))
        self.label[:len(pos), 0] = 1
        self.label[len(pos):, 1] = 1
        self.batch_num = 0
        if batch_size==-1:
            self.batch_size = self.data.shape[0]
        else:
            self.batch_size = batch_size

    def get_label(self):
        return self.label

    def get_data(self):
        return self.data

    def next(self):
        batch_end = 0
        output_data = 0
        output_label = 0
        if (self.batch_size + self.batch_num < self.label.shape[0]):
            batch_end = self.batch_size + self.batch_num
            output_data = self.data[self.batch_num:batch_end]
            output_label = self.label[self.batch_num:batch_end]
            self.batch_num += self.batch_size
        else:
            batch_end = self.label.shape[0]
            output_data = self.data[self.batch_num:batch_end]
            output_label = self.label[self.batch_num:batch_end]
            self.batch_num = 0

        return output_data, output_label

dt = data(10)
