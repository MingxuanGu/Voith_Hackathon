#    Copyright 2016 Stefan Steidl
#    Friedrich-Alexander-Universität Erlangen-Nürnberg
#    Lehrstuhl für Informatik 5 (Mustererkennung)
#    Martensstraße 3, 91058 Erlangen, GERMANY
#    stefan.steidl@fau.de


#    This file is part of the Python Classification Toolbox.
#
#    The Python Classification Toolbox is free software:
#    you can redistribute it and/or modify it under the terms of the
#    GNU General Public License as published by the Free Software Foundation,
#    either version 3 of the License, or (at your option) any later version.
#
#    The Python Classification Toolbox is distributed in the hope that
#    it will be useful, but WITHOUT ANY WARRANTY; without even the implied
#    warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#    See the GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with the Python Classification Toolbox.
#    If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import numpy.matlib


class LinearLogisticRegression(object):

    def __init__(self, learningRate = 0.5, maxIterations = 100):
        self.learningRate = learningRate
        self.maxIterations = maxIterations

    # X: (23,2)
    def fit(self, X, y):
        # make the label(y) either 0 or 1
        y = np.array(y / max(y))
        # concatenate the bias term
        x = np.array(np.concatenate((X, np.ones((X.shape[0], 1))), axis=1).T)
        # y = theta.T * x
        # initialize the weight(theta)
        self.__theta = numpy.random.rand(x.shape[0],1)
        for i in range(self.maxIterations):
            # grad = np.zeros((3,1))
            hessian = np.zeros((X.shape[0],X.shape[0]))
            # print(np.tile((y.reshape(1, -1) - self.gFunc(x, self.__theta)), (x.shape[0], 1)).shape)
            # input()
            grad = np.sum(np.tile((y.reshape(1, -1) - self.gFunc(x, self.__theta)), (x.shape[0], 1)) * x, axis=1).reshape(X.shape[0],1)
            for j in range(X.shape[0]):
                xj = x[:, j].reshape(X.shape[0], 1)
                fx = self.gFunc(xj, self.__theta)
                # grad += ((y[j] - fx) * xj)
                hessian -= fx * (1 - fx) * np.dot(xj, xj.T)
            self.__theta = self.__theta - self.learningRate * np.dot(np.linalg.inv(hessian),grad)

    # gaussian pdf
    def gFunc(self, X, theta):
        return 1.0/(1+numpy.exp(-(numpy.dot(theta.T,X))))


    def predict(self, X):
        x = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        y = self.gFunc(x.T, self.__theta)
        y[y<0.5] = 0
        y[y>0.5] = 1
        y[y==0.5] = np.random.randint(0,2,1)

        return y


