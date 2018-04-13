# -*- encoding: utf -*-

from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

def format_datasets(X, y):
    training = SupervisedDataSet(X.shape[1], y.shape[1])
    for i in range(x_train.shape[0]):
        training.addSample(x_train[i], y_train[i])

    testing = SupervisedDataSet(X.shape[1], y.shape[1])
    for i in range(x_test.shape[0]):
        testing.addSample(x_test[i], y_test[i])

    net = buildNetwork(X.shape[1], 100, y.shape[1], bias=True)