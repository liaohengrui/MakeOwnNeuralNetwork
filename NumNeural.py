#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/28
# @Author  : Henry

import NeuralNetwork
from scipy import misc
import numpy
from matplotlib import pyplot

n = NeuralNetwork.ReadyNumNeuralNetwork()

img_array = misc.imread("aaa.png", flatten=True)
img_data = 255.0 - img_array.reshape(784)
img_data = (img_data / 255.0 * 0.99) + 0.01
print(img_data)
outputs = n.query(img_data)
print(outputs)
label = numpy.argmax(outputs)
print(label)
pyplot.imshow(img_array, cmap='Greys', interpolation='None')
pyplot.show()

# test_data_file = open("mnist/mnist_test.csv.txt", 'r')
# test_data_list = test_data_file.readlines()
# test_data_file.close()
#
# print(numpy.asfarray(test_data_list[0].split(",")[1:]) / 255.0 * 0.99 + 0.01)
