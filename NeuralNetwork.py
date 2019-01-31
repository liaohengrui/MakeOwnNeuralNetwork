#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/28
# @Author  : Henry

import numpy
from scipy import special
from scipy import misc


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate

        # 创建权值矩阵, wih.txt 和 who

        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # 激活函数
        self.activity_function = lambda x: special.expit(x)
        pass

    def train(self, input_list, target_list):
        inputs = numpy.array(input_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activity_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activity_function(final_inputs)

        targets = numpy.array(target_list, ndmin=2).T
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # 更新权值
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                        numpy.transpose(inputs))
        pass

    def query(self, input_list):
        inputs = numpy.array(input_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activity_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activity_function(final_inputs)
        return final_outputs


class ReadyNumNeuralNetwork:
    def __init__(self):
        self.inodes = 784
        self.hnodes = 100
        self.onodes = 10

        # 创建权值矩阵, wih.txt 和 who
        self.wih = numpy.load("wih.npy")
        self.who = numpy.load("who.npy")

        # 激活函数
        self.activity_function = lambda x: special.expit(x)
        pass

    def query(self, input_list):
        inputs = numpy.array(input_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activity_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activity_function(final_inputs)
        return final_outputs


if __name__ == '__main__':
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.3

    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    training_data_file = open("mnist/mnist_train.csv.txt", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    for record in training_data_list:
        all_values = record.split(",")
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99

        n.train(inputs, targets)

    test_data_file = open("mnist/mnist_test.csv.txt", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    scorecard = []

    for record in test_data_list:
        all_values = record.split(",")
        current_label = int(all_values[0])
        print('current:', current_label)
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = n.query(inputs)
        label = numpy.argmax(outputs)
        print('network:', label)

        if label == current_label:
            scorecard.append(1)
        else:
            scorecard.append(0)

    print("识别率", scorecard.count(1) / len(scorecard) * 100, "%")


