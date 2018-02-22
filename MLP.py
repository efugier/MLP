# A simple Multi-Layer Perceptron Re-implentation
# Copyright (C) 2011  Nicolas P. Rougier (base program)

import numpy as np


def sigmoid(x):
    """Sigmoidal function i.e. something like 1/(1+exp(-x))"""
    return np.tanh(x)  # tanh because it has simple derivatives


def dsigmoid(x):
    """Derivative of the sigmoid function"""
    return 1.0 - x ** 2  # x = rank 2 Taylor expansion of tanh (for faster calculation)


class MLP:
    """Multi Layers Perceptron aka MLP"""

    def __init__(self, *args):
        self.args = args
        n = len(args)  # number of layers

        self.layers = [np.ones(args[i] + (i == 0))  # i == 0 adds a bias node one the first layer
                       for i in range(0, n)]

        self.weights = list()
        for i in range(n - 1):  # weight matrix
            R = np.random.random(
                (self.layers[i].size, self.layers[i + 1].size))
            self.weights.append((2 * R - 1) * 0.20)  # centering the weigths

        self.m = [0] * len(self.weights)

    def update(self, inputs):
        """Propagates the input through the network
           next_layer_input = sigmoid(Matrix products of input_vector times weight_matrix) """
        self.layers[0][:-1] = inputs

        for i in range(1, len(self.layers)):
            self.layers[i] = sigmoid(
                np.dot(self.layers[i - 1], self.weights[i - 1]))

        return self.layers[-1]

    def back_propagate(self, inputs, outputs, a=0.5, m=0.1):
        """a: learning factor, m: momentum factor"""

        error = outputs - self.update(inputs)
        de = error * dsigmoid(self.layers[-1])
        deltas = list()
        deltas.append(de)

        for i in range(len(self.layers) - 2, 0, -1):
            deh = np.dot(deltas[-1], self.weights[i].T) * \
                dsigmoid(self.layers[i])
            deltas.append(deh)

        deltas.reverse()

        for i, _ in enumerate(self.weights):  # j unused, is it normal ???

            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])

            dw = np.dot(layer.T, delta)
            self.weights[i] += a * dw + m * self.m[i]
            self.m[i] = dw

    def train(self, ex_list, iterations=1000):
        """ex_list = [[input, desired output], ...]
           list of examples the MLP will train upon"""
        for _ in range(iterations):
            for ex in ex_list:
                self.back_propagate(ex[0], ex[1])

    def test(self, ex_list):
        for ex in ex_list:
            print(ex[0], '->', self.update(ex[0]))


def demo():
    """XOR funtion"""  # interesing because not linearly separable (Hello Mr.Minsky =)
    pat = (((0, 0), 0),
           ((0, 1), 1),
           ((1, 0), 1),
           ((1, 1), 0))

    neural_network = MLP(2, 2, 1)
    neural_network.train(pat)
    neural_network.test(pat)
    return neural_network


demo()
