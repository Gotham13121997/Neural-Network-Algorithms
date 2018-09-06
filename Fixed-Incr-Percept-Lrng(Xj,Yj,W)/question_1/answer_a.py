"""
Created by Gotham on 06-09-2018.
"""
import random


class FixedIncrPerceptron:
    def __init__(self, inputs, outputs, bias, learning_rate):
        self.input_nodes = []
        self.bias = bias
        for inp in inputs:
            self.input_nodes.append([bias]+inp)
        self.weights = []
        for i in range(0, len(self.input_nodes[0])):
            x = random.random()
            self.weights.append(x)
        self.outputs = outputs
        self.learning_rate = learning_rate

    def sum_of_wi_xi(self, ind):
        net = 0
        for i in range(0, len(self.input_nodes[ind])):
            net = net + self.input_nodes[ind][i]*self.weights[i]
        if net > 0:
            return 1
        return 0

    def update_weights(self, plus, ind):
        plus_or_minus = -1
        if plus:
            plus_or_minus = 1
        for i in range(0, len(self.weights)):
            self.weights[i] = self.weights[i] + \
                              plus_or_minus*self.learning_rate*self.input_nodes[ind][i]

    def train(self):
        for i in range(0, len(self.input_nodes)):
            while True:
                net = self.sum_of_wi_xi(i)
                if net == self.outputs[i]:
                    break
                self.update_weights(net < self.outputs[i], i)

    def predict(self, inputs):
        inputs = [self.bias] + inputs
        net = 0
        for i in range(0, len(inputs)):
            net = net + inputs[i] * self.weights[i]
        if net > 0:
            return 1
        return 0
