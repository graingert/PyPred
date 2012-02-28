#!/usr/bin/python
import random
import math

import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    """Perceptron"""

    def __init__(self, size, bail=100):
        self.size = size
        self.weights = 0.0001 * np.random.randn(size + 1)
        self.learn_rate = 1
        self.bail = bail
    def __str__(self):
        return str(self.weights)
        
    def prep_data(self,inputdata):
        training_set = [(np.array(data[0] + [1]), data[1]) for data in inputdata]
        random.shuffle(training_set)
        return training_set

    def train_on(self, inputdata):
        training_set = self.prep_data(inputdata)

        error_history = []

        for it in range(0,self.bail):
            false_neg, false_pos = self.train_iteration(training_set)
            error = abs(false_neg) + abs(false_pos)
            error_history.append(error)
            if error == 0: return error_history
            print(str(it) + " " + str(error) + " of " + str(len(inputdata)))
        return error_history

    def train_iteration(self, training_set):
        false_pos = 0
        false_neg = 0

        for data in training_set:
            error = self.train_single(data[0],data[1])
            if error > 0:
                false_neg += 1
            if error < 0:
                false_pos += 1
        return false_neg, false_pos

    def train_single(self, input_values, expected):
        result = self._calculate(input_values)
        error = expected - result
        self.weights += (error * self.learn_rate * input_values)
        return error

    def predict(self, input_data):
        return self._calculate(np.array(input_data + [1]))

    def predict_real(self, input_data):
        return np.dot(self.weights,np.array(input_data + [1]))

    def _calculate(self, input_data):
        if np.dot(self.weights,input_data) > 0:
            return 1
        else:
            return 0

def is_valid(point):
    if point[1] > 80 - point[0] * 0.5:
    #if (point[0]*point[1]) > 1:
    #if 0.75 * point[0] > point[1] + 1:
        return 1
    else:
        return 0

def main():
    max_x = 100
    max_y = 100

    vis_fig = plt.figure()
    err_fig = plt.figure()
    
    plt.figure(vis_fig.number)
    
    plt.ion()

    p = Perceptron(2, bail=1000)
    points = [[random.uniform(0,max_x), random.uniform(0,max_y)] for i in range(1000)]
    inputs = [[point, is_valid(point)] for point in points]
    #plt.scatter([point[0][0] for point in inputs],[point[0][1] for point in inputs], c="#00ff00", marker = 'x')

    error_history = p.train_on(inputs)

    plt.figure(err_fig.number)
    plt.plot([v/1000.0 for v in error_history])
    plt.xlabel("Itteration")
    plt.ylabel("Errors (%)")
    plt.figure(vis_fig.number)
    plt.xlabel("x")
    plt.ylabel("y")
    
    b = p.weights[2]
    w1 = p.weights[0]
    w2 = p.weights[1]
    y1 = -((w1 * 0) + b ) / w2
    y2 = -((w1 * max_x) + b ) / w2

    plt.plot([-0,max_x],[y1,y2])
    
    results = [(data[0],p.predict(data[0]),data[1]) for data in inputs]
        
    plt.scatter([point[0][0] for point in results if point[1] == 1],[point[0][1] for point in results if point[1] == 1], marker = 'x', c="#00ff00")
    plt.scatter([point[0][0] for point in results if point[1] == 0],[point[0][1] for point in results if point[1] == 0], marker = '+', c="#ff0000")
    error_points = [point for point in results if point[1] != point[2]]
    if len(error_points) > 0:
        plt.scatter([point[0][0] for point in error_points],[point[0][1] for point in error_points], marker = 'o', c="#0000ff")
    plt.xlim(0,max_x)
    plt.ylim(0,max_y)
    plt.draw()
    raw_input("fin") 


if __name__ == "__main__":
    main()
