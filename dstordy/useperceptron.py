#!/usr/bin/python
import cProfile

import matplotlib.pyplot as plt
import numpy as np

from perceptron import Perceptron
from fileparser import parse as parse_protein_file
from protein import Protein


window_len = 5
test_proteins = parse_protein_file("protein-secondary-structure.test")
train_proteins = parse_protein_file("protein-secondary-structure.train")

def inter_result(v):
    if v == 0: return "h"
    if v == 1: return "e"
    if v == 2: return "_"

def imax(l):
    maxi = 0;
    maxv = l[0];
    for i in range(1, len(l)):
        if l[i] > maxv:
            maxv = l[i]
            maxi = i
    return maxi

def check_error_percent(h, e, c, input_data):
    correct = 0.0
    total = 0.0
    for protein_d in input_data:
        protein = protein_d.get_encoded_data(window_len)
        total += len(protein)
        result = []
        expected = np.array(protein_d.ss_vals())
        for i, window in enumerate(protein):
            w = list(window)
            hv = h.predict_real(w)
            ev = e.predict_real(w)
            cv = c.predict_real(w)
            r = imax([hv,ev,cv])
            result.append(r)
            if expected[i] == r:
                correct += 1
    return correct / total

def run_predictions(h, e, c):
    for protein_d in test_proteins:
        protein = protein_d.get_encoded_data(window_len)
        result = []
        for window in protein:
            w = list(window)
            hv = h.predict_real(w)
            ev = e.predict_real(w)
            cv = c.predict_real(w)
            r = imax([hv,ev,cv])
            result.append(r)
            if expected[i] == r:
                correct += 1

        print protein_d
        print "".join([inter_result(v) for v in result])

def main():
    print len(train_proteins)

    h = Perceptron(window_len * 20)
    e = Perceptron(window_len * 20)
    c = Perceptron(window_len * 20)

    training_data = []

    for protein in train_proteins:
        training_data.extend(protein.get_training_data(window_len))

    l = [[list(d[0]),d[1]] for d in training_data]
    hdata = h.prep_data([[d[0],d[1][0]] for d in l])
    edata = e.prep_data([[d[0],d[1][1]] for d in l])
    cdata = c.prep_data([[d[0],d[1][2]] for d in l])
    her = []
    eer = []
    cer = []
    train_errors = []
    test_errors = []
    for i in range(0,50):
        print i
        her.append(h.train_iteration(hdata))
        eer.append(e.train_iteration(edata))
        cer.append(c.train_iteration(cdata))
        train_err = check_error_percent(h, e, c, train_proteins)
        print train_err
        train_errors.append(train_err)
        test_err =  check_error_percent(h, e, c, test_proteins)
        test_errors.append(test_err)
        print test_err
    #error_history = h.train_on([[d[0],d[1][0]] for d in l])
    #e.train_on([[d[0],d[1][1]] for d in l])
    #c.train_on([[d[0],d[1][2]] for d in l])

    plt.ion()
#    plt.plot(her)
#    plt.plot(eer)
#    plt.plot(cer)
    plt.plot(train_errors)
    plt.plot(test_errors)
    plt.xlabel("Itteration")
    plt.ylabel("No. Errors")
    plt.draw()

    #run_predictions(h, e, c)

    raw_input("fin")

if __name__ == "__main__":
    main()
