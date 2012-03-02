#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       untitled.py
#       
#       Copyright 2011 Thomas Grainger <graingert@graingert-DeskUbu>
#       
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#       
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#       
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.

import data as mydata
import perceptron as myp
from protein import Protein
import collections
import numpy as np
import array
import sys
import random
from StringIO import StringIO
import math
import matplotlib.pyplot as plt


from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

class cc:
	def __init__(self):
		self.p = 0.0
		self.n = 0.0
		self.o = 0.0
		self.u = 0.0
	def performance(self):
		p = self.p
		n = self.n
		o = self.o
		u = self.u
		denom = math.sqrt((p+o+1)*(p+u+1)*(n+o+1)*(n+u+1))
		if denom == 0:
			print('\033[91m')
			print (self)
			print('\033[0m')
			return 2
		return ((p*n)-(o*u))/denom
	def __str__(self):
		return 'p:%(p)s,n:%(n)s,o:%(o)s,u:%(u)s' % {'p':self.p,'n':self.n,'o':self.o,'u':self.u}

def calculatecoef(targets,predicts):
	ch = cc()
	ce = cc()
	cu = cc()
	for target, prediction in zip(targets,predicts):
		for coef,structure in zip((ch,ce,cu),('H','E','_')):
			if target == structure and prediction == structure:
				coef.p += 1
			if target != structure and prediction != structure:
				coef.n += 1
			if target != structure and prediction == structure:
				coef.o += 1
			if target == structure and prediction != structure:
				coef.u += 1
	return ch,ce,cu

def testold():
	perceptron_h = myp.Perceptron(100)
	perceptron_e = myp.Perceptron(100)
	perceptron_u = myp.Perceptron(100)
	
	trainingProteins = []
	for aastr,targetstr in mydata.trainingData():
		trainingProteins.append(Protein(aastr,targetstr))
	
	testingProteins = []
	for aastr,targetstr in mydata.testData():
		testingProteins.append(Protein(aastr,targetstr))
	
	inputs_targets = []
	for p in trainingProteins:
		inputs_targets.extend(zip(p.inputss,p.targetss))
	
	def train():
		random.shuffle(inputs_targets)
		for inputs,targets in inputs_targets:
			perceptron_h.train(inputs,targets['H'])
			perceptron_e.train(inputs,targets['E'])
			perceptron_u.train(inputs,targets['U'])
	
	def test():
		for protein in testingProteins:
			print (protein.acids)
			print (protein.types)
			sys.stdout.write('\033[91m')
			i = 0
			for inputs,targets in zip(protein.inputss,protein.targetss):
				i = i + 1
				if i < 3:
					continue
				highestOutput = perceptron_h.test(inputs);
				bestPrediction = 'H';
				e = perceptron_e.test(inputs)
				if e > highestOutput:
					highestOutput = e;
					bestPrediction = 'E'
				u = perceptron_u.test(inputs)
				if u > highestOutput:
					highestOutput = u;
					bestPrediction = '_'
				sys.stdout.write(bestPrediction)
			print('\033[0m')
	for i in range(5):
		print i
		train()
	test()
	return 0

def testnew():
	X = list()
	yH = list()
	yE = list()
	yU = list()
	
	perceptron_h = myp.Perceptron(100)
	perceptron_e = myp.Perceptron(100)
	perceptron_u = myp.Perceptron(100)
	
	trainingProteins = []
	for aastr,targetstr in mydata.trainingData():
		trainingProteins.append(Protein(aastr,targetstr))
	
	testingProteins = []
	for aastr,targetstr in mydata.testData():
		testingProteins.append(Protein(aastr,targetstr))
	
	inputs_targets = []
	for p in trainingProteins:
		inputs_targets.extend(zip(p.inputss,p.targetss))
	
	def train():
		random.shuffle(inputs_targets)
		for inputs,targets in inputs_targets:
			perceptron_h.train(inputs,targets['H'])
			perceptron_e.train(inputs,targets['E'])
			perceptron_u.train(inputs,targets['U'])
	
	def test():
		alltargets = StringIO()
		allpredictions = StringIO()
		for protein in testingProteins:
			alltargets.write(protein.types)
			i = 0;
			for inputs,targets in zip(protein.inputss,protein.targetss):
				i = i + 1
				if i < 3:
					continue
				highestOutput = perceptron_h.test(inputs);
				bestPrediction = 'H';
				e = perceptron_e.test(inputs)
				if e > highestOutput:
					highestOutput = e;
					bestPrediction = 'E'
				u = perceptron_u.test(inputs)
				if u > highestOutput:
					highestOutput = u;
					bestPrediction = '_'
				allpredictions.write(bestPrediction)
		ch,ce,cu = calculatecoef(alltargets.getvalue(),allpredictions.getvalue())
		yH.append(ch.performance())
		yE.append(ce.performance())
		yU.append(cu.performance())
		print('')
		
	for i in range(1,100):
		print i
		X.append(i)
		train()
		test()
	plt.plot(X,yH,c='red', label = "Alpha Helix")
	plt.plot(X,yE,c='blue', label = "Beta Sheet")
	plt.plot(X,yU,c='green', label = "Coil")
	
	plt.xlabel("Iteration")
	plt.ylabel("Correlation Coefficient")
	plt.legend()
	plt.show()
	
	print(perceptron_h.weights)
	print(perceptron_e.weights)
	print(perceptron_u.weights)
	
	return 0
	
def testpybrain():
	X = list()
	yH = list()
	yE = list()
	yU = list()
	
	perceptron = buildNetwork(100, 5, 3, bias=True)
	
	trainingProteins = []
	for aastr,targetstr in mydata.trainingData():
		trainingProteins.append(Protein(aastr,targetstr))
	
	testingProteins = []
	for aastr,targetstr in mydata.testData():
		testingProteins.append(Protein(aastr,targetstr))
	
	inputs_targets = []
	for p in trainingProteins:
		inputs_targets.extend(zip(p.inputss,p.targetss))
		
	
	ds = SupervisedDataSet(100,3)
	for inputs,targets in inputs_targets:
		ds.addSample(inputs,(targets['H'],targets['E'], targets['U']))
		
	trainer = BackpropTrainer(perceptron, ds)
	
	def test():
		alltargets = StringIO()
		allpredictions = StringIO()
		for protein in testingProteins:
			alltargets.write(protein.types)
			i = 0;
			for inputs in protein.inputss:
				i = i + 1
				if i < 3:
					continue
				highestOutput = 0
				bestPrediction = 'H'
				for prediction,value in zip(perceptron.activate(inputs),('H','E','_')):
					if prediction > highestOutput:
						highestOutput = prediction;
						bestPrediction = value
				allpredictions.write(bestPrediction)
		ch = cc()
		ce = cc()
		cu = cc()
		ch,ce,cu = calculatecoef(alltargets.getvalue(),allpredictions.getvalue())
		
		yH.append(ch.performance())
		yE.append(ce.performance())
		yU.append(cu.performance())
		
		print(ch.performance(),ce.performance(),cu.performance())
		
	for i in range(1,100):
		print i
		X.append(i)
		trainer.train()
		test()
	plt.plot(X,yH,c='red', label = "Alpha Helix")
	plt.plot(X,yE,c='blue', label = "Beta Sheet")
	plt.plot(X,yU,c='green', label = "Coil")
	
	plt.xlabel("Iteration")
	plt.ylabel("Correlation Coefficient")
	plt.legend()
	plt.show()
	
	print(perceptron_h.weights)
	print(perceptron_e.weights)
	print(perceptron_u.weights)
	
	return 0

def main():
	#testpybrain()
	
	testingProteins = []
	for aastr,targetstr in mydata.testData():
		print aastr
		

if __name__ == '__main__':
	main()

