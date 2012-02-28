#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#	   perceptron.py
#	   
#	   Copyright 2011 Thomas Grainger <graingert@graingert-DeskUbu>
#	   
#	   This program is free software; you can redistribute it and/or modify
#	   it under the terms of the GNU General Public License as published by
#	   the Free Software Foundation; either version 2 of the License, or
#	   (at your option) any later version.
#	   
#	   This program is distributed in the hope that it will be useful,
#	   but WITHOUT ANY WARRANTY; without even the implied warranty of
#	   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#	   GNU General Public License for more details.
#	   
#	   You should have received a copy of the GNU General Public License
#	   along with this program; if not, write to the Free Software
#	   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#	   MA 02110-1301, USA.
#
#		ref: http://www.mathworks.com/help/toolbox/nnet/histori3.html

import numpy
import random
import array
import pylab

def hardlim(x):
	if (isinstance(x,numpy.ndarray)):
		print(x)
		returnVal = numpy.empty(numpy.shape(x))
		for ix, var in enumerate(x):
			returnVal[ix] = hardlim(var)
			
def hardlim2D(x):
	if x >= 0:
		return 1
	else:
		return 0

class Protein:
	proteinalphabet = "ACDEFGHIKLMNPQRSTVWY"
	def get_aa_value(self,aa):
		index = self.proteinalphabet.rindex(aa)
		value = numpy.zero([20])
		value[index] = 1
		return value


class Perceptron2D:
	__weights = numpy.array([0.0,0.0])
	__bias = 0.0
	def __init__(self,inputs,targets):
		self.__inputs = inputs
		self.__targets = targets
		self.__weights = numpy.zeros(2)
		
	def train(self):
		for ix, input in enumerate(self.__inputs):
			a = hardlim2D(numpy.dot(self.__weights,input) + self.__bias)
			e = self.__targets[ix] - a
			self.__weights = self.__weights + (e*input)
			self.__bias = self.__bias + e
			
	def trainIter(self, iterations):
		for i in range(iterations):
			self.train()
			
	def f(self, x):
		return -((self.__weights[0] * x) + self.__bias) / self.__weights[1]
	
	def __str__(self):
		return 'Weights:\n%(weights)s\nBias:\n%(bias)s' % \
		  {'weights': self.__weights, 'bias': self.__bias}
		  
class Perceptron:
	
	class Not2D(Exception):
		def __init__(self, value):
			self.value = value
		def __str__(self):
			return repr(self.value)
			
	__bias = 0.0
	
	def __init__(self,inputs,targets):
		self.__inputs = inputs
		self.__targets = targets
		self.__weights = numpy.zeros((numpy.size(targets,1),numpy.size(inputs,1)))
		
	def train(self):
		for ix, input in enumerate(self.__inputs):
			a = hardlim(numpy.add(numpy.dot(self.__weights,input), self.__bias))
			e = numpy.subtract(self.__targets[ix], a)
			self.__weights = self.__weights + (e*input)
			self.__bias = self.__bias + e
			
	def trainIter(self, iterations):
		for i in range(iterations):
			self.train()
			
	def f(self, x):
		if numpy.size(self.__weights,1) == 2:
			return -((self.__weights[0][0] * x) + self.__bias) / self.__weights[0][1]
		else:
			raise(self.Not2D('The perceptron is not 2D'))
			
	
	def __str__(self):
		return 'Inputs:\n%(inputs)s\nTargets:\n%(targets)s\nWeights:\n%(weights)s\nBias:\n%(bias)s' % \
		  {'inputs': self.__inputs, 'targets': self.__targets, 'weights': self.__weights, 'bias': self.__bias}
		
		

def demo2D():
	inputs = numpy.empty([100,2])
	targets = numpy.empty((numpy.size(inputs,0),1),'f')
	for ix,input in enumerate(inputs):
		input[0] = random.random()
		input[1] = random.random()
		if input[0] > 0.25*input[1]:
			targets[ix] = 1
		else:
			targets[ix] = 0
	
	perceptron = Perceptron2D(inputs,targets)
	perceptron.trainIter(100)
	
	for ix,input in enumerate(inputs):
		if targets[ix] == 1.0:
			pylab.scatter(x=input[0],y=input[1],c='red')
		else:
			pylab.scatter(x=input[0],y=input[1],c='blue')
	
	
	limits = [0.0,1.0,0.0,1.0]
	pylab.plot([limits[0],limits[1]],[perceptron.f(limits[0]),perceptron.f(limits[1])])
	pylab.axis(limits)
	pylab.show()
	
	return 0

def demo():
	"""compare with http://en.literateprograms.org/Perceptron_%28Java%29"""
	inputs = numpy.array(
		[[ 0, 0, 0, 0 ], 
		[ 0, 0, 0, 1 ], 
		[ 0, 0, 1, 0 ],
		[ 0, 0, 1, 1 ], 
		[ 0, 1, 0, 0 ], 
		[ 0, 1, 0, 1 ], 
		[ 0, 1, 1, 0 ],
		[ 0, 1, 1, 1 ], 
		[ 1, 0, 0, 0 ], 
		[ 1, 0, 0, 1 ]])
	
	targets = numpy.array(
		[ 
	[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
	[ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], 
	[ 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 ],
	[ 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 ], 
	[ 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 ],
	[ 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 ], 
	[ 1, 1, 1, 1, 1, 1, 0, 0, 0, 0 ],
	[ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0 ], 
	[ 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 ],
	[ 1, 1, 1, 1, 1, 1, 1, 1, 1, 0 ] ])
	perceptron = Perceptron(inputs,targets)
	perceptron.trainIter(100)
	print(perceptron)
	
		
	  
def main():
	demo2D()

if __name__ == '__main__':
	main()

