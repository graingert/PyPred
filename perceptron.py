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

import numpy as np
import random
import matplotlib.pyplot as plt
import array
from mpl_toolkits.mplot3d import axes3d, Axes3D

def hardlim(x):
	if x >=0:
		return 1
	else:
		return 0
		  
class Perceptron:
	def __init__(self,numberOfinputs):
		self.bias = 0.0
		self.weights = np.zeros(numberOfinputs)
		
	def train(self,inputs,target):
		a = hardlim(self.test(inputs))
		e = target - a
		self.weights = np.add(self.weights,(e*inputs))
		self.bias = self.bias + e
			
	def trainIter(self, inputss, targets,iterations):
		for i in range(iterations):
			for j, inputs in enumerate(inputss):
				self.train(inputs,targets[j])
	
	def test(self,inputs):
		return ((np.dot(self.weights,inputs)) + self.bias)
		
	def __str__(self):
		return 'Weights:\n%(weights)s\nBias:\n%(bias)s' % \
		  {'weights': self.weights, 'bias': self.bias}

def demo2D():
	inputss = np.empty([100,2])
	targets = np.empty((np.size(inputss,0),1),'f')
	for ix,inputs in enumerate(inputss):
		inputs[0] = random.random()
		inputs[1] = random.random()
		if inputs[0] > 0.25*inputs[1]:
			targets[ix] = 1
		else:
			targets[ix] = 0
			
	perceptron = Perceptron(np.size(inputss,1))
	perceptron.trainIter(inputss,targets,1000)
	print(perceptron)
	f = lambda x: -((perceptron.weights[0] * x) + perceptron.bias) / perceptron.weights[1]
	
	for ix,inputs in enumerate(inputss):
		if targets[ix] == 1.0:
			plt.scatter(x=inputs[0],y=inputs[1],c='red')
		else:
			plt.scatter(x=inputs[0],y=inputs[1],c='blue')
	
	limits = [0.0,1.0,0.0,1.0]
	plt.plot([limits[0],limits[1]],[f(limits[0]),f(limits[1])])
	plt.axis(limits)
	plt.show()

def demo3D():
	inputss = np.empty([100,3])
	targets = np.empty((np.size(inputss,0),1),'f')
	for ix,inputs in enumerate(inputss):
		inputs[0] = random.random()
		inputs[1] = random.random()
		inputs[2] = random.random()
		if inputs[0]+inputs[1] > inputs[2]:
			targets[ix] = 1
		else:
			targets[ix] = 0
			
	perceptron = Perceptron(np.size(inputss,1))
	perceptron.trainIter(inputss,targets,1000)
	print(perceptron)
	
	f = lambda x,y: -(
						(
							(perceptron.weights[0] * x) + perceptron.bias
						)+(
							(perceptron.weights[1] * y) + perceptron.bias
						)
					) / perceptron.weights[2]
	
	fig = plt.figure()
	
	ax = Axes3D(fig)
	class surface:
		def __init__(self):
			self.xs = array.array('d')
			self.ys = array.array('d')
			self.zs = array.array('d')
	red = surface()
	blue = surface()
	for ix,inputs in enumerate(inputss):
		if targets[ix] == 1:
			red.xs.append(inputs[0])
			red.ys.append(inputs[1])
			red.zs.append(inputs[2])
		else:
			blue.xs.append(inputs[0])
			blue.ys.append(inputs[1])
			blue.zs.append(inputs[2])
	
	ax.scatter3D(xs=red.xs,ys=red.ys,zs=red.zs,c='blue')
	ax.scatter3D(xs=blue.xs,ys=blue.ys,zs=blue.zs,c='red')
	
	X = np.arange(0, 1, 0.125)
	Y = np.arange(0, 1, 0.125)
	X, Y = np.meshgrid(X, Y)
	Z = f(X,Y)
	ax.plot_wireframe(X,Y,Z)
	
	ax.set_xlim3d((0,1))
	ax.set_ylim3d((0,1))
	ax.set_zlim3d((0,1))
	
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	
	plt.show()

def demo():
	"""compare with http://en.literateprograms.org/Perceptron_%28Java%29"""
	inputss = np.array(
		[[ 0, 0, 0, 0 ], 
		[ 0, 0, 0, 1 ], 
		[ 0, 0, 1, 0 ],
		[ 0, 0, 1, 1 ], 
		[ 0, 1, 0, 0 ], 
		[ 0, 1, 0, 1 ], 
		[ 0, 1, 1, 0 ],
		[ 0, 1, 1, 1 ], 
		[ 1, 0, 0, 0 ],
		[ 1, 0, 0, 1 ] ])
	
	targetss = np.array(
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
	
	print(targetss)
	targetss = np.rot90(targetss)
	print(targetss)
	for targets in targetss:
		perceptron = Perceptron(np.size(inputss,1))
		perceptron.trainIter(inputss,targets,1000)
		print(perceptron)
	
		
	  
def main():
	#demo2D()
	demo3D()

if __name__ == '__main__':
	main()

