#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       protien.py
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
import collections
import numpy as np
import array
import sys

class Protein:
	
	def __init__(self, acids, types):
		self.acids = acids.upper()
		self.types = types.upper()

		self.inputss = list()
		self.targetss = list()
		
		window = collections.deque('00000',maxlen=5)
		halfwindow = collections.deque('000',maxlen=3)
		for acid, atype in zip(self.acids,self.types):
			window.append(acid)
			halfwindow.append(atype)
			self.inputss.append(self.to_inputs(window))
			self.targetss.append(self.to_targets(halfwindow[0]))
		
		
		window.append('0')
		halfwindow.append('0')
		self.inputss.append(self.to_inputs(window))
		self.targetss.append(self.to_targets(halfwindow[0]))
		
		window.append('0')
		halfwindow.append('0')
		self.inputss.append(self.to_inputs(window))
		self.targetss.append(self.to_targets(halfwindow[0]))
			
		self.inputss = np.array(self.inputss)
		self.targetss = np.array(self.targetss)
			
	
	def to_inputs(self,deque):
		inputs = array.array('i')
		for acid in deque:
			inputs.extend(self.get_aa_value(acid))
		return inputs
		
	protein_alphabet = "ACDEFGHIKLMNPQRSTVWY"
	def get_aa_value(self,aa):
		value = np.zeros([20],dtype=np.dtype(int))
		if aa == "0":
			return value
		index = self.protein_alphabet.rindex(aa)
		value[index] = 1
		return value
	
	type_alphabet = "EH_"
	def to_targets(self,char):
		char.upper()
		if char == "0":
			return {'E': 0,'H':0,'U':0}
		if char == "E":
			return {'E': 1,'H':0,'U':0}
		if char == "H":
			return {'E': 0,'H':1,'U':0}
		if char == "_":
			return {'E': 0,'H':0,'U':1}

def main():
	protein = Protein("ACDEFGHIKLMNPQRSTVWY", "hhhee___hheeee____hh")
	for inputs,targets in zip(protein.inputss,protein.targetss):
		print(inputs)
		print targets
	return 0

if __name__ == '__main__':
	main()

