from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np


def parse_cfg(cfg_file):
	"""
		Function that takes a configuration file and returns a list of network blocks

		Args:

			cfgfile (.cfg file): A configuration file for the network
	"""


	#Read and clean configuration file
	file = open(cfg_file, 'r')
	lines = file.read().split('\n')
	lines = [x for x in lines if len(x)>0]
	lines = [x for x in lines if x[0] != "#"]
	lines = [x.rstrip() for x in lines]

	#Initialize dictionary for each block and list for blocks to be appended
	block = {}
	blocks = []

	#Loop through lines appending blocks to block list
	for line in lines:
		if line[0] == "[":
			if len(block)!=0:
				blocks.append(block)
				block = {}
			block['type'] = line[1:-1].rstrip()
		else:
			key, value = line.split("=")
			block[key.rstrip()] = value.lstrip()
		blocks.append(block)

	return blocks


def create_modules(blocks):
	"""
		Function to create network modules based on configuration blocks

		Args:

			blocks (list): list of configuration blocks
	"""
	net_info = blocks[0]
	module_list = nn.Module_list()    #Initialize module list
	prev_filters = 3
	output_filters = []

	for index, x in enumerate(blocks[1:]):
		module = nn.Sequential()

		# Determine Activation Function
		if (x['type'] == 'convolutional'):
			activation= x['activation']

			# Determine whether or not batch normalization
			try:
				batch_normalize = int(x["batch_normalize"])
				bias = False
			except:
				batch_normalize = 0
				bias = True

			# Determine convolution info
			filters = int(x['filters'])
			padding = int(x['pad'])
			kernel_size = int(x['size'])
			stride = int(x['stride'])

			if padding:
				pad = (kernel_size - 1)//2
			else:
				pad = 0













