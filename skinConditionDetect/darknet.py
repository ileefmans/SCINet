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
	module_list = nn.ModuleList()    #Initialize module list
	prev_filters = 3
	output_filters = []

	for index, x in enumerate(blocks[1:]):
		module = nn.Sequential()

		# Define Convolutional Layer parameters
		if (x['type'] == 'convolutional'):
			# Determine Activation Function
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

			#Create Convolutional layer and add to Sequential() module
			conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
			module.add_module("conv_{0}".format(index), conv)

			#Create Batch Norm layer and add to Sequential() module
			if batch_normalize:
				bn = nn.BatchNorm2d(filters)
				module.add_module("batch_norm_{0}".format(index), bn)

			# Create Activation layer and add to Sequential() module
			if activation == "leaky":
				act = nn.LeakyReLU(0.1, inplace=True)
				module.add_module("leaky_{0}".format(index), act)

		# Define Upsample layer parameters
		elif (x['type'] =='upsample'):
			stride = int(x["stride"])
			upsample = nn.Upsample(scale_factor=2, mode = 'bilinear')
			module.add_module("upsample_{}".format(index), upsample)

		# Define route layer
		elif (x['type'] == "route"):
			x['layers'] = x['layers'].split(',')

			#Create start and end of route
			start = int(x["layers"][0])
			try:
				end = int(x["layers"][1])
			except:
				end = 0


			if start>0:
				start = start - index
			if end>0:
				end = end - index

			route = EmptyLayer()

			# Add route to Sequential() module
			module.add_module("route_{0}".format(index), route)

			if end<0:
				filters = output_filters[index+start] + output_filters[index+end]
			else:
				filters = output_filters[index+start]

		# Define skip connection
		elif (x['type'] == 'shortcut'):
			shortcut = EmptyLayer()
			module.add_module("shortcut_{}".format(index), shortcut)

		# Define Detection Layer
		elif (x['type'] == 'yolo'):
			mask = x['mask'].split(',')
			mask = [int(x) for x in mask]

			anchors = x['anchors'].split(',')
			anchors = [int(a) for a in anchors]
			anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
			anchors = [anchors[i] for i in mask]

			detection = DetectionLayer(anchors)
			module.add_module("Detection_{}".format(index), detection)

		module_list.append(module)
		prev_filters = filters
		output_filters.append(filters)

	return (net_info, module_list)











#Create Empty Layer module
class EmptyLayer(nn.Module):
	"""
		Class for Empty Layer
	"""
	def __init__(self):
		super(EmptyLayer, self).__init__()

#Create Detection Layer module
class DetectionLayer(nn.Module):
	"""
		Class for Detection Layer
	"""
	def __init__(self, anchors):
		super(DetectionLayer, self).__init__()
		self.anchors = anchors

if __name__ == "__main__":
	blocks = parse_cfg("yolov3.cfg")
	print(create_modules(blocks))





















