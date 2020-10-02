import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torchvision 
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor




class STN(nn.Module):
	"""
		Spatial Transformer Network Module
	"""
	def __init__(self, input_size):
		"""
			Args:

					input_size (torch.Size): Size of input tensor    ex: x.size()

		"""
		super(STN, self).__init__()
		self.input_size = input_size[-1]
		self.channels = input_size[1]
		# Spacial Transformer Localization Network
		self.kernel_size1 = 7
		self.kernel_size2 = 5
		self.localization = nn.Sequential(
			nn.Conv2d(self.channels, 8, kernel_size=self.kernel_size1),
			nn.MaxPool2d(2, stride=2),
			nn.ReLU(True),
			nn.Conv2d(8,10, kernel_size=self.kernel_size2),
			nn.MaxPool2d(2, stride=2),
			nn.ReLU(True)
		)

		# calculate dimentions after convolutions (needed for fully connected layers)
		self.local_size  = round(((round(((self.input_size-(self.kernel_size1-1))/2)-.01) - (self.kernel_size2-1))/2)-.01)
		

		# Regressor for the 3x2 affine matrix
		self.fc_loc = nn.Sequential(
			nn.Linear(10*self.local_size*self.local_size, 32),
			nn.ReLU(True),
			nn.Linear(32, 3*2)
		)

		# Initialize the weights/bias with identity transformation
		self.fc_loc[2].weight.data.zero_()
		self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

	def forward(self, x):
		# Obtain transformation parameters (theta)
		xs = self.localization(x)
		xs= xs.view(-1, 10*self.local_size*self.local_size)
		#Create 3x2 affine matrix
		theta = self.fc_loc(xs)
		theta = theta.view(-1, 2, 3)
		# Grid Generator
		grid = F.affine_grid(theta, x.size())
		x = F.grid_sample(x, grid)

		return x



class ConvLayer(nn.Module):
	"""
		Convolutional layer consisting of a 2d convolution, (optionally 2d dropout), max pooling and a ReLU 
	"""
	def __init__(self, in_channels, out_channels, kernel_size=5, dropout=False):
		"""
			Args:

				in_channels (int): 	Number of input channels

				out_channels (int): Number of output channels
				
				kernel_size (int): 	Kernel Size for convolution
				
				dropout (bool): 	Dropout layer included if True
		
		"""
		super(ConvLayer, self).__init__()

		self.in_channels = in_channels
		self.out_channels =  out_channels
		self.kernel_size = kernel_size

		self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size)
		self.pool = nn.MaxPool2d(kernel_size=2, return_indices=True)
		self.relu = nn.ReLU()

		self.dropout = dropout
		self.drop = nn.Dropout2d()

	def forward(self, x):
		x = self.conv(x)
		if self.dropout==True:
			x = self.drop(x)
		pre_pool_dim = x.size()
		x, idx = self.pool(x)
		x = self.relu(x)

		return x, idx, pre_pool_dim


class DeConvLayer(nn.Module):
	"""
		De-convolutional layer consisting of a 2d de-convolution, (optionally 2d dropout), max unpooling and a ReLU 
	"""
	def __init__(self, pool_index, pre_pool_dims, in_channels, out_channels, kernel_size=5, dropout=False):
		"""
			Args:

				pool_index (idx): 	Index returned from nn.MaxPool2d operation

				pre_pool_dims (torch.Size): Dimensions prior to the corresponding pooling operation

				in_channels (int): 	Number of input channels

				out_channels (int): Number of output channels
				
				kernel_size (int): 	Kernel Size for convolution
				
				dropout (bool): 	Dropout layer included if True
		"""
		super(DeConvLayer, self).__init__()

		self.pool_index = pool_index
		self.pre_pool_dims = pre_pool_dims
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size

		self.convT = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5)
		self.unpool = nn.MaxUnpool2d(kernel_size=2)
		self.relu = nn.ReLU()

		self.dropout = dropout
		self.drop = nn.Dropout2d()

	def forward(self, x):
		x = self.unpool(x, self.pool_index, self.pre_pool_dims)
		if self.dropout==True:
			x = self.drop(x)
		x = self.convT(x)
		x = self.relu(x)

		return x
		




class Encoder(nn.Module):

	def __init__(self):
		super(Encoder, self).__init__()
		self.conv1 = ConvLayer(3, 10, kernel_size=5, dropout=True)
		self.conv2 = ConvLayer(10, 20, kernel_size=5, dropout=False)
		
	def forward(self, x):
		stn = STN(x.size())
		x = stn(x)
		x, idx1, pre_pool_dim1 = self.conv1(x)
		stn = STN(x.size())
		x = stn(x)
		x, idx2, pre_pool_dim2 = self.conv2(x)

		return x, [idx1, idx2], [pre_pool_dim1, pre_pool_dim2]



class Decoder(nn.Module):

	def __init__(self, index_list, dimension_list):
		"""
			Args:

					index_list = List of pooling indices returned from Encoder
		"""
		super(Decoder, self).__init__()
		self.index_list = index_list
		self.dimension_list = dimension_list

		self.convT1 = DeConvLayer(self.index_list[1], self.dimension_list[1], 20, 10, kernel_size=5, dropout=False)
		self.convT2 = DeConvLayer(self.index_list[0], self.dimension_list[0], 10, 3, kernel_size=5, dropout=True)

	def forward(self, x):
		stn = STN(x.size())
		x = stn(x)
		x = self.convT1(x)
		stn = STN(x.size())
		x = stn(x)
		x = self.convT2(x)

		return x

class IANet(nn.Module):

	def __init__(self):
		super(IANet, self).__init__()

		self.encoder = Encoder()

	def forward(self, x):
		x, idx, dim = self.encoder(x)
		decoder = Decoder(idx, dim)
		x = decoder(x)

		return x





		






