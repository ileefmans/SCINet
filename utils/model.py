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

		# Create Spacial Transformer Localization Network
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
		grid = F.affine_grid(theta, x.size(), align_corners=True)
		x = F.grid_sample(x, grid, align_corners=True)

		return x



class ConvLayer(nn.Module):
	"""
		Convolutional layer consisting of a 2d convolution, (optionally 2d dropout), max pooling and a ReLU 
	"""
	def __init__(self, in_channels, out_channels, kernel_size=5, dropout=False, final_layer=False):
		"""
			Args:

				in_channels (int): 	Number of input channels

				out_channels (int): Number of output channels
				
				kernel_size (int): 	Kernel Size for convolution
				
				dropout (bool): 	Dropout layer included if True
		
		"""
		super(ConvLayer, self).__init__()

		# Channels and kernel size
		self.in_channels = in_channels
		self.out_channels =  out_channels
		self.kernel_size = kernel_size

		# Different types of modules to be used
		self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size)
		self.pool = nn.MaxPool2d(kernel_size=2, return_indices=True)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

		# Whether or not dropout will be used, also whether or not this block is the final block of a portion 
		# of the network
		self.dropout = dropout
		self.drop = nn.Dropout2d()
		self.final_layer=final_layer

	def forward(self, x):
		"""
			Forward pass of the network
		"""
		x = self.conv(x)
		if self.dropout==True:
			x = self.drop(x)
		pre_pool_dim = x.size()
		x, idx = self.pool(x)
		if self.final_layer ==False:
			x = self.relu(x)
		else:
			x = self.sigmoid(x)

		return x, idx, pre_pool_dim


class DeConvLayer(nn.Module):
	"""
		De-convolutional layer consisting of a 2d de-convolution, (optionally 2d dropout), max unpooling and a ReLU 
	"""
	def __init__(self, pool_index, pre_pool_dims, in_channels, out_channels, kernel_size=5, dropout=False, final_layer=False):
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

		# Index and dimensions for unpooling, channels, and kerel size
		self.pool_index = pool_index
		self.pre_pool_dims = pre_pool_dims
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size

		# Layers used by network
		self.convT = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5)
		self.unpool = nn.MaxUnpool2d(kernel_size=2)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

		# Whether or not dropout will be used, also whether or not this block is the final block of a portion 
		# of the network
		self.dropout = dropout
		self.drop = nn.Dropout2d()
		self.final_layer = final_layer

	def forward(self, x):
		"""
			Forward pass of the network
		"""
		x = self.unpool(x, self.pool_index, self.pre_pool_dims)
		if self.dropout==True:
			x = self.drop(x)
		x = self.convT(x)
		if self.final_layer==False:
			x = self.relu(x)
		else:
			x = self.sigmoid(x)


		return x
		




class Encoder(nn.Module):
	"""
		Encoder portion of each branch of the bi-headed network SCINet2.0
	"""

	def __init__(self):
		super(Encoder, self).__init__()

		# Custom convolutional blocks with spacial transformer network integrated between feature mappings
		self.conv1 = ConvLayer(3, 10, kernel_size=5, dropout=True)
		self.conv2 = ConvLayer(10, 20, kernel_size=5, dropout=False, final_layer=True)
		
	def forward(self, x):
		"""
			Forward pass of the network
		"""
		stn = STN(x.size())
		x = stn(x)
		x, idx1, pre_pool_dim1 = self.conv1(x)
		stn = STN(x.size())
		x = stn(x)
		x, idx2, pre_pool_dim2 = self.conv2(x)

		return x, [idx1, idx2], [pre_pool_dim1, pre_pool_dim2]



class Decoder(nn.Module):
	"""
		Decoder portion of each branch of the bi-headed network SCINet2.0
	"""

	def __init__(self, index_list, dimension_list):
		"""
			Args:

					index_list (list): List of pooling indices returned from Encoder

					dimension_list (list): List of prepooling dimensions returned from Encoder
		"""
		super(Decoder, self).__init__()

		# index and dimensions for unpooling
		self.index_list = index_list
		self.dimension_list = dimension_list

		# Custom de-convolutional blocks with spacial transformer network integrated between feature mappings
		self.convT1 = DeConvLayer(self.index_list[1], self.dimension_list[1], 20, 10, kernel_size=5, dropout=False)
		self.convT2 = DeConvLayer(self.index_list[0], self.dimension_list[0], 10, 3, kernel_size=5, dropout=True, final_layer=True)

	def forward(self, x):
		"""
			Forward pass of the network
		"""
		stn = STN(x.size())
		x = stn(x)
		x = self.convT1(x)
		stn = STN(x.size())
		x = stn(x)
		x = self.convT2(x)

		return x

class SCINet20(nn.Module):
	"""
		Bi-headed autoencoder with Spacial Transformer Network modules between the convolutional feature mappings
	"""

	def __init__(self):
		super(SCINet20, self).__init__()

		# Instantiate encoder for each head of the network
		self.encoder1 = Encoder()
		self.encoder2 = Encoder()

	def forward(self, x1, x2):
		"""
			Args:

				x1 (torch tensor): tensor to be fed to first autoencoder

				x2 (torch tensor): tensor to be fed to second autoencoder

			Forward pass of the network
		"""

		# Feed tensors into two heads of the network
		z1, idx1, dim1 = self.encoder1(x1)
		z2, idx2, dim2 = self.encoder1(x2)

		# Instantiate decoders using the indexes and dimensions recieved from the 2 encoders
		decoder1 = Decoder(idx1, dim1)
		decoder2 = Decoder(idx2, dim2)

		# feed output tensor of encoders into the decoders
		x1 = decoder1(z1)
		x2 = decoder2(z2)

		# return output from each autoencoder along with the latent representation of each autoencoder
		return x1, x2, z1, z2

		





		






