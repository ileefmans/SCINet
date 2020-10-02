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
		self.local_size  = round((round((self.input_size-(self.kernel_size1-1))/2) - (self.kernel_size2-1))/2)
		print(self.local_size)

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
		self.pool = nn.MaxPool2d(2)

		self.dropout = dropout
		self.drop = nn.Dropout2d()

	def forward(self, x):
		x = self.conv(x)
		if self.dropout==True:
			x = self.drop(x)
		x = self.pool(x)

		return x
		




class Encoder(nn.Module):

	def __init__(self):
		super(Encoder, self).__init__()
		self.conv1 = ConvLayer(3, 10, kernel_size=5, dropout=True)
		self.conv2 = ConvLayer(10, 20, kernel_size=5, dropout=False)
		
	def forward(self, x):
		stn = STN(x.size())
		x = stn(x)
		x = self.conv1(x)
		stn = STN(x.size())
		x = stn(x)
		x = self.conv2(x)
		return x

class Decoder(nn.Module):

	def __init__(self):
		super(Decoder, self).__init__()






		

	















# class MVP(nn.Module):
# 	""" 
# 		Class for Pretrained Faster-RCNN
# 	"""

# 	def __init__(self):
# 		super(MVP, self).__init__()

# 		self.model = fasterrcnn_resnet50_fpn(pretrained=True)
# 		self.num_classes = 7
# 		self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features
# 		self.model.roi_heads.box_predictor = FastRCNNPredictor(self.in_features, self.num_classes)

		


# 	def forward(self, image, annotations):
		
# 		output = self.model(image, annotations)


# 		return output






# class Flatten(nn.Module):
    
#     """
#         Module which flattens tensor to a 1D tensor
#     """
    
#     def forward(self, x):
#         return x.view(x.size(0), -1)



# class Unflatten(nn.Module):
    
#     """
#         Module which reformats flattened tensor into original format
#     """
    
#     def __init__(self, x, num_features=None, dimensions=None):
#         super(Unflatten, self).__init__()
        
#         self.dimensions = dimensions
#         self.num_features = num_features
    
#     def forward(self, x):
#         if self.num_features==None or self.dimensions==None:
#             raise Exception("Mandatory argument not assigned")
        
#         return x.view(x.size(0), self.num_features*8, self.dimensions[0], self.dimensions[1])




# class Fold(nn.Module):
    
#     """
#         Module which folds 1D tensor (1 X n) resulting in a (2 X n/d) tensor
#     """
    
#     def forward(self, x):
#         return x.view(-1, 2, int(x.size(1)/2))



# class VAE(nn.Module):
    
#     """
#         Class for Variational Auto Encoder with Convolutional Layers
#     """
#     def __init__(self, num_features):
#         super(VAE, self).__init__()
        
#         self.conv1 = nn.Conv2d(3, num_features, 5)
#         self.conv2 = nn.Conv2d(num_features, num_features*2, 5)
#         self.conv3 = nn.Conv2d(num_features*2, num_features*4, 5)
#         self.conv4 = nn.Conv2d(num_features*4, num_features*8, 5)
        
#         self.convT1 = nn.ConvTranspose2d(num_features*8, num_features*4, 5)
#         self.convT2 = nn.ConvTranspose2d(num_features*4, num_features*2, 5)
#         self.convT3 = nn.ConvTranspose2d(num_features*2, num_features, 5)
#         self.convT4 = nn.ConvTranspose2d(num_features, 3, 5)
        
        
#         self.pool = nn.MaxPool2d(kernel_size=2, return_indices=True)
#         self.unpool = nn.MaxUnpool2d(kernel_size=2)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#         self.flatten = Flatten()
#         self.fold = Fold()
#         self.num_features = num_features
        

    
    
#     def Encoder(self, x):
        
#         """
#             Encoder segment of Variational AutoEncoder
#         """
        
#         x = self.conv1(x)
#         dim1 = x.size()
#         x, idx1 = self.pool(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         dim2 = x.size()
#         x, idx2 = self.pool(x)
#         x = self.relu(x)
#         x = self.conv3(x)
#         dim3 = x.size()
#         x, idx3 = self.pool(x)
#         x = self.relu(x)
#         x = self.conv4(x)
#         dim4 = x.size()
#         x, idx4 = self.pool(x)
#         x = self.relu(x)
#         pic_dim = (x.size(2), x.size(3))
#         x = self.flatten(x)
#         x = self.fold(x)
#         idx = [idx1, idx2, idx3, idx4]
#         prepool_dim = [dim1, dim2, dim3, dim4]
#         return x, idx, pic_dim, prepool_dim
    
#     def Decoder(self, x, idx, prepool_dim):
#         """
#            Decoder segment of Variational AutoEncoder
#         """
        
#         unpool1 = nn.MaxUnpool2d(kernel_size=2)
#         unpool2 = nn.MaxUnpool2d(kernel_size=2)
#         unpool3 = nn.MaxUnpool2d(kernel_size=2)
#         unpool4 = nn.MaxUnpool2d(kernel_size=2)
        
#         x = unpool1(x, idx[3], prepool_dim[3])
#         x = self.convT1(x)
#         x = self.relu(x)
#         x = unpool2(x, idx[2], prepool_dim[2])
#         x = self.convT2(x)
#         x = self.relu(x)
#         x = unpool3(x, idx[1], prepool_dim[1])
#         x = self.convT3(x)
#         x = self.relu(x)
#         x = unpool4(x, idx[0], prepool_dim[0])
#         x = self.convT4(x)
#         x = self.sigmoid(x)
#         return x
        


#     def reparameterize(self, mu, logvar):
#         """
#            Function for reparameterization of Variational Autoencoder:
#                 Samples from a Guassian Distribution
#         """
#         if self.training:
#             std= logvar.mul(0.5).exp_()
#             eps1 = std.data.new(std.size()).normal_()
#             eps = std.data.new(std.size()).normal_()
#             return torch.cat((eps1.mul(std).add_(mu), eps.mul(std).add_(mu)), 1)
#         else:
#             return torch.cat((mu, mu), 1)


#     def forward(self, x):
#         x, idx, dimensions, prepool_dim = self.Encoder(x)
#         #logger.debug("  Tensor after Flatten and Fold is {},".format(x.size()))
#         mu = x[:,0,:]
#         logvar = x[:,1,:]
#         x = self.reparameterize(mu, logvar)
#         #logger.debug("  Tensor after Reparameterize is {},".format(x.size()))
#         unflatten = Unflatten(x, dimensions=dimensions, num_features = self.num_features)
#         x = unflatten(x)
#         #logger.debug("  Tensor after Unflatten is {},".format(x.size()))
#         x = self.Decoder(x, idx, prepool_dim)
#         #logger.debug("  Tensor after Decoder is {},\n".format(x.size()))
        
        
#         return x, mu, logvar







