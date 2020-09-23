import torch
import torch.nn as nn
import torch.optim 
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class MVP(nn.Module):
	""" 
		Class for Spatial Transformer Network (STN)
	"""

	def __init__(self):
		super(MVP, self).__init__()

		self.model = fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=7)


	def forward(self, image, annotations):
		output = self.model(image, annotations)


		return output