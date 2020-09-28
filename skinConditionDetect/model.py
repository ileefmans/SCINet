import torch
import torch.nn as nn
import torch.optim 
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


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








