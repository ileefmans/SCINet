import os
import torch
import numpy as np 
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import argparse
from tqdm import tqdm




from datahelper import CreateDataset, my_collate
#from model import MVP




def get_args():
	parser = argparse.ArgumentParser(description = "Model Options")
	parser.add_argument("--model_version", type=int, default=1, help="Version of model to be trained: options = {1:'MVP', ...)")
	parser.add_argument("--local", type=int, default=0, help="1 if running on local machine, 0 if running on AWS")
	parser.add_argument("--local_pickle_path", type=str, default="/Users/ianleefmans/Desktop/Insight/Project/Re-Identifying_Persistent_Skin_Conditions/skinConditionDetect/annotation_dict.pkl", help="path to local pickled annotation path dictionary")
	parser.add_argument("--remote_pickle_path", type=str, default="annotation_dict.pkl")
	parser.add_argument("--local_data_directory", type=str, default="/Users/ianleefmans/Desktop/Insight/Project/Data", help="Path to data")
	parser.add_argument("--remote_data_directory", type=str, default="<blank>", help="no remote data dictionary applicable")
	parser.add_argument("--image_size", type=tuple, default=(1000,1000), help="Size all images will be transformed to (height,width)")
	parser.add_argument("--batch_size", type=int, default=30, help="Minibatch size")
	parser.add_argument("--num_workers", type=int, default=5, help="Number of workers for dataloader")
	parser.add_argument("--shuffle", type=bool, default=True, help="True if dataloader shuffles input samples before batching, False if samples are batched in order")
	parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for Adam Optimizer")
	parser.add_argument("--epochs", type=int, default=10, help="Number of epochs model will be trained for")
	parser.add_argument("--local_save_path", type=str, default="/Users/ianleefmans/Desktop/Insight/Project/Data/model_checkpoints", help="Path to folder to save model parameters after trained")
	parser.add_argument("--load_weights", type=bool, default=False, help="Determines whether or not pretrained weights will be loaded during training")







	return parser.parse_args()




class Trainer:
	"""
		Class for training project models
	"""

	def __init__(self):

		self.ops = get_args()
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.model_version = self.ops.model_version
		
		# Import model
		if self.model_version==1:
			self.model = fasterrcnn_resnet50_fpn(pretrained=True).to(self.device) 
			self.num_classes = 7
			self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features
			self.model.roi_heads.box_predictor = FastRCNNPredictor(self.in_features, self.num_classes)
		else:
			# ENTER OTHER MODEL INITIALIZATIONS HERE
			pass

		# Set local or remote paths
		self.local = self.ops.local
		print("LOCAL: ", self.local)
		if self.local ==1:
			self.local=True
			print("TRUE")
			self.pickle_path = self.ops.local_pickle_path
			self.data_directory = self.ops.local_data_directory
			self.save_path = self.ops.local_save_path
		else:
			self.local=False
			print("False")
			self.pickle_path = self.ops.remote_pickle_path
			self.data_directory = self.ops.remote_data_directory
			

		self.img_size = self.ops.image_size
		self.transform = torchvision.transforms.ToTensor()
		self.batch_size = self.ops.batch_size
		self.num_workers = self.ops.num_workers
		print("NUM WORKERS: ", self.num_workers)
		self.shuffle = self.ops.shuffle
		self.learning_rate = self.ops.learning_rate
		self.epochs = self.ops.epochs
		self.load_weights=self.ops.load_weights



		# Instantiate Dataloaders

		self.dataset = CreateDataset(self.pickle_path, self.data_directory, img_size=self.img_size, local=self.local, transform=self.transform)
		self.train_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, collate_fn=my_collate)

		# Instatntiate Optimizer
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)


		# TRAINING

	def train(self):
		# Load weights if applicable
		if self.load_weights == True:
			start_epoch, loss = self.load_model(self.model, self.optimizer, "FASTERRCNN")
			start_epoch+=1
			print("\n \n [WEIGHTS LOADED]")
		else:
			start_epoch = 0

		#start_epoch = 0

			# Start Training loop
		for epoch in range(start_epoch, self.epochs+1):

			# TRAIN
			if epoch>0:
				self.model.train()
				train_loss = 0
				counter=0
				for image, annotation in tqdm(self.train_loader, desc= "Train Epoch "+str(epoch)):

					# image tensors and bounding box and label tensors to device
					image = [im.to(self.device) for im in image]
					annotation = [{k: v.to(self.device) for k, v in t.items()} for t in annotation]

					loss_dict = self.model(image, annotation)
					loss = sum(loss for loss in loss_dict.values())
					train_loss+=loss

					# Clear optimizer gradient
					self.optimizer.zero_grad()
					# Backprop
					loss.backward()
					# Take a step 
					self.optimizer.step()
					
					if counter%10==0:
						self.save_model(self.model, self.optimizer, "FASTERRCNN", epoch, train_loss)

				print(f'====> Epoch: {epoch} Average loss: {train_loss / len(self.train_loader.dataset):.4f}\n')

			with torch.no_grad():
				self.model.eval()
				for image, annotation in tqdm(self.train_loader, desc= "Test Epoch "+str(epoch)):
					test_loss = 0

					image = [im.to(self.device) for im in image]
					annotation = [{k: v.to(self.device) for k, v in t.items()} for t in annotation]

					output = self.model(image)
			print("[DONE EPOCH{}".format(epoch))

		print("[DONE]")



				

				
			






	#SAVE MODEL PARAMETERS
	def save_model(self, model, optimizer, model_name, epoch, loss):
		"""
		Function for saving model parameters
		"""
		save_path = os.path.join(self.save_path, model_name)

		if not os.path.exists(save_path):
			os.makedirs(save_path)

		torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(), "loss": loss}, save_path+"/params.tar")


	#LOAD MODEL PARAMETERS
	def load_model(self, model, optimizer, model_name):
		"""
		Function for loading model parameters 
		"""
		load_path = os.path.join(self.save_path, model_name, "params.tar")
		checkpoint = torch.load(load_path)
		model.load_state_dict(checkpoint["model_state_dict"])
		optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		epoch = checkpoint["epoch"]
		loss = checkpoint["loss"]

		return epoch, loss




if __name__=="__main__":
	trainer = Trainer()
	trainer.train()







		