import os
import torch
import numpy as np 
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from model import SCINet20
import argparse
from tqdm import tqdm
import boto3




from datahelper import CreateDataset, my_collate, my_collate2




# Function creating arg parser
def get_args():
	parser = argparse.ArgumentParser(description = "Model Options")
	parser.add_argument("--local", type=int, default=0, help="1 if running on local machine, 0 if running on AWS")
	parser.add_argument("--local_pickle_path", type=str, default="/Users/ianleefmans/Desktop/Insight/Project/Re-Identifying_Persistent_Skin_Conditions/skinConditionDetect/pickle/simple_train_dict.pkl", help="path to local pickled annotation path dictionary")
	parser.add_argument("--remote_pickle_path", type=str, default="simple_train_dict.pkl")
	parser.add_argument("--local_val_pickle_path", type=str, default="/Users/ianleefmans/Desktop/Insight/Project/Re-Identifying_Persistent_Skin_Conditions/skinConditionDetect/pickle/simple_train_dict.pkl", help="path to local val pickled annotation path dictionary")
	parser.add_argument("--remote_val_pickle_path", type=str, default="simple_val_dict.pkl")
	parser.add_argument("--local_data_directory", type=str, default="/Users/ianleefmans/Desktop/Insight/Project/Data", help="Path to data")
	parser.add_argument("--remote_data_directory", type=str, default="<blank>", help="no remote data dictionary applicable")
	parser.add_argument("--image_size", type=tuple, default=(256,256), help="Size all images will be transformed to (height,width)")
	parser.add_argument("--batch_size", type=int, default=30, help="Minibatch size")
	parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")
	parser.add_argument("--shuffle", type=bool, default=True, help="True if dataloader shuffles input samples before batching, False if samples are batched in order")
	parser.add_argument("--geometric", type=bool, default=False, help= "True if geometric mapping, False if deep learning mapping")
	parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for Adam Optimizer")
	parser.add_argument("--epochs", type=int, default=10, help="Number of epochs model will be trained for")
	parser.add_argument("--local_save_path", type=str, default="/Users/ianleefmans/Desktop/Insight/Project/Data/model_checkpoints", help="Path to folder to save model parameters after trained")
	parser.add_argument("--load_weights", type=bool, default=False, help="Determines whether or not pretrained weights will be loaded during training")
	parser.add_argument("--access_key", type=str, default="", help="AWS Access Key")
	parser.add_argument("--secret_access_key", type=str, default="", help="AWS Secret Access Key")


	return parser.parse_args()




class Trainer:
	"""
		Class for training SCINet2.0
	"""

	def __init__(self):

		# run function to get arguments from argparser
		self.ops = get_args()

		# Set device to GPU if available otherwise cpu
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #if torch.cuda.is_available() else "cpu")
		
		# Instantiate SCINet2.0 model and send to device
		self.model = SCINet20().to(self.device)
		self.model_name = "SCINet"

		

		# Set local or remote paths 
		self.local = self.ops.local

		# Access key and Secret Access Key for AWS
		self.access_key = self.ops.access_key
		self.secret_access_key = self.ops.secret_access_key

		# Set attributes that vary upon being remote or local
		if self.local ==1:
			self.local=True
			print("TRUE")
			self.pickle_path = self.ops.local_pickle_path
			self.val_pickle_path = self.ops.local_val_pickle_path
			self.data_directory = self.ops.local_data_directory
			self.save_path = self.ops.local_save_path
		else:
			self.local=False
			print("False")
			self.pickle_path = self.ops.remote_pickle_path
			self.val_pickle_path = self.ops.remote_val_pickle_path
			self.data_directory = self.ops.remote_data_directory
			
		# Set attributes that don't depend on being local or remote
		self.img_size = self.ops.image_size
		self.transform = torchvision.transforms.ToTensor()
		self.batch_size = self.ops.batch_size
		self.num_workers = self.ops.num_workers
		self.shuffle = self.ops.shuffle
		self.geometric = self.ops.geometric
		self.learning_rate = self.ops.learning_rate
		self.epochs = self.ops.epochs
		self.load_weights=self.ops.load_weights
		self.s3 = boto3.client('s3')



		# Instantiate Dataloaders

		self.trainset = CreateDataset(self.pickle_path, self.data_directory, img_size=self.img_size, local=self.local, access_key=self.access_key, secret_access_key=self.secret_access_key, geometric=self.geometric, transform=self.transform)
		self.valset = CreateDataset(self.val_pickle_path, self.data_directory, img_size=self.img_size, local=self.local, access_key=self.access_key, secret_access_key=self.secret_access_key, geometric=self.geometric, transform=self.transform)
		if self.geometric==True:
			self.train_loader = DataLoader(dataset=self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, collate_fn=my_collate)
			self.val_loader = DataLoader(dataset=self.valset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, collate_fn=my_collate)
		else:
			self.train_loader = DataLoader(dataset=self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, collate_fn=my_collate2)
			self.val_loader = DataLoader(dataset=self.valset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, collate_fn=my_collate2)

		# Instatntiate Optimizer
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
		

	# Loss Function
	def loss_fcn(self,x1, x2, x1_hat, x2_hat, z1, z2):
		BCEx1 = nn.functional.binary_cross_entropy(x1_hat.view(-1, 3*256*256), x1.view(-1, 3*256*256), reduction='sum')
		BCEx2 = nn.functional.binary_cross_entropy(x2_hat.view(-1, 3*256*256), x2.view(-1, 3*256*256), reduction='sum')
		BCEz1z2 = nn.functional.binary_cross_entropy(z1.view(-1, 20*61*61), z2.view(-1, 20*61*61), reduction='sum')
		loss = (BCEx1 + BCEx2 + BCEz1z2)/3

		return loss




	def train(self):
		"""
			Method for training model
		"""

		# Ensure everything is sent to GPU if being trained on the cloud
		if self.local==False:
			torch.set_default_tensor_type(torch.cuda.FloatTensor)
			print("\n \n EVERYTHING TO CUDA \n \n")

		# Load weights if applicable
		if self.load_weights == True:
			start_epoch, loss = self.load_checkpoint(self.model, self.optimizer, self.model_name)
			start_epoch+=1
			print("\n \n [WEIGHTS LOADED]")
		else:
			start_epoch = 0


		# Start Training Loop
		for epoch in range(start_epoch, self.epochs+1):

			# TRAIN
			if epoch>0:

				# Set model to training mode
				self.model.train()
				
				# Initialize loss and counter that will allow model weights to be saved and overwritten every 10 minibatches
				train_loss = 0
				counter=0

				# Iterate through train set
				for image1, image2, annotation1, annotation2, landmark1, landmark2 in tqdm(self.train_loader, desc= "Train Epoch "+str(epoch)):

					# image tensors and bounding box and label tensors to device
					image1 = image1.to(self.device)
					image2 = image2.to(self.device)

					# Forward pass of model
					x1_hat, x2_hat, z1, z2 = self.model(image1, image2)
					
					# Calculate loss from one pass and append to training loss
					loss = self.loss_fcn(image1, image2, x1_hat, x2_hat, z1.detach(), z2.detach())
					train_loss+=loss.item()

					# Clear optimizer gradient
					self.optimizer.zero_grad()

					# Backprop
					loss.backward()

					# Take a step with optimizer
					self.optimizer.step()
					
					# Save/overwrite model weights every 10 minibatches
					if counter%10==0:
						self.save_checkpoint(self.model, self.optimizer, self.model_name, epoch, train_loss)

				print(f'====> Epoch: {epoch} Average train loss: {train_loss / len(self.train_loader.dataset):.4f}\n')

				# Save entire model as .pt after every epoch of training
				if self.local==True:
					torch.save(self.model, os.path.join(self.save_path, self.model_name+".pt"))
				else:
					torch.save(self.model, self.model_name+"_epoch"+str(epoch)+".pt")
					print("SAVED MODEL EPOCH " +str(epoch))



			# Evaluate on Validation Set after each epoch
			with torch.no_grad():

				# Set model to evaluation mode
				self.model.eval()

				# Iterate through validation set
				for image1, image2, annotation1, annotation2 in tqdm(self.val_loader, desc= "Validation Epoch "+str(epoch)):
					
					# Initialize validation loss
					val_loss = 0

					# Send images to device
					image1 = image1.to(self.device)
					image2 = image2.to(self.device)

					# Forward pass of model
					x1_hat, x2_hat, z1, z2 = self.model(image1, image2)

					# Calculate loss and append to validation loss
					loss = self.loss_fcn(image1, image2, x1_hat, x2_hat, z1.detach(), z2.detach())
					val_loss+=loss

				print(f'====> Epoch: {epoch} Average test loss: {val_loss / len(self.val_loader.dataset):.4f}\n')

			print("[DONE EPOCH{}]".format(epoch))

		print("[DONE TRAINING]")

		# Save model after all epochs are finished
		if self.local==True:
			torch.save(self.model, os.path.join(self.save_path, self.model_name+".pt"))
		else:
			torch.save(self.model, self.model_name+".pt")


			


	#SAVE MODEL PARAMETERS
	def save_checkpoint(self, model, optimizer, model_name, epoch, loss):
		"""
			Args:

				model: model whose parameters we desire to save

				optimizer: optimizer whose parameters we desire to save

				model_name (string): Name we will save the model as

				epoch (int): current epoch when we are saving checkpoint

				loss (float): current loss when we are saving checkpoint

		Saves model and optimizer parameters as well as epoch and loss to accomidate training 
		where you left off at a later time
		"""

		# For local
		if self.local==True:

			# Create model path
			save_path = os.path.join(self.save_path, model_name)
			if not os.path.exists(save_path):
				os.makedirs(save_path)

			# Save parameters
			torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(), "loss": loss}, save_path+"/params.tar")

		# For remote	
		else: 

			# Create path and save parameters
			save_path =  model_name+".tar"
			torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(), "loss": loss}, save_path)

		

	#LOAD MODEL PARAMETERS
	def load_checkpoint(self, model, optimizer, model_name):
		"""
			Args:

				model: Model to which parameters are to be loaded

				optimizer: Optimizer to which parameters are to be loaded

				model_name: Name of model in path where parameters are stored

		Loads training checkpoint for training at a later time
		"""

		# For local
		if self.local==True:

			# create path and load checkpoint
			load_path = os.path.join(self.save_path, model_name, "params.tar")
			checkpoint = torch.load(load_path)

		# For remote	
		else:
			
			# load checkpoint
			checkpoint = torch.load(model_name+".tar")

		# Load model state dict into model, optimizer state dict into optimizer and load epoch and loss
		model.load_state_dict(checkpoint["model_state_dict"])
		optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		epoch = checkpoint["epoch"]
		loss = checkpoint["loss"]

		return epoch, loss





if __name__=="__main__":
	trainer = Trainer()
	trainer.train()







		