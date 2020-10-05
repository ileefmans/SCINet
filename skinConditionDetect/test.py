import os
import torch
import numpy as np 
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from model import IANet
import argparse
from tqdm import tqdm
import boto3
from PIL import Image



def get_args():
	parser = argparse.ArgumentParser(description = "Model Options")
	parser.add_argument("--model_version", type=int, default=1, help="Version of model to be trained: options = {1:'MVP', ...)")
	parser.add_argument("--model_path", type=str, default="SCINet.pt", help="Path to saved model")
	parser.add_argument("--local", type=int, default=0, help="1 if running on local machine, 0 if running on AWS")
	parser.add_argument("--local_test_pickle_path", type=str, default="/Users/ianleefmans/Desktop/Insight/Project/Re-Identifying_Persistent_Skin_Conditions/skinConditionDetect/pickle/simple_test_dict.pkl", help="path to local val pickled annotation path dictionary")
	parser.add_argument("--remote_test_pickle_path", type=str, default="simple_test_dict.pkl")
	parser.add_argument("--local_data_directory", type=str, default="/Users/ianleefmans/Desktop/Insight/Project/Data", help="Path to data")
	parser.add_argument("--remote_data_directory", type=str, default="<blank>", help="no remote data dictionary applicable")
	parser.add_argument("--image_size", type=tuple, default=(256,256), help="Size all images will be transformed to (height,width)")
	parser.add_argument("--batch_size", type=int, default=30, help="Minibatch size")
	parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")
	parser.add_argument("--shuffle", type=bool, default=True, help="True if dataloader shuffles input samples before batching, False if samples are batched in order")
	parser.add_argument("--geometric", type=bool, default=False, help= "True if geometric mapping, False if deep learning mapping")
	parser.add_argument("--load_weights", type=bool, default=False, help="Determines whether or not pretrained weights will be loaded during training")
	parser.add_argument("--access_key", type=str, default="", help="AWS Access Key")
	parser.add_argument("--secret_access_key", type=str, default="", help="AWS Secret Access Key")


	return parser.parse_args()




class Test:
	"""
		Class for testing and evaluating models
	"""

	def __init__(self):

		self.ops = get_args()
		self.model_path = self.ops.model_path
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.model = torch.load(self.model_path)
		self.model = self.model.to(self.device)
		self.local = self.ops.local
		self.access_key = self.ops.access_key
		self.secret_access_key = self.ops.secret_access_key

		if self.local ==1:
			self.local=True
			self.test_pickle_path = self.ops.local_test_pickle_path
			self.data_directory = self.ops.local_data_directory
		else:
			self.local=False
			self.test_pickle_path = self.ops.remote_test_pickle_path
			self.data_directory = self.ops.remote_data_directory

		self.img_size = self.ops.image_size
		self.transform = torchvision.transforms.ToTensor()
		self.batch_size = self.ops.batch_size
		self.num_workers = self.ops.num_workers
		self.shuffle = self.ops.shuffle
		self.geometric = self.ops.geometric
		self.load_weights = self.ops.load_weights


		# Instantiate Dataloader
		self.testset = CreateDataset(self.pickle_path, self.data_directory, img_size=self.img_size, local=self.local, access_key=self.access_key, secret_access_key=self.secret_access_key, geometric=self.geometric, transform=self.transform)

		if self.geometric==True:
			self.test_loader = DataLoader(dataset=self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, collate_fn=my_collate)
		else:
			self.test_loader = DataLoader(dataset=self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, collate_fn=my_collate2)


		def calculate_metric(self):
			pass

		def evaluate(self):

			if self.local==False:
				torch.set_default_tensor_type(torch.cuda.FloatTensor)
				print("\n \n EVERYTHING TO CUDA \n \n")

			# Load weights if applicable
			if self.load_weights == True:
				start_epoch, loss = self.load_checkpoint(self.model, self.optimizer, self.model_name)
				print("\n \n [WEIGHTS LOADED]")

			with torch.no_grad():
				self.model.eval()

				for image1, image2, annotation1, annotation2 in tqdm(self.val_loader, desc= "Validation Epoch "+str(epoch)):

					# Get dimensions of both images
					width1 = image1.size(-1)
					height1 = image1.size(-2)
					width2 = image2.size(-1)
					height2 = image2.size(-2)

					# Loop through each bounding box for image 1
					for i in range(len(annotation1[0]['boxes'].size(0))):
						box1_im = Image.new('RGB', (width1, height1))
						

						box = annotation1[0]['boxes'][i,:]
						x = int

					# Loop through each bounding box for image 2
					for i in range(len(annotation2[0]['boxes'].size(0))):















