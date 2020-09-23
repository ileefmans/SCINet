import os
import torch
import numpy as np 
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import argparse
from tqdm import tqdm

from datahelper import CreateDataset, my_collate
from model import MVP




def get_args():
    parser = argparse.ArgumentParser(description = "Model Options")
    parser.add_argument("--model_version", type=int, default=1, help="Version of model to be trained: options = {1:'MVP', ...)")
    parser.add_argument("--local", type=bool, default=True, help="True if running on local machine, False if running on AWS")
    parser.add_argument("--local_pickle_path", type=str, default="/Users/ianleefmans/Desktop/Insight/Project/Re-Identifying_Persistent_Skin_Conditions/skinConditionDetect/annotation_dict.pkl", help="path to local pickled annotation path dictionary")
    parser.add_argument("--local_data_directory", type=str, default="/Users/ianleefmans/Desktop/Insight/Project/Data", help="Path to data")
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
			self.model = MVP().to(self.device) 
		else:
			# ENTER OTHER MODEL INITIALIZATIONS HERE
			pass

		# Set local or remote paths
		self.local = self.ops.local
		if self.local==True:
			self.pickle_path = self.ops.local_pickle_path
			self.data_directory = self.ops.local_data_directory
		else:
			# CREATE AWS PATHS HERE
			pass

		self.img_size = self.ops.image_size
		self.transform = torchvision.transforms.ToTensor()
		self.batch_size = self.ops.batch_size
		self.num_workers = self.ops.num_workers
		self.shuffle = self.ops.shuffle
		self.learning_rate = self.ops.learning_rate
		self.epochs = self.ops.epochs



		# Instantiate Dataloaders

		self.dataset = CreateDataset(self.pickle_path, self.data_directory, img_size=self.img_size, local=self.local, transform=self.transform)
		self.train_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, collate_fn=my_collate)


		self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)


		# TRAINING

		def train(self):

			for epoch in range(start_epoch, self.epochs+1):














		