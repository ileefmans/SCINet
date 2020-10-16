import os
import torch
import numpy as np 
import torch
from torch import nn
from torch.utils.data import DataLoader
from datahelper import CreateDataset, my_collate, my_collate2
from preprocess import Image_Process
import torchvision
from model import SCINet20
import argparse
from tqdm import tqdm
import boto3
from PIL import Image
import cv2




def get_args():
	"""
		Function for parsing arguments
	"""
	parser = argparse.ArgumentParser(description = "Model Options")
	parser.add_argument("--model_version", type=int, default=1, help="Version of model to be trained: options = {1:'MVP', ...)")
	parser.add_argument("--model_path", type=str, default="/Users/ianleefmans/Desktop/Insight/Project/Data/model_checkpoints/SCINet_epoch1.pt", help="Path to saved model")
	parser.add_argument("--local", type=int, default=0, help="1 if running on local machine, 0 if running on AWS")
	parser.add_argument("--local_test_pickle_path", type=str, default="/Users/ianleefmans/Desktop/Insight/Project/Re-Identifying_Persistent_Skin_Conditions/skinConditionDetect/pickle/simple_train_dict.pkl", help="path to local val pickled annotation path dictionary")
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

		# Run function to get argparser
		self.ops = get_args()

		# Set up path to model
		self.model_path = self.ops.model_path

		# Set up device, gpu if avalable otherwise cpu
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		# Load model and send to device
		self.model = torch.load(self.model_path, map_location=torch.device('cpu'))
		self.model = self.model.to(self.device)

		# Set up attribute for local or remote testing along with access and secret access keys
		self.local = self.ops.local
		self.access_key = self.ops.access_key
		self.secret_access_key = self.ops.secret_access_key

		# Set up attributes that vary whether or not the script is being run locally or remotely
		if self.local ==1:
			self.local=True
			self.test_pickle_path = self.ops.local_test_pickle_path
			self.data_directory = self.ops.local_data_directory
		else:
			self.local=False
			self.test_pickle_path = self.ops.remote_test_pickle_path
			self.data_directory = self.ops.remote_data_directory

		# Set up attributes that do not vary
		self.img_size = self.ops.image_size
		self.transform = torchvision.transforms.ToTensor()
		self.batch_size = self.ops.batch_size
		self.num_workers = self.ops.num_workers
		self.shuffle = self.ops.shuffle
		self.geometric = self.ops.geometric
		self.load_weights = self.ops.load_weights


		# Instantiate Dataloader
		self.testset = CreateDataset(self.test_pickle_path, self.data_directory, img_size=self.img_size, local=self.local, access_key=self.access_key, secret_access_key=self.secret_access_key, geometric=self.geometric, transform=self.transform)

		if self.geometric==True:
			self.test_loader = DataLoader(dataset=self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, collate_fn=my_collate)
		else:
			self.test_loader = DataLoader(dataset=self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, collate_fn=my_collate2)


	def process_for_inference(self, box, width, height):
		"""
			Args:

				box (iterable obj): Bounding box coordinates

				width: (int): Width of image

				height: (int): Height of image

			Method to create blank image with bouding box filled in
		"""

		# Create blank image size of picture
		box_im = Image.new('RGB', (width, height))
		box_im = cv2.cvtColor(np.array(box_im), cv2.COLOR_RGB2BGR)

		# Get dimensions for bboxes
		x1 = int(box[0])
		y1 = int(box[1])
		x2 = int(box[2])
		y2 = int(box[3])

		# Draw bounding boxes
		cv2.rectangle(box_im,(x1,y1),(x2,y2),(0,255,0),-1) 

		# Transform to PIL Image for preprocessing before entering SCINet
		box_im = Image.fromarray(box_im)
		box_im = box_im.convert('RGB')

		# Preprocess
		image_process = Image_Process((256,256))
		box_im = image_process.expand(box_im)
		transform = torchvision.transforms.ToTensor()
		box_im = transform(box_im)
		box_im = image_process.uniform_size(box_im)

		return box_im


	def IoU(self, input1, input2):
		"""
			Args:
				input1(torch tensor): First tensor of image to be evaluated via Intersection over Union

				input2(torch tensor): Second tensor of image to be evaluated via Intersection over Union

			Method to evaluate intersection over union
		"""
		union = np.count_nonzero(np.array(input1) + np.array(input2))
		total  = np.count_nonzero(np.array(input1)) + np.count_nonzero(np.array(input2))
		intersection = total-union
		IoU = intersection/union
		return IoU


	def evaluate(self):
		"""
			Method to evaluate SCINet2.0's performance
		"""

		# If remote send everything to GPU
		if self.local==False:
			torch.set_default_tensor_type(torch.cuda.FloatTensor)
			print("\n \n EVERYTHING TO CUDA \n \n")

		# Load weights if applicable
		if self.load_weights == True:
			start_epoch, loss = self.load_checkpoint(self.model, self.optimizer, self.model_name)
			print("\n \n [WEIGHTS LOADED]")

		# Set model to evaluate
		with torch.no_grad():
			self.model.eval()

			# Initialize list for results
			results = []

			# Iterate over test set
			for image1, image2, annotation1, annotation2, landmark1, landmark2 in tqdm(self.test_loader, desc= "Test "):

				# Get dimensions of both images
				width1 = image1.size(-1)
				height1 = image1.size(-2)
				width2 = image2.size(-1)
				height2 = image2.size(-2)

				# Number of boxes for each image
				num_boxes1 = annotation1[0]['boxes'].size(0)
				num_boxes2 = annotation2[0]['boxes'].size(0)


				# Blank lists where output tensors will go for each image's bounding boxes
				output_im1 = []
				output_im2 = []

				# Loop through each bounding box for image 1
				for i in range(max(num_boxes1, num_boxes2)):
					
					# Get bounding boxes
					try:
						box1 = annotation1[0]['boxes'][i,:]
					except:
						# if out of range for image with less boxes, fill in blank box to feed to other branch of SCINet
						box1 = [0, 0, 0, 0]
					finally:
						pass
					try:
						box2 = annotation2[0]['boxes'][i,:]
					except:
						# if out of range for image with less boxes, fill in blank box to feed to other branch of SCINet
						box2 = [0, 0, 0, 0]
					finally:
						pass

					# process for inference with SCINet
					box1_im = self.process_for_inference(box1, width1, height1)
					box2_im = self.process_for_inference(box2, width2, height2)

					# Feed to SCINet
					box1_im = box1_im.view(1, 3, 256, 256)
					box2_im = box2_im.view(1, 3, 256, 256)
					box1_im = box1_im.to(self.device)
					box2_im = box2_im.to(self.device)

					x1_hat, x2_hat, z1, z2 = self.model(box1_im, box2_im)

					# Append applicable results to list 
					if i<num_boxes1:
						output_im1.append(x1_hat)
					if i<num_boxes2:
						output_im2.append(x2_hat)
				

				# calculate boxes with highest IoU
				matched_boxes = {}
				box_list = []
				for i in range(len(output_im1)):
					for j in range(len(output_im2)):
						if annotation1[0]['labels'][i] == annotation2[0]['labels'][j]:
							IoU = self.IoU(output_im1[i], output_im2[j])
							if IoU>0:
								if i in box_list:
									if IoU > matched_boxes[i][0]:
										matched_boxes[i] = (IoU, j)
									else:
										matched_boxes[i] = (IoU, j)
										box1_list.append(i)
				results.append(matched_boxes)

		if len(results)==len(self.test_loader):
			print(results)
			print("DONE")
		else:
			print(len(results), len(self.test_loader))
			print results


if __name__ == "__main__":
	test = Test()
	test.evaluate()






        					



















