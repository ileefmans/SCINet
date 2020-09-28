import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
import boto3
import argparse
from tqdm import tqdm
import os

from datahelper import CreateDataset, my_collate

##### NEED TO ADD IN LOCAL OPTION IMPORTING MODEL!!!!!

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
	parser.add_argument("--local_save_path", type=str, default="/Users/ianleefmans/Desktop/Insight/Project/Data/model_checkpoints", help="Path to folder to save model parameters after trained")
	parser.add_argument("--access_key", type=str, default="", help="AWS Access Key")
	parser.add_argument("--secret_access_key", type=str, default="", help="AWS Secret Access Key")

	return parser.parse_args()


class Inference:
	"""
		Class for running inference on new data with model
	"""
	def __init__(self):

		self.ops = get_args()
		self.device = torch.device("cuda")
		self.model_version = self.ops.model_version
		self.access_key = self.ops.access_key
		self.secret_access_key = self.ops.secret_access_key
		self.local = self.ops.local


		# Import Model
		if self.model_version==1:
			self.model_name = "FASTERRCNN"
			#self.s3 = boto3.client("s3", aws_access_key_id=self.access_key, aws_secret_access_key=self.secret_access_key)
			#self.obj = self.s3.get_object(Bucket="models-and-checkpoints", Key=os.path.join('models/', self.model_name+".pt"))
			self.model = torch.load(self.model_name+".pt")
			self.model = self.model.cuda()
		else:
			pass

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
		self.shuffle = self.ops.shuffle

		# Instantiate Dataloaders
		self.dataset = CreateDataset(self.pickle_path, self.data_directory, img_size=self.img_size, local=self.local, access_key=self.access_key, secret_access_key=self.secret_access_key, transform=self.transform)
		self.train_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, collate_fn=my_collate)


	def evaluate(self):
		self.model.eval()

		with torch.no_grad():
			for image, annotation in tqdm(self.train_loader):	
				image = [im.cuda() for im in image]
				annotation = [{k: v.cuda() for k, v in t.items()} for t in annotation]
				output = self.model(image)
				print(output)





if __name__ == "__main__":
	inference = Inference()
	inference.evaluate()

