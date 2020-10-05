import os
#from google.cloud import storage
import pickle
import torch
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from preprocess import Image_Process
import boto3
import cv2



# Define collate function for dataloader
def my_collate(batch):
    image0 = [item[0] for item in batch]
    image1 = [item[1] for item in batch]
    annotation0 = [item[2] for item in batch]
    annotation1 = [item[3] for item in batch]
    return image0, image1, annotation0, annotation1

def my_collate2(batch):
    image0 = [item[0] for item in batch]
    image1 = [item[1] for item in batch]
    image0 = torch.stack(image0)
    image1 = torch.stack(image1)
    annotation0 = [item[2] for item in batch]
    annotation1 = [item[3] for item in batch]
    return image0, image1, annotation0, annotation1


# Define class for creating and uploaded pickled annotation dictionary
class Annotation_Dict:
	"""
		Class for creating and importing pickled dictionary with 
	"""
	def __init__(self, pickle_file_name):
		"""
			Args:

			pickle_file_name (string): name of desired pickle file, format: '<name>.pkl"
			annotation_directory (string): Directory where anotation json files are kept

		"""
		self.pickle_file_name = pickle_file_name
		#self.filepath = filepath
		


	# def set_pickle(self):
	# 	annotation_dict = {}
	# 	count = 0 
	# 	for filename in tqdm(os.listdir(self.filepath)):
	# 		df = pd.read_json(self.filepath+filename)
	# 		df = df.loc[(df.image_details.notnull()) or (df.image_path.notnull()),:].reset_index()
	# 		for i in range(len(df)):
	# 			annotation_dict[count] = (filename, i)
	# 			count+=1
	# 	with open(self.pickle_file_name, 'wb') as handle:
	# 		pickle.dump(followup_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

	def get_pickle(self):
		with open(self.pickle_file_name, 'rb') as handle:
			dictionary = pickle.load(handle)
			return dictionary





class CreateDataset(torch.utils.data.Dataset):
	
	"""
		Creates iterable dataset as subclass of Pytorch Dataset
	"""
	
	
	def __init__(self, pickle_path, data_directory, img_size = (256, 256), local=1, access_key="", secret_access_key="", geometric=False, transform=None):
		
		"""
			Args:
			
			pickle_path (string): Path to pickle file containing annotation dictionary

			data_directory (string): Directory where data is kept

			img_size (tuple): Size to convert all images to (height, width)

			local (boolean): True if running on local machine, False if running on AWS

			access_key (string): Access Key for AWS

			secret_access_key (string) Secret Access Key for AWS
			
			transform (callable, optional): Optional transform to be applied on a sample
		"""

		self.pickle_path = pickle_path
		self.data_dir = data_directory
		self.img_size = img_size
		self.local = local
		if self.local==1:
			self.local=True
		else:
			self.local=False
		self.access_key = access_key
		self.secret_access_key = secret_access_key
		self.geometric = geometric
		self.transform = transform
		#self.s3 = boto3.client('s3')

		# get pickled dictionary for annotation paths
		self.annotation_source = Annotation_Dict(self.pickle_path)
		self.annotation_dict = self.annotation_source.get_pickle()

		# crate dictionary for labelings of bounding boxes
		self.label_dict = {"pimple-region":1, "come-region":2, 
						   "darkspot-region":3, "ascar-region":4, "oscar-region":5, 
						   "darkcircle":6}

		# instantiate custom preprocessing class
		self.Image_Process = Image_Process(self.img_size)




	def destring(self, list_string):
		"""
			Function for converting a list of strings containing bounding box coordinates to tensors

			Args:

				list_string (list): List of strings to be converted
		"""
		out = []
		for i in list_string:
			x = i.split(',')
			x = [float(j) for j in x]
			out.append(x)
		out = torch.tensor(out) 
		return out     



	def annotation_conversion(self, total_annotation):
		"""
			Function to convert annotations into a form usable in a deep learning object detection model

			Args:

				annotation (list): list of dictionaries containing all annotation infromation
		"""
		annotation = {}

		count = 0
		for i in total_annotation['annotation']:
			try:
				#print("ENTERING TRY")
				if (i['condition']=='Detected') and ('bounding_boxes' in i.keys()):
					#print("ABOUT TO DESTRING")
					box = self.destring(i['bounding_boxes'])
					#print("DESTRING DONE!!!")
					#print(box)
					label = torch.ones([box.size(0)], dtype=torch.int64)*self.label_dict[i['label']]
					if count == 0:
						boxes = box
						labels = label
					else:
						boxes = torch.cat((boxes, box), 0) 
						labels = torch.cat((labels, label), 0)
					count+=1
			except:
				pass
			finally:
				pass
 		
		annotation['boxes'] = boxes
		annotation['labels'] = labels

		return annotation

	

	def __len__(self):
		#return len(self.annotation_dict)
		return 2

	def __getitem__(self, index):

		# open annotated json as dataframe and get corresponding image path
		if self.local is True:
			annotation_df1 = pd.read_json(os.path.join(self.data_dir, 'followup_data/', self.annotation_dict[index][0][0]))
			annotation_df2 = pd.read_json(os.path.join(self.data_dir, 'followup_data/', self.annotation_dict[index][1][0]))
		else:
			s3 = boto3.client("s3", aws_access_key_id=self.access_key, aws_secret_access_key=self.secret_access_key)
			obj11 = s3.get_object(Bucket="followup-annotated-data", Key=os.path.join('followup_data/', self.annotation_dict[index][0][0]))
			obj12 = s3.get_object(Bucket="followup-annotated-data", Key=os.path.join('followup_data/', self.annotation_dict[index][1][0]))
			annotation_df1 = pd.read_json(obj11['Body'])
			annotation_df2 = pd.read_json(obj12['Body'])


		
		image_path1 = annotation_df1.iloc[self.annotation_dict[index][0][1]].image_path
		image_path2 = annotation_df2.iloc[self.annotation_dict[index][1][1]].image_path

		# extract bounding boxes and labels corresponding to image
		total_annotation1 = annotation_df1.iloc[self.annotation_dict[index][0][1]].image_details
		total_annotation2 = annotation_df2.iloc[self.annotation_dict[index][1][1]].image_details
		annotation1 = self.annotation_conversion(total_annotation1)
		annotation2 = self.annotation_conversion(total_annotation2)


		
		if self.local is True:
			if self.geometric==True:
				image1 = cv2.imread(os.path.join(self.data_dir, 'images', image_path1))
				image2 = cv2.imread(os.path.join(self.data_dir, 'images', image_path2))
				return image1, image2, annotation1, annotation2
			else:
				image1 =  Image.open(os.path.join(self.data_dir, 'images', image_path1))
				image2 =  Image.open(os.path.join(self.data_dir, 'images', image_path2))



		else:
			s3 = boto3.client("s3", aws_access_key_id=self.access_key, aws_secret_access_key=self.secret_access_key)
			obj21 = s3.get_object(Bucket="followup-annotated-data", Key=image_path1)
			obj22 = s3.get_object(Bucket="followup-annotated-data", Key=image_path2)

####--------------------------here

			if self.geometric==True:
				image1 = cv2.imread(obj21['Body'])
				image2 = cv2.imread(obj22['Body'])
				return image1, image2, annotation1, annotation2
			else:
				image1 = Image.open(obj21['Body'])
				image2 = Image.open(obj22['Body'])

		if image1.mode != 'RGB':
			image1 = image1.convert('RGB')
		if image2.mode != 'RGB':
			image2 = image2.convert('RGB')

		

		image1 = self.Image_Process.expand(image1)
		image2 = self.Image_Process.expand(image2)
		if self.transform:
			image1 = self.transform(image1)
			image2 = self.transform(image2)
		image1 = self.Image_Process.uniform_size(image1)
		image2 = self.Image_Process.uniform_size(image2)



		return image1, image2, annotation1, annotation2



		
		









