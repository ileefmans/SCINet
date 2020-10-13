from facealign import FaceAlign, CalculateMatches
from datahelper import CreateDataset, my_collate
import argparse
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd


# Create argument parser
def get_args():
	parser = argparse.ArgumentParser(description = "Model Options")
	parser.add_argument("--predictor", type=str, default='shape_predictor_68_face_landmarks.dat', help="facial landmark predictor from dlib")
	parser.add_argument("--detector", type=str, default='HOG', help='type of facial detector model used, "HOG" for HOG detector, "CNN" for CNN detector') 
	parser.add_argument("--local", type=bool, default=False, help="False if running on AWS, True if running locally")
	parser.add_argument("--local_pickle_path", type=str, default="/Users/ianleefmans/Desktop/Insight/Project/SCINet/skinConditionDetect/pickle/simple_train_dict.pkl", help="path to local pickled annotation path dictionary")
	parser.add_argument("--remote_pickle_path", type=str, default="simple_train_dict.pkl")
	parser.add_argument("--local_data_directory", type=str, default="/Users/ianleefmans/Desktop/Insight/Project/Data", help="Path to data")
	parser.add_argument("--remote_data_directory", type=str, default="<blank>", help="no remote data dictionary applicable")
	parser.add_argument("--shuffle", type=bool, default=False, help="True if dataloader shuffles input samples before batching, False if samples are batched in order")
	parser.add_argument("--batch_size", type=int, default=1, help="Minibatch size")
	parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")
	parser.add_argument("--geometric", type=bool, default=True, help="True: return samples condusive for geometric transform, False: return smaples condusive for deep learning")
	parser.add_argument("--access_key", type=str, default="", help="AWS Access Key")
	parser.add_argument("--secret_access_key", type=str, default="", help="AWS Secret Access Key")

	return parser.parse_args()





class RunSCINet10:
	"""
		Class running SCINet 1.0 (geometric matching) on existing dataset
	"""
	def __init__(self):

		# Import argparser and get args
		self.ops = get_args()
		self.predictor = self.ops.predictor
		self.detector = self.ops.detector
		self.local = self.ops.local

		# Define attributes for local vs remote
		if self.local==False:
			self.pickle_path = self.ops.remote_pickle_path
			self.data_directory = self.ops.remote_data_directory

		else:
			self.pickle_path = self.ops.local_pickle_path
			self.data_directory = self.ops.local_data_directory
			
		self.shuffle = self.ops.shuffle
		self.batch_size = self.ops.batch_size
		self.num_workers = self.ops.num_workers
		self.transform = torchvision.transforms.ToTensor()
		self.geometric = self.ops.geometric
		self.access_key = self.ops.access_key
		self.secret_access_key = self.ops.secret_access_key



		# Initialize train loader
		self.trainset = CreateDataset(self.pickle_path, self.data_directory, local=self.local, geometric=self.geometric, access_key=self.access_key, secret_access_key=self.secret_access_key, transform=self.transform)
		self.train_loader = DataLoader(dataset=self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, collate_fn=my_collate)


	def euclidean_distance(self, x1, y1, x2, y2):
		"""
			Helper function calculating euclidean distance
		"""
		return ((x1-x2)**2 + (y1-y2)**2)**(1/2)


	def confidence(self, landmarks1, landmarks2, box1, box2):
		"""
			Args:
				landmarks1 ()
			Helper function to calculate confidence for matched boxes
		"""
		total_distance = 0
		count = 0
		for i in range(len(landmarks1)):
			distance1 = self.euclidean_distance(box1[0], box1[1], landmarks1[i][0], landmarks1[i][1])
			distance2 = self.euclidean_distance(box1[0], box1[1], landmarks2[i][0], landmarks2[i][1])
			distance = abs(distance1-distance2)
			normalized_distance = distance/((256**2 +256**2)**(1/2))
			total_distance += normalized_distance
			count+=1
		avg_distance =  total_distance/count

		confidence = 1 -  avg_distance
		return confidence


	def evaluate(self, sample1, sample2, landmarks1, landmarks2):
		fa1 = FaceAlign(sample1, self.predictor, self.detector)
		fa2 = FaceAlign(sample2, self.predictor, self.detector)

		image1, boxes1, box_list1 = fa1.forward()
		image2, boxes2, box_list2 = fa2.forward()

		box1_list = []
		matched_boxes = {}
		for i in range(len(boxes1)):
			for j in range(len(boxes2)):
				ca = CalculateMatches(image1, image2, boxes1[i], boxes2[j])
				IoU = ca.evaluate()
				if IoU>0:
					confidence = self.confidence(landmarks1[0], landmarks2[0], box_list1[i], box_list2[j])

					metric = 0.5*IoU + 0.5*confidence
					if i in box1_list:
						if metric> matched_boxes[i][0]:
							matched_boxes[i] = (metric, IoU, confidence, j)
					else:
						matched_boxes[i] = (metric, IoU, confidence, j)
						box1_list.append(i)

		return matched_boxes





	def run(self):
		results =[]
		metric = []
		IoU = []
		confidence = []
		data = {'metric': metric, 'IoU': IoU, 'confidence': confidence}
		df = pd.DataFrame(data=data)
		for sample in tqdm(self.train_loader):
			sample1 = (sample[0], sample[2])
			sample2 = (sample[1], sample[3])
			landmarks1 = sample[4]
			landmarks2 = sample[5]
			

			try:
				# Evaluate matches
				matched_boxes = self.evaluate(sample1, sample2, landmarks1, landmarks2)

				# If boxes were matched append metric, IoU and confidence to dataframe
				if len(matched_boxes)>0:
					for i in matched_boxes:
						print("IN LOOP")
						new_row = {'metric': matched_boxes[i][0], 'IoU': matched_boxes[i][1], 'confidence': matched_boxes[i][2]}
						print("NEW_ROW LEN:", len(new_row))
						df = df. append(new_row, ignore_index=True)
						print("LEN DF:", len(df))
						# Save or overwrite csv file with each iteration
						df.to_csv("table_of_metrics.csv")

				# Append results to list
				results.append(matched_boxes)

			except:
				#print("NO FACE FOUND")
				pass
			finally:
				pass

			

		if len(results)==len(self.train_loader):
			print(results)
			print("DONE")
		else:
			print(results)
			print(len(results), len(self.train_loader))
		return results



	def calc_performance(self):
		results = self.run()
		count = 0
		total_metric = 0
		for i in results:
			for  j in i:
				total_metric+=i[j][0]
				count+=1
		avg_metric = total_metric/count

		print("{} Observations: {}".format(count, avg_metric))
		return avg_metric




if __name__ == "__main__":
	runSCINet = RunSCINet10()
	runSCINet.calc_performance()





