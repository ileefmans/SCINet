from facealign import FaceAlign, CalculateMatches
from datahelper2 import CreateDataset, my_collate
import argparse
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_args():
	parser = argparse.ArgumentParser(description = "Model Options")
	parser.add_argument("--predictor", type=str, default='shape_predictor_68_face_landmarks.dat', help="facial landmark predictor from dlib")
	#parser.add_argument("-i", "--sample1", required=True, help="path to first input image")   
	#parser.add_argument("-i", "--sample1", required=True, help="path to second input image")  
	parser.add_argument("--local", type=bool, default=False, help="False if running on AWS, True if running locally")
	parser.add_argument("--local_pickle_path", type=str, default="/Users/ianleefmans/Desktop/Insight/Project/Re-Identifying_Persistent_Skin_Conditions/skinConditionDetect/pickle/simple_train_dict.pkl", help="path to local pickled annotation path dictionary")
	parser.add_argument("--remote_pickle_path", type=str, default="simple_train_dict.pkl")
	parser.add_argument("--local_data_directory", type=str, default="/Users/ianleefmans/Desktop/Insight/Project/Data", help="Path to data")
	parser.add_argument("--remote_data_directory", type=str, default="<blank>", help="no remote data dictionary applicable")
	parser.add_argument("--shuffle", type=bool, default=True, help="True if dataloader shuffles input samples before batching, False if samples are batched in order")
	parser.add_argument("--batch_size", type=int, default=1, help="Minibatch size")
	parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")
	parser.add_argument("--geometric", type=bool, default=True, help="True: return samples condusive for geometric transform, False: return smaples condusive for deep learning")
	parser.add_argument("--access_key", type=str, default="", help="AWS Access Key")
	parser.add_argument("--secret_access_key", type=str, default="", help="AWS Secret Access Key")

	return parser.parse_args()





class GeoMatch:
	def __init__(self):

		self.ops = get_args()
		self.predictor = self.ops.predictor
		self.local = self.ops.local
		print('\n \n', self.local, '\n \n' )

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
		return ((x1-x2)**2 + (y1-y2)**2)**(1/2)


	def confidence(self, landmarks1, landmarks2, box1, box2):
		total_distance = 0
		for i in range(len(landmarks1)):
			distance1 = self.euclidean_distance(box1[0], box1[1], landmarks1[i][0], landmarks1[i][1])
			distance2 = self.euclidean_distance(box1[0], box1[1], landmarks1[i][0], landmarks1[i][1])
			total_distance+=abs(distance1-distance2)
		avg_distance = total_distance/((256**2 +256**2)**(1/2))
		confidence = 1 -  avg_distance
		return confidence





	def run(self):
		results =[]
		for sample in tqdm(self.train_loader):
			sample1 = (sample[0], sample[2])
			sample2 = (sample[1], sample[3])
			landmarks1 = sample[4]
			landmarks2 = sample[5]

			fa1 = FaceAlign(sample1, self.predictor)
			fa2 = FaceAlign(sample2, self.predictor)

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
								matched_boxes[i] = (metric, j)
						else:
							matched_boxes[i] = (metric, j)
							box1_list.append(i)

			results.append(matched_boxes)

		if len(results)==len(self.train_loader):
			print(results)
			print("DONE")
		else:
			print(len(results), len(self.train_loader))


if __name__ == "__main__":
	geomatch = GeoMatch()
	geomatch.run()






