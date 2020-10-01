from facealign import FaceAlign, CalculateMatches
from datahelper2 import CreateDataset, my_collate


def get_args():
    parser = argparse.ArgumentParser(description = "Model Options")
    parser.add_argument("--predictor", type=str default='shape_predictor_68_face_landmarks.dat', help="facial landmark predictor from dlib")
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

    return vars(parser.parse_args())





class GeoMatch:
	def __init__(self):

		self.ops = get_args()
		self.predictor = self.ops.predictor
		self.local = self.ops.local

		if self.local==False:
			self.pickle_path = self.ops.remote_pickle_path
			self.data_directory = self.ops.remote_data_directory
			self.save_path = self.ops.remote_save_path
		else:
			self.pickle_path = self.ops.remote_pickle_path
			self.data_directory = self.ops.remote_data_directory
			self.save_path = self.ops.remote_save_path
		self.shuffle = shuffle
		self.batch_size = self.ops.batch_size
		self.num_workers = self.ops.num_workers
		self.transform = torchvision.transforms.ToTensor()

		# Initialize train loader
		self.trainset = CreateDataset(self.pickle_path, self.data_directory, local=self.local, geometric=self.geometric, transform=self.transform)
		self.train_loader = DataLoader(dataset=self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, collate_fn=my_collate)

	def run(self):
		results =[]
		for sample in tqdm(train_loader):
			sample1 = (sample[0], sample[2])
			sample2 = (sample[1], sample[3])

			fa1 = FaceAlign(sample1, predictor)
			fa2 = FaceAlign(sample2, predictor)

			image1, boxes1 = fa1.forward()
			image2, boxes2 = fa2.forward()

			box1_list = []
			matched_boxes = {}
			for i in range(len(boxes1)):
			    for j in range(len(boxes2)):
			        ca = CalculateMatches(image1, image2, boxes1[i], boxes2[j])
			        IoU = ca.evaluate()
			        if IoU>0:
			            if i in box1_list:
			                if IoU > matched_boxes[i][0]:
			                    matched_boxes[i] = (IoU, j)
			            else:
			                matched_boxes[i] = (IoU, j)
			                box1_list.append(i)


			results.append(maxed_boxes)

		if len(results)==len(train_loader):
			print("DONE")
		else:
			print(len(results), len(train_loader))


if __name__ == "__main__":
	geomatch = GeoMatch()
	geomatch.run()






