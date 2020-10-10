from datahelper2 import CreateDataset, my_collate
from torch.utils.data import DataLoader
import torchvision
import torch
from torch.utils.data import DataLoader
import argparse



def get_args():
	parser = argparse.ArgumentParser(description = "Model Options")
	parser.add_argument("--predictor", type=str, default='shape_predictor_68_face_landmarks.dat', help="facial landmark predictor from dlib")
	#parser.add_argument("-i", "--sample1", required=True, help="path to first input image")   
	#parser.add_argument("-i", "--sample1", required=True, help="path to second input image")  
	parser.add_argument("--local", type=bool, default=False, help="False if running on AWS, True if running locally")
	parser.add_argument("--local_pickle_path", type=str, default="pickle/simple_train_dict.pkl", help="path to local pickled annotation path dictionary")
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


ops = get_args()

print("\n \n \n ", "Local:", ops.local, "Geometric:", ops.geometric, "Pickle Path:", ops.remote_pickle_path, "\n \n \n")


# Create dataset
dataset = CreateDataset(ops.remote_pickle_path, ops.remote_data_directory, local=ops.local, geometric=ops.geometric, access_key=ops.access_key, secret_access_key=ops.secret_access_key, transform=torchvision.transforms.ToTensor())


# def my_collate(batch):
#     image0 = [item[0] for item in batch]
#     image1 = [item[1] for item in batch]
#     annotation0 = [item[2] for item in batch]
#     annotation1 = [item[3] for item in batch]
#     landmark0 = [item[4] for item in batch]
#     landmark1 = [item[5] for item in batch]

#     return image0, image1, annotation0, annotation1, landmark0, landmark1

print(len(dataset[0]))


train_loader = DataLoader(dataset=dataset, batch_size=ops.batch_size, num_workers=ops.num_workers, shuffle=ops.shuffle, collate_fn=my_collate)
sample = iter(train_loader).next()

print(sample[0])
print("\n \n \n \n")
print(sample[1])
print("\n \n \n \n")
print(sample[2])
print("\n \n \n \n")
print(sample[3])
print("\n \n \n \n")
print(sample[4])
print("\n \n \n \n")
print(sample[5])
#print(sample[0][0].size(), type(sample[1][0]))



