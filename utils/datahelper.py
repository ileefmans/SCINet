import os
import pickle
import torch
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from preprocess import Image_Process
import boto3
import cv2
import matplotlib.pyplot as plt


def my_collate(batch):
    """
    Custom collate function for dataloader if using SCINet1.0
    """
    image0 = [item[0] for item in batch]
    image1 = [item[1] for item in batch]
    annotation0 = [item[2] for item in batch]
    annotation1 = [item[3] for item in batch]
    landmark0 = [item[4] for item in batch]
    landmark1 = [item[5] for item in batch]
    return image0, image1, annotation0, annotation1, landmark0, landmark1


def my_collate2(batch):
    """
    Custom collate function for dataloader if using SCINet2.0
    """
    image0 = [item[0] for item in batch]
    image1 = [item[1] for item in batch]
    image0 = torch.stack(image0)
    image1 = torch.stack(image1)
    annotation0 = [item[2] for item in batch]
    annotation1 = [item[3] for item in batch]
    landmark0 = [item[4] for item in batch]
    landmark1 = [item[5] for item in batch]

    return image0, image1, annotation0, annotation1, landmark0, landmark1


# Define class for uploading pickled annotation dictionary
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

    def get_pickle(self):
        with open(self.pickle_file_name, "rb") as handle:
            dictionary = pickle.load(handle)
            return dictionary


class CreateDataset(torch.utils.data.Dataset):

    """
    Creates iterable dataset as subclass of Pytorch Dataset
    """

    def __init__(
        self,
        pickle_path,
        data_directory,
        img_size=(256, 256),
        local=1,
        access_key="",
        secret_access_key="",
        geometric=False,
        transform=None,
    ):

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
        if self.local == 1:
            self.local = True
        else:
            self.local = False
        self.access_key = access_key
        self.secret_access_key = secret_access_key
        self.geometric = geometric
        self.transform = transform

        # get pickled dictionary for annotation paths
        self.annotation_source = Annotation_Dict(self.pickle_path)
        self.annotation_dict = self.annotation_source.get_pickle()

        # crate dictionary for labelings of bounding boxes
        self.label_dict = {
            "pimple-region": 1,
            "come-region": 2,
            "darkspot-region": 3,
            "ascar-region": 4,
            "oscar-region": 5,
            "darkcircle": 6,
        }

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
            # split on comma
            x = i.split(",")
            # list comprehension to turn list of strings in to bounding box coordinates
            x = [float(j) for j in x]
            out.append(x)
        # turn bounding box list into a pytorch tensor
        out = torch.tensor(out)
        return out

    def annotation_conversion(self, total_annotation):
        """
        Function to convert annotations into a form usable in a deep learning object detection model

        Args:

                annotation (list): list of dictionaries containing all annotation infromation
        """

        # Initialize annotatin dictionary
        annotation = {}
        count = 0
        for i in total_annotation["annotation"]:

            # Error handling to catch cases where there are no bounding boxes provided etc
            try:
                if (i["condition"] == "Detected") and ("bounding_boxes" in i.keys()):
                    box = self.destring(i["bounding_boxes"])
                    label = (
                        torch.ones([box.size(0)], dtype=torch.int64)
                        * self.label_dict[i["label"]]
                    )
                    if count == 0:
                        # Start creating boxes and labels for each sample
                        boxes = box
                        labels = label
                    else:
                        # stack the box and label tensors
                        boxes = torch.cat((boxes, box), 0)
                        labels = torch.cat((labels, label), 0)
                    count += 1
            except:
                pass
            finally:
                pass
        # create dictionary for annotation with boxes and labels
        annotation["boxes"] = boxes
        annotation["labels"] = labels

        return annotation

    def get_landmarks(self, total_annotation):
        """
        Args:
                total_annotation: full annotation for each sample

        Method to extract facial landmarks for each image
        """

        # List of landmarks to be used
        landmarks = [
            "RIGHT_EYE_RIGHT_CORNER",
            "LEFT_EYE_LEFT_CORNER",
            "NOSE_BOTTOM_CENTER",
            "MOUTH_LEFT",
            "MOUTH_RIGHT",
        ]

        # Initialize anchor list
        anchor_list = []
        for i in range(len(landmarks)):
            anchors = total_annotation["landmarks"][landmarks[i]]
            # extract x and y coordinates and turn into ordered pair
            x = anchors["x"]
            y = anchors["y"]
            anchor = (x, y)
            anchor_list.append(anchor)

        return anchor_list

    def __len__(self):
        """
        Overide the length function of torch.utils.data.Dataset class

        Note that this returns the length of the dataset, changing this results in a change in the
        size of the dataset. Very easy way to alter the dataset for testing examples etc
        """
        return len(self.annotation_dict)

    def __getitem__(self, index):
        """
        Overide the indexing function of torch.utils.data.Dataset class
        """

        # open annotated json as dataframe and get corresponding image path
        if self.local is True:
            annotation_df1 = pd.read_json(
                os.path.join(
                    self.data_dir, "followup_data/", self.annotation_dict[index][0][0]
                )
            )
            annotation_df2 = pd.read_json(
                os.path.join(
                    self.data_dir, "followup_data/", self.annotation_dict[index][1][0]
                )
            )
        else:
            s3 = boto3.client(
                "s3",
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_access_key,
            )
            obj11 = s3.get_object(
                Bucket="followup-annotated-data",
                Key=os.path.join("followup_data/", self.annotation_dict[index][0][0]),
            )
            obj12 = s3.get_object(
                Bucket="followup-annotated-data",
                Key=os.path.join("followup_data/", self.annotation_dict[index][1][0]),
            )
            annotation_df1 = pd.read_json(obj11["Body"])
            annotation_df2 = pd.read_json(obj12["Body"])

        # image paths
        image_path1 = annotation_df1.iloc[self.annotation_dict[index][0][1]].image_path
        image_path2 = annotation_df2.iloc[self.annotation_dict[index][1][1]].image_path

        # extract bounding boxes and labels corresponding to image
        total_annotation1 = annotation_df1.iloc[
            self.annotation_dict[index][0][1]
        ].image_details
        total_annotation2 = annotation_df2.iloc[
            self.annotation_dict[index][1][1]
        ].image_details
        annotation1 = self.annotation_conversion(total_annotation1)
        annotation2 = self.annotation_conversion(total_annotation2)
        landmark1 = self.get_landmarks(total_annotation1)
        landmark2 = self.get_landmarks(total_annotation2)

        # get images using image paths

        # For running locally
        if self.local is True:

            # For SCINet1.0
            if self.geometric == True:
                image1 = cv2.imread(os.path.join(self.data_dir, "images", image_path1))
                image2 = cv2.imread(os.path.join(self.data_dir, "images", image_path2))
                # return data formatted for SCINet1.0
                return image1, image2, annotation1, annotation2, landmark1, landmark2

            # For SCINet2.0
            else:
                image1 = Image.open(os.path.join(self.data_dir, "images", image_path1))
                image2 = Image.open(os.path.join(self.data_dir, "images", image_path2))

        # For running remotely on AWS
        else:
            s3 = boto3.client(
                "s3",
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_access_key,
            )
            obj21 = s3.get_object(Bucket="followup-annotated-data", Key=image_path1)
            obj22 = s3.get_object(Bucket="followup-annotated-data", Key=image_path2)

            # For SCINet1.0
            if self.geometric == True:
                image1 = Image.open(obj21["Body"])
                image2 = Image.open(obj22["Body"])
                image1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
                image2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)

                # A large number of images need to be rotated after they are transformed from a PIL
                # image to a cv2 image. This is not the case for all images though. This problem is
                # further addressed in facealign.py
                image1 = cv2.rotate(image1, cv2.ROTATE_90_COUNTERCLOCKWISE)
                image2 = cv2.rotate(image2, cv2.ROTATE_90_COUNTERCLOCKWISE)
                # return data formatted for SCINet1.0
                return image1, image2, annotation1, annotation2, landmark1, landmark2

            # For SCINet2.0
            else:
                image1 = Image.open(obj21["Body"])
                image2 = Image.open(obj22["Body"])

        # Make sure all images are RGB
        if image1.mode != "RGB":
            image1 = image1.convert("RGB")
        if image2.mode != "RGB":
            image2 = image2.convert("RGB")

        # Expand image
        image1 = self.Image_Process.expand(image1)
        image2 = self.Image_Process.expand(image2)

        # Trasform to torch tensor
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        # Make images a uniform size
        image1 = self.Image_Process.uniform_size(image1)
        image2 = self.Image_Process.uniform_size(image2)

        # return data formatted for SCINet2.0
        return image1, image2, annotation1, annotation2, landmark1, landmark2
