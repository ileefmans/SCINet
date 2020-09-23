import os
#from google.cloud import storage
import pickle
import torch
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from preprocess import Image_Process


#def get_direclists():
    #client = storage.Client.from_service_account_json('ian-access-key.json')
    #image_list = list(client.list_blobs('followup_annotated_data', prefix='images'))
    #annotation_list = list(client.list_blobs('followup_annotated_data', prefix='followup_data'))

    #return image_list, annotation_list

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
        


    def set_pickle(self):
        annotation_dict = {}
        count = 0 
        for filename in tqdm(os.listdir("followup_data")):
            df = pd.read_json("followup_data"+filename)
            for i in range(len(df)):
                annotation_dict[count] = (filename, i)
                count+=1
        with open(self.pickle_file_name, 'wb') as handle:
            pickle.dump(followup_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_pickle(self):
        with open(self.pickle_file_name, 'rb') as handle:
            dictionary = pickle.load(handle)
            return dictionary





class CreateDataset(torch.utils.data.Dataset):
    
    """
        Creates iterable dataset as subclass of Pytorch Dataset
    """
    
    
    def __init__(self, pickle_path, data_directory, img_size = (1000,1000), local=True, transform=None):
        
        """
            Args:
            
            pickle_path (string): Path to pickle file containing annotation dictionary

            data_directory (string): Directory where data is kept
            
            transform (callable, optional): Optional transform to be applied on a sample
        """

        self.pickle_path = pickle_path
        self.data_dir = data_directory
        self.img_size = img_size
        self.local = local
        self.transform = transform
        self.annotation_source = Annotation_Dict(self.pickle_path)
        self.annotation_dict = self.annotation_source.get_pickle()
        self.label_dict = {"pimple-region":1, "come-region":2, 
                           "darkspot-region":3, "ascar-region":4, "oscar-region":5, 
                           "darkcircle":6}
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
                if i['condition']=='Detected':
                    box = self.destring(i['bounding_boxes'])
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
        annotation['labels'] =  labels

        return annotation

    

    def __len__(self):
        #return len(self.annotation_dict)
        return 10000

    def __getitem__(self, index):
        
        annotation_df = pd.read_json(os.path.join(self.data_dir, 'followup_data/', self.annotation_dict[index][0]))
        image_path = annotation_df.iloc[self.annotation_dict[index][1]].image_path
        total_annotation = annotation_df.iloc[self.annotation_dict[index][1]].image_details
        annotation = self.annotation_conversion(total_annotation)


        if self.local == False:
            image = Image.open(image_path)
        else:
            image =  Image.open(os.path.join(self.data_dir, 'images', image_path))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = self.Image_Process.expand(image)
        if self.transform:
            image = self.transform(image)
        image = self.Image_Process.uniform_size(image)
        
        #instance = {'image': image, 'annotation': annotation}

        return image, annotation



        
        









