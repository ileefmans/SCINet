import os
#from google.cloud import storage
import pickle
import torch
from PIL import Image
import pandas as pd
from tqdm import tqdm


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

        """
        self.pickle_file_name = pickle_file_name


    def set_pickle(self):
        annotation_dict = {}
        count = 0 
        for filename in tqdm(os.listdir('followup_data')):
            df = pd.read_json('followup_data/'+filename)
            for i in range(len(df)):
                annotation_dict[count] = (filename, i)
                count+=1
        with open(self.pickle_file_name, 'wb') as handle:
            pickle.dump(annotation_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_pickle(self):
        with open(self.pickle_file_name, 'rb') as handle:
            dictionary = pickle.load(handle)
            return dictionary





class CreateDataset(torch.utils.data.Dataset):
    
    """
        Creates iterable dataset as subclass of Pytorch Dataset
    """
    
    
    def __init__(self, dir_pickle, local=True, transform=None):
        
        """
            Args:
            
            image_blobs (list): List of directories of all image blobs in Google Cloud Storage
            
            transform (callable, optional): Optional transform to be applied on a sample
        """

        self.dir_pickle = dir_pickle
        self.local = local
        self.transform = transform
        self.annotation_source = Annotation_Dict(self.dir_pickle)
        self.annotation_dict = self.annotation_source.get_pickle()


        
    

    def __len__(self):
        return len(self.annotation_dict)

    def __getitem__(self, index):
        
        annotation_df = pd.read_json(os.path.join('followup_data/', self.annotation_dict[index][0]))
        annotation = annotation_df.iloc[self.annotation_dict[index][1]].image_details
        image_path = annotation_df.iloc[self.annotation_dict[index][1]].image_path

        if self.local == False:
            image = Image.open(image_path)
        else:
            image =  Image.open(os.path.join('images', image_path))

        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        instance = {'image': image, 'annotation': annotation}

        return instance



        
        









