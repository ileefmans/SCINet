import os
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch import nn
import torchvision


class Image_Process:
    """
    Class for preprocessing images
        
        """
    def __init__(self, desired_dimensions):
    
        """
        Args:
            desired_dimensions (tuple): (height, width) Desired height and width for all images to conform to.
            
            """
    
        self.new_size = desired_dimensions
    

    def expand(self, image, resample=0):
        """
        Args:
        
            image (Pil image): image that is to be expanded
        
            resample: (see Pil.Image.resize for more documentation):
                0 :Nearest neighbors (default)
                PIL.Image.BILINEAR
                Image.BICUBIC
                Image.LANCZOS
                
        Method to expand or reduce size of image keeping aspect ration intact
            """

        # Calculate new width and new height
        new_width = round((image.width/image.height)*self.new_size[0])
        new_height = round((image.height/image.width)*self.new_size[1])
    
        # set height and width for expanding image while keeping height width ration intact
        if new_width<= self.new_size[1]:
            new_height = self.new_size[0]
        elif new_height<= self.new_size[0]:
            new_width = self.new_size[1]
        else:
            raise Exception("Error in expand() go back and check how images are resized")
    
        # expand image
        image = image.resize((new_width, new_height), resample=resample )
        return image



    def uniform_size(self, x):
        """
        Args:
            
            x (pytorch tensor): image tensor to be conformed to be padded

        Method to add padding to make pictures uniform size    
            
            """
        height = self.new_size[0]
        width = self.new_size[1]
        tup_val1 = round(((width-x.size()[2])/2)-.1)
        tup_val2 = round(((width-x.size()[2])/2)+.1)
        tup_val3 = round(((height-x.size()[1])/2)-.1)
        tup_val4 = round(((height-x.size()[1])/2)+.1)
    
        tup = (tup_val1, tup_val2, tup_val3, tup_val4)
    
        pad = nn.ZeroPad2d(tup)
        padded = pad(x)
        return padded





