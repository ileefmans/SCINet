from imutils.face_utils.facealigner import FaceAligner
from imutils.face_utils import rect_to_bb
from PIL import Image
import imutils
import argparse
import dlib
import cv2
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description = "Model Options")
    parser.add_argument("-p", "--shape_predictor", required=True, help="path to facial landmark predictor")
    parser.add_argument("-i", "--image", required=True, help="path to input image")   
    return vars(parser.parse_args())



class FaceAlign:
    def __init__(self, sample, predictor):
        """
            Args:
                image (JPEG): Image to be processed
                predictor (dlib predictor): facial landmark predictor from dlib
        """
        self.sample = sample
        self.image = self.sample[0][0]
        self.height = np.size(self.image, 0)
        self.width = np.size(self.image, 1)
        self.annotation = self.sample[1][0]
        self.predictor = predictor

        #initialize face detector, facial landmark predictor, and facial aligner
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor)
        self.face_align = FaceAligner(self.predictor, desiredFaceWidth=256)
        
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        


    def facebox(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to greyscale
        rects = self.detector(gray, 2) # returns a list of bounding boxes around face
        if len(rects) != 1:
            raise Exception("Input Error: detected more than one face in input image, expected one face")
        else:
            (x, y, w, h) = rect_to_bb(rects[0])
            return rects[0]
        
        
    def annotation_extract(self, sample, height, width):
        sample = sample[1][0]
        output = []
        for i in range(sample['boxes'].size(0)):
            x = int(sample['boxes'][i,:][0])
            y = int(sample['boxes'][i,:][1])
            w = int(sample['boxes'][i,:][2])
            h = int(sample['boxes'][i,:][3])
            label = int(sample['labels'][i])
            box_image = Image.new('RGB', (width, height))
            box_image = cv2.cvtColor(np.array(box_image), cv2.COLOR_RGB2BGR)
            cv2.rectangle(box_image,(x,y),(x+w,y+h),(0,255,0),-1)
            output.append((box_image, label))
        return output


    def forward(self):
        facebox = self.facebox(self.image)
        aligned_face = self.face_align.align(self.image, self.gray, facebox)
        box_ims = self.annotation_extract(self.sample, self.height, self.width)
        aligned_boxes = []
        for im in box_ims:
            aligned_box = self.face_align.align(im[0], self.gray, facebox)
            aligned_boxes.append((aligned_box, im[1]))

        return aligned_face, aligned_boxes




class CalculateMatches:
    def __init__(self, image1, image2, box_image1, box_image2):
        """
            Args:

                image1 (opencv image): First image post-alignment
                image2 (opencv image): Second image post-alignment
                box_image1 (opencv image): First image of bounding box post-alignment
                box_image2 (opencv image): Second image of bounding box post-alignment

        """
        self.im1 = image1
        self.im2 = image2
        self.box_im1 = box_image1
        self.box_im2 = box_image2
        
    def calc_IoU(self):
        if self.box_im1[1]!=self.box_im2[1]:
            return False
        union = np.count_nonzero(self.box_im1[0] + self.box_im2[0])
        total  = np.count_nonzero(self.box_im1[0]) + np.count_nonzero(self.box_im2[0])
        intersection = total-union
        IoU = intersection/union
        return IoU
    def calc_confidence(self):
        pass
    def evaluate(self):
        IoU = self.calc_IoU()
        return IoU












