from imutils.face_utils.facealigner import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import argparse
import dlib
import cv2

def get_args():
    parser = argparse.ArgumentParser(description = "Model Options")
    parser.add_argument("-p", "--shape_predictor", required=True, help="path to facial landmark predictor")
    parser.add_argument("-i", "--image", required=True, help="path to input image")   
    return vars(parser.parse_args())

class FaceAlign:
    def __init__(self):
        self.ops = get_args()
        self.shape_predictor = self.ops['shape_predictor']
        self.image = self.ops['image']

        #initialize face detector, facial landmark predictor, and facial aligner
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.shape_predictor)
        self.face_align = FaceAligner(self.predictor, desiredFaceWidth=256)

    def align(self):

        image = cv2.imread(self.image)
        image = imutils.resize(image, width=800)     # resize
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to greyscale

        cv2.imshow("Input", image)
        rects = self.detector(gray, 2) # returns a list of bounding boxes around face

        for rect in rects:
            # extract the ROI of the *original* face, then align the face
            (x, y, w, h) = rect_to_bb(rect)
            faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
            faceAligned = self.face_align.align(image, gray, rect)
            
            cv2.imshow("Original", faceOrig)
            cv2.imshow("Aligned", faceAligned)
            cv2.waitKey(0)