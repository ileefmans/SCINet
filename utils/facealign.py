from imutils.face_utils.facealigner import FaceAligner
from imutils.face_utils import rect_to_bb
from PIL import Image
import imutils
import argparse
import dlib
import cv2
import numpy as np





class FaceAlign:
	def __init__(self, sample, predictor, detector="HOG"):
		"""
			Args:
				sample (dataloader obj): Sample from dataloader
				predictor (dlib predictor): facial landmark predictor from dlib
				detector (string): Type of facial detector used "HOG": HOG detector, "CNN": CNN detector
		"""
		self.sample = sample
		self.image = self.sample[0][0]
		self.height = np.size(self.image, 0)
		self.width = np.size(self.image, 1)
		self.annotation = self.sample[1][0]
		self.predictor = predictor
		self.detector = detector

		#initialize face detector, facial landmark predictor, and facial aligner
		if self.detector == "HOG":
			self.detector = dlib.get_frontal_face_detector()
		else:
			self.detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

		self.predictor = dlib.shape_predictor(self.predictor)
		self.face_align = FaceAligner(self.predictor, desiredFaceWidth=256)
		
		self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
	

	def try_detector_rotations(self, image):

		"""
			Args: 
				image (cv2 image): image to be rotated


		Try different rotations until image is in the right orientation
		Issue originates from converting from PIL image to CV2 image

		"""
		rects = self.detector(image, 1)
		angle = 0
		if len(rects)==0:
			image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
			rects = self.detector(image, 1)
			angle = 1
			if len(rects)==0:
				image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
				rects = self.detector(image, 1)
				angle = 2
				if len(rects)==0:
					image = cv2.rotate(image, cv2.ROTATE_180)
					rects = self.detector(image, 1)
					angle = 3
		# return rectangle for bounding box around face, image, and key for angle which image was rotated
		return rects, image, angle



	def facebox(self, image):
		"""
			Args: 
				image (cv2 image): Patient image in which you are trying to detect a face

			Method detects a face in an image and returns the boudning box coordinates around the face

		"""
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to greyscale
		#rects = self.detector(gray, 1) # returns a list of bounding boxes around face
		rects, image, angle = self.try_detector_rotations(gray)
		if len(rects) != 1:
			print(len(rects))
			#raise Exception("Input Error: detected more than one face in input image, expected one face")
			(x, y, w, h) = rect_to_bb(rects[0])
			return rects[0], image, angle
		else:
			(x, y, w, h) = rect_to_bb(rects[0])
			return rects[0], image, angle
		
		
	def annotation_extract(self, sample, height, width):
		"""
			Args:
				sample (dataloader obj): Sample from dataloader 

				height (int): desired height of blank image to be created

				width (int): desired width of the blank image to be created

			Method extracts bounding boxes around skin conditions from a sample in the dataloader
			and then creates a blank image with only the bounding box in the image
		"""
		sample = sample[1][0]
		output = []
		box_list = []
		for i in range(sample['boxes'].size(0)):
			x = int(sample['boxes'][i,:][0])
			y = int(sample['boxes'][i,:][1])
			w = int(sample['boxes'][i,:][2])
			h = int(sample['boxes'][i,:][3])
			label = int(sample['labels'][i])
			box_image = Image.new('RGB', (width, height))
			box_image = cv2.cvtColor(np.array(box_image), cv2.COLOR_RGB2BGR)
			cv2.rectangle(box_image,(x,y),(w,h),(0,255,0),-1)
			output.append((box_image, label))
			box_list.append((x,y,w,h))
		return output, box_list

	def rotate(self, image, angle):
		# rotate image by the desired degree
		if angle==0:
			pass
		elif angle==1:
			image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
		elif angle==2:
			image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
		else: 
			image = cv2.rotate(image, cv2.ROTATE_180)

		return image 



	def forward(self):
		#print("IMAGE TYPE", type(self.image))
		facebox, image, angle = self.facebox(self.image)

		# Rotate gray image same degrees as image was rotated to find facebox
		gray = self.rotate(self.gray, angle)

		aligned_face = self.face_align.align(image, gray, facebox)
		box_ims, box_list = self.annotation_extract(self.sample, self.height, self.width)
		aligned_boxes = []
		for im in box_ims:
			aligned_box = self.face_align.align(self.rotate(im[0], angle), gray, facebox)
			aligned_boxes.append((aligned_box, im[1]))

		return aligned_face, aligned_boxes, box_list




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













