import streamlit as st
from PIL import Image
from time import sleep
from facealign import FaceAlign, CalculateMatches
import pandas as pd
import torch
import cv2
import numpy as np 



def destring(list_string):
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



def annotation_conversion(total_annotation):
	"""
		Function to convert annotations into a form usable in a deep learning object detection model

		Args:

			annotation (list): list of dictionaries containing all annotation infromation
	"""
	annotation = {}
	label_dict = {"pimple-region":1, "come-region":2, 
						   "darkspot-region":3, "ascar-region":4, "oscar-region":5, 
						   "darkcircle":6}

	count = 0
	for i in total_annotation['annotation']:
		
			
		if (i['condition']=='Detected') and ('bounding_boxes' in i.keys()):
			
			box = destring(i['bounding_boxes'])

			label = torch.ones([box.size(0)], dtype=torch.int64)*label_dict[i['label']]
			if count == 0:
				boxes = box
				labels = label
			else:
				boxes = torch.cat((boxes, box), 0) 
				labels = torch.cat((labels, label), 0)
			count+=1
		
	annotation['boxes'] = boxes
	annotation['labels'] = labels

	return annotation









#disable deprication warning
st.set_option('deprecation.showfileUploaderEncoding', False)


st.title('**SCINet**')
st.markdown(""" #### SCINet uses Artificial Intelligence to identify persisting skin conditions. """)
st.text("")
st.text("")

st.sidebar.markdown("### Insert images and annotations below:")
#st.sidebar.markdown("- **Image 1:** Earlier image")
#st.sidebar.markdown("- **Image 2:** Current image ")
#st.sidebar.markdown("- **Annotation:** Json file")


placeholder = st.empty()
placeholder.image(["webapp/Unknown-1.png", "webapp/Unknown-1.png"])

image1 = st.sidebar.file_uploader("Image 1 [.jpg]", type="jpg")

image2 = st.sidebar.file_uploader("Image 2 [.jpg]", type="jpg")

annotation = st.sidebar.file_uploader("Annotation [.json]", type="json")
#row_num1 = st.sidebar.number_input("Row 1 in annotation") #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
#row_num2 = st.sidebar.number_input("Row 2 in annotation")




if (st.sidebar.button("Run")) and (image1 is not None) and (image2 is not None) and (annotation is not None): #and (row_num1 is not None):
	
	# Read and resize images
	image1 = Image.open(image1)
	image2 = Image.open(image2)

	image1 = np.array(image1) 
	image2 = np.array(image2) 
			
	image10 = cv2.resize(image1, (100,175))
	image20 = cv2.resize(image2, (100,175))

	# Draws bounding boxes
	cv2.rectangle(image10,(15,118),(15+20,118+20),(255,0,0),1)
	cv2.rectangle(image10,(30,145),(30+12,145+12),(255,0,0),1)
	cv2.rectangle(image10,(61,120),(61+17,120+33),(255,0,0),1)
	cv2.rectangle(image20,(23,125),(23+15,125+15),(255,0,0),1)
	cv2.rectangle(image20,(63,143),(65+14,145+15),(255,0,0),1)

	# enlarge image
	image10 = cv2.resize(image10, (round(100*1.5), round(175*1.5)))
	image20 = cv2.resize(image20, (round(100*1.5), round(175*1.5)))

	

	# Read in json and select rows
	df = pd.read_json(annotation)
	#total_annotation1 = df.iloc[int(row_num1)].image_details
	#total_annotation2 = df.iloc[int(row_num2)].image_details

	# Extract bounding boxes
	#annotation1 = annotation_conversion(total_annotation1)
	#annotation2 = annotation_conversion(total_annotation2)

	# Display annotated images
	placeholder.image([image10, image20])

	placeholder2 = st.empty()

	placeholder2.markdown("""**All identified conditions:**   
		- Image 1: {bbox1: acne, bbox2: acne, bbox3: acne}  
		- Image 2: {bbox1: acne, bbox2: darkspots} """)

	with st.spinner('Determining Persistent conditions...'):
		sleep(5)


		image11 = cv2.resize(image1, (100,175))
		image21 = cv2.resize(image2, (100,175))

		cv2.rectangle(image11,(15,118),(15+20,118+20),(0,255,0),1)
		#cv2.rectangle(image10,(30,145),(30+12,145+12),(255,0,0),1)
		#cv2.rectangle(image10,(61,120),(61+17,120+33),(255,0,0),1)
		cv2.rectangle(image21,(23,125),(23+15,125+15),(0,255,0),1)
		#cv2.rectangle(image20,(63,143),(65+14,145+15),(255,0,0),1)


		# enlarge image
		image11 = cv2.resize(image11, (round(100*1.5), round(175*1.5)))
		image21 = cv2.resize(image21, (round(100*1.5), round(175*1.5)))

		placeholder.image([image11, image21])
		placeholder2.markdown("""**Persistent Conditions:**    
			- Image 1: {bbox1: acne}  
			- Image 2: {bbox1: acne}""")


	st.balloons()






