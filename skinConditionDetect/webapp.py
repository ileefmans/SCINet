import streamlit as st
from PIL import Image
from time import sleep
from facealign import FaceAlign, CalculateMatches
import pandas as pd



def annotation_conversion(total_annotation):
	"""
		Function to convert annotations into a form usable in a deep learning object detection model

		Args:

			annotation (list): list of dictionaries containing all annotation infromation
	"""
	annotation = {}

	count = 0
	for i in total_annotation['annotation']:
		try:
			
			if (i['condition']=='Detected') and ('bounding_boxes' in i.keys()):
				
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
	annotation['labels'] = labels

	return annotation









#disable deprication warning
st.set_option('deprecation.showfileUploaderEncoding', False)


st.title('**SCINet**')
st.markdown(""" #### SCINet uses Artificial Intelligence to identify persisting skin conditions.   
	Follow the directions in the side bar and see your results below.""")

st.sidebar.markdown("### Insert images and annotations below:")
st.sidebar.markdown("- **Image 1:** Earlier image")
st.sidebar.markdown("- **Image 2:** Current image ")
st.sidebar.markdown("- **Annotation:** Json file")


placeholder = st.empty()
placeholder.image(["webapp/Unknown-1.png", "webapp/Unknown-1.png"])

image1 = st.sidebar.file_uploader("Image 1 [.jpg]", type="jpg")

image2 = st.sidebar.file_uploader("Image 2 [.jpg]", type="jpg")

annotation = st.sidebar.file_uploader("Annotation 1 [.json]", type="json")
row_num1 = st.sidebar.number_input("Row 1 in annotation") #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
row_num2 = st.sidebar.number_input("Row 2 in annotation")




if (st.sidebar.button("Run")) and (image1 is not None) and (image2 is not None) and (annotation is not None) and (row_num1 is not None):
	with st.spinner('Wait for it...'):

		sleep(5)
		image1 = Image.open(image1)
		image2 = Image.open(image2)
		df = pd.read_json(annotation)
		total_annotation1 = df.iloc[int(row_num1)].image_details
		total_annotation2 = df.iloc[int(row_num2)].image_details

		
		annotation1 = annotation_conversion(total_annotation1)
		annotation2 = annotation_conversion(total_annotation2)


		#type1 = str(type(image1))

		placeholder.image([image1, image2])

		st.markdown(str(annotation1))

		st.balloons()






