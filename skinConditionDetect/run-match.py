from facealign import FaceAlign, CalculateMatches


def get_args():
    parser = argparse.ArgumentParser(description = "Model Options")
    parser.add_argument("--predictor", type=str default='shape_predictor_68_face_landmarks.dat', help="facial landmark predictor from dlib")
    #parser.add_argument("-i", "--sample1", required=True, help="path to first input image")   
    #parser.add_argument("-i", "--sample1", required=True, help="path to second input image")  
    parser.add_argument("--local", type=bool, default=False, help="False if running on AWS, True if running locally")
    parser.add_argument("--local_pickle_path", type=str, default="/Users/ianleefmans/Desktop/Insight/Project/Re-Identifying_Persistent_Skin_Conditions/skinConditionDetect/pickle/train_annotation_dict.pkl", help="path to local pickled annotation path dictionary")
	parser.add_argument("--remote_pickle_path", type=str, default="train_annotation_dict.pkl")
	parser.add_argument("--local_data_directory", type=str, default="/Users/ianleefmans/Desktop/Insight/Project/Data", help="Path to data")
	parser.add_argument("--remote_data_directory", type=str, default="<blank>", help="no remote data dictionary applicable")
	parser.add_argument("--shuffle", type=bool, default=True, help="True if dataloader shuffles input samples before batching, False if samples are batched in order")
    return vars(parser.parse_args())





class GeoMatch:
	def __init__(self):

		self.ops = get_args()
		self.predictor = self.ops.predictor
		self.local = self.ops.local
		
