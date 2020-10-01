from facealign import FaceAlign, CalculateMatches


def get_args():
    parser = argparse.ArgumentParser(description = "Model Options")
    parser.add_argument("-p", "--shape_predictor", required=True, help="path to facial landmark predictor")
    parser.add_argument("-i", "--image1", required=True, help="path to first input image")   
    parser.add_argument("-i", "--image1", required=True, help="path to second input image")  
    return vars(parser.parse_args())





class GeoMatch:
	def __init__(self):
