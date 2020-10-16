from preprocess import Image_Process

class ImageProcessTest(Image_Process):
	"""
		Class inheriting functionality of Image_Process, to be run in unitest via Travis
	"""
	def __init__(self):
		super(ImageProcessTest, self).__init__()
		self.test=True