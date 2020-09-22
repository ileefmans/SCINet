import unittest
from preprocess import image_process
import torch
from PIL import Image
import requests


class test_image_process(unittest.TestCase):
    
    def setUp(self):
        response1 = requests.get("https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/LACMTA_Square_Orange_Line.svg/1200px-LACMTA_Square_Orange_Line.svg.png", stream=True).raw
        
        response2 = requests.get("http://www.math.uwaterloo.ca/~hdesterc/websiteW/personal/pictures/argentina2003/200311-set8/images/200311-set8_4_400x600.jpg", stream=True).raw
        response3 = requests.get("https://www.govisithawaii.com/wp-content/uploads/2009/11/image5.png", stream=True).raw
    
        self.image1 = Image.open(response1)
        self.image2 = Image.open(response2)
        self.image3 = Image.open(response3)
    
    
        self.tensor1 = torch.rand(3, 400, 600)
        self.tensor2 = torch.rand(3, 400, 400)
        self.tensor3 = torch.rand(3, 600, 400)
    
    def tearDown(self):
        pass

    def test_expand_square(self):
        self.assertEqual(image_process((400,600)).expand(self.image1).size, (400,400))
        self.assertEqual(image_process((600,600)).expand(self.image1).size, (600,600))
        self.assertEqual(image_process((600,400)).expand(self.image1).size, (400,400))

    def test_expand_tall_rectangle(self):
        self.assertEqual(image_process((300,700)).expand(self.image2).height, 300)
        self.assertEqual(image_process((700,700)).expand(self.image2).height, 700)
        self.assertEqual(image_process((700,300)).expand(self.image2).width, 300)

    def test_expand_long_rectangle(self):
        self.assertEqual(image_process((300,700)).expand(self.image2).height, 300)
        self.assertEqual(image_process((700,700)).expand(self.image1).width, 700)
        self.assertEqual(image_process((700,300)).expand(self.image1).width, 300)

    def test_uniform_size_long_rectangle(self):
        self.assertEqual(image_process((400,700)).uniform_size(self.tensor1).size(), torch.Size([3, 400, 700]))
        self.assertEqual(image_process((500,600)).uniform_size(self.tensor1).size(), torch.Size([3, 500, 600]))


    def test_uniform_size_square(self):
        self.assertEqual(image_process((400,500)).uniform_size(self.tensor2).size(), torch.Size([3, 400, 500]))
        self.assertEqual(image_process((500,400)).uniform_size(self.tensor2).size(), torch.Size([3, 500, 400]))

    def test_uniform_size_short_rectangle(self):
        self.assertEqual(image_process((700,400)).uniform_size(self.tensor3).size(), torch.Size([3, 700, 400]))
        self.assertEqual(image_process((600,500)).uniform_size(self.tensor3).size(), torch.Size([3, 600, 500]))
        print("DONE PREPROCESS TEST")















if __name__ == '__main__':
    unittest.main()





