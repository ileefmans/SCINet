from preprocess import Image_Process
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision
import matplotlib.pyplot as plt


"""
Script to get example image from Bi-Autoencoder
"""

# function to process an example sample for inference with SCINet2.0
def process_for_inference(box, width, height):
    # Create blank image size of picture
    box_im = Image.new("RGB", (width, height))
    box_im = cv2.cvtColor(np.array(box_im), cv2.COLOR_RGB2BGR)

    # Get dimensions for bboxes
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])

    # Draw bounding boxes

    cv2.rectangle(box_im, (x1, y1), (x2, y2), (0, 255, 0), -1)

    # Transform to PIL Image for preprocessing before entering SCINet
    # box_im = cv2.cvtColor(box_im, cv2.COLOR_BGR2RBG)
    box_im = Image.fromarray(box_im)
    box_im = box_im.convert("RGB")

    # Preprocess
    image_process = Image_Process((256, 256))
    box_im = image_process.expand(box_im)
    transform = torchvision.transforms.ToTensor()
    box_im = transform(box_im)
    box_im = image_process.uniform_size(box_im)

    return box_im


# Load model
model = torch.load(
    "/Users/ianleefmans/Desktop/Insight/Project/Data/model_checkpoints/SCINet_epoch1.pt",
    map_location=torch.device("cpu"),
)

# Set model to evaluate
model.eval()

# Define box coordinates
box1 = (759, 864, 804, 908)
box2 = (596, 663, 637, 711)

# Process boxes
box1_im = process_for_inference(box1, 1440, 1440)
box2_im = process_for_inference(box2, 1440, 1440)
box1_im = box1_im.view(1, 3, 256, 256)
box2_im = box2_im.view(1, 3, 256, 256)

# Forward pass of model
x1, x2, z1, z2 = model(box1_im, box2_im)

# Transform output to PIL Image and print dimensions for evaluating example
trans = torchvision.transforms.ToPILImage()
print("\n \n", x1.size(), "\n \n")
x1 = x1.reshape(3, 256, 256)
print("\n \n", x1.size(), "\n \n")
x1 = trans(x1)

# Save image
x1.save("sampleim.jpg")
