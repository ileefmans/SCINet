from model import SCINet20
import torch
import argparse

# Define argparser
def get_args():
	parser = argparse.ArgumentParser(description = "Model Options")
	parser.add_argument("--size", type=int, default=256, help="desired height and width of image tensor")
	return parser.parse_args()

# Get arguments
args = get_args()
# Create random tensors of desired height and width
input1 = torch.randn(10, 3, args.size, args.size)
input2 = torch.randn(10, 3, args.size, args.size)
# Instantiate model
model = SCINet20()
# Run forward pass of model
output1, output2, z1, z2 = model.forward(input1, input2)

# Print dimensions and whether or not input dimensions match output dimensions
print("\n \nOutput 1 size: {} \nOutput 2 size: {} \nz1 size: {} \nz2 size: {} \n \n".format(output1.size(), output2.size(), z1.size(), z2.size()))
if output1.size()==input1.size() and output2.size()==input2.size():
	print("Input and Output Sizes Match!!! \n \n")
else:
	print("input and Output Sizes DO NOT Match :( \n \n")