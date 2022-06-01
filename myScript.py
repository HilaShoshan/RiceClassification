import sys
from PIL import Image
import numpy as np
import torch
from torch import nn


class NeuralNetwork(nn.Module):
  def __init__(self):
    super(NeuralNetwork, self).__init__()
    self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
        nn.Linear(image_width*image_height, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, len(img_labels)),
        nn.ReLU()
    )
  
  def forward(self, x):
    x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits


image_width = 250
image_height = 250
img_labels = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']


# get image path
arg_list = sys.argv
# print('Argument List:', arg_list)
path = arg_list[1]  # the second argument (after the file name) is the image path
# print(path)

# retrieve model
model = torch.load('model.pth', torch.device('cpu'))  
# print(model.eval())

# Generate prediction
image = Image.open(path).convert('L')
image = np.array(image)
new_shape = tuple([1]) + tuple([1]) + image.shape  # shape: [N, C, W, H], where N and C = 1
image = np.reshape(image, new_shape)

# convert the image to tensor
input = torch.from_numpy(image).float()

# Predicted class value using argmax
prediction = model(input)
prediction_class = torch.argmax(prediction, 1)
# print(prediction_class)

# print the class
class_idx = prediction_class.numpy()[0]
print(img_labels[class_idx])