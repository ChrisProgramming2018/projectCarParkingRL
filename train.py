import torch 
import torch.nn.functional as F
from torch import nn
import numpy as np 
import math 
import torchvision
from torch.autograd import Variable


import pandas as pd
import cv2
from transformer_v2 import Semi_Transformer

print(torch.__version__)
"""
vidcap = cv2.VideoCapture('_-Z6wFjXtGQ.mkv')
success, image = vidcap.read()

t_frame = np.stack(image)
print(t_frame.shape)
success, image = vidcap.read()
t_frame = np.stack((t_frame, image))
for i in range(62):
    success, image = vidcap.read()
    image = np.expand_dims(image, axis=0)
    t_frame = np.vstack((image, t_frame))
    

print(t_frame.shape)
"""
model = Semi_Transformer(num_classes=625, seq_len=64)
model = model.double()
torch_a = torch.ones((1, 64, 3, 400, 400), dtype=torch.float32)
print("start training")
print("")
print("")
print("")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model(torch_a.type(torch.DoubleTensor))

print("DOne")

