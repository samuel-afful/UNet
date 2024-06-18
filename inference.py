# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 07:41:43 2024

@author: user
"""
from unet import UNet
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def single_inference(model_path,image_path):
    model = UNet(in_channels=3,num_classes=1)
    model.load_state_dict(torch.load(model_path))
    
    transform = transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor()])
    img = transform(Image.open(image_path))
    
    img = img.unsqueeze(0)
    pred_mask = model(img)
    
    plt.imshow(img)
    img = img.squeeze(0)
    img = img.permute(1,2,0)
    plt.imshow(img)
    
    pred_mask = pred_mask.squeeze(0)
    pred_mask = pred_mask.permute(1,2,0)
    
    pred_mask[pred_mask>0] = 1
    pred_mask[pred_mask<0] = 0
    
    
    
    
    