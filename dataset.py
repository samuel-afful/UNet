# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 08:05:53 2024

@author: user
"""

from PIL import Image
import os
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class caravan_Datasets(Dataset):
    def __init__(self,root_path,test=True):
        self.root_path = root_path
        if test:
            self.images = [root_path+'/test/'+i for i in os.listdir(root_path+'/test/')]
            self.masks = [root_path+'/test_mask/'+i for i in os.listdir(root_path+'/test_mask/')]
            
        else:
            self.images = [root_path+'/train/'+i for i in os.listdir(root_path+'/train/')]
            self.masks = [root_path+'/train_mask/'+i for i in os.listdir(root_path+'/train_mask/')]
        
        self.transform = transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor()])
    
    def __getitem__(self,index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")
        
        return self.transform(img), self.transform(mask)
    
    def __len__(self):
        return len(self.images)