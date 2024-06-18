# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 09:39:20 2024

@author: user
"""
from unet import UNet
from torch import nn,optim
from torch.utils.data import DataLoader,random_split
from dataset import caravan_Datasets
import torch
from tqdm import tqdm

if __name__ == '__main__':
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 4
    EPOCHS = 2
    DATA_PATH = 'C:/Users/user/UNET/data'
    MODEL_SAVE_PATH = 'C:/Users/user/UNET/models/unet.pth'
    
    #Loading data
    generator = torch.Generator().manual_seed(42)
    train_data = caravan_Datasets(DATA_PATH)
    
    train_set, test_set = random_split(train_data, [0.8,0.2], generator=generator)
    train_dataloader = DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
    test_dataloader = DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=True)
    
    
    
    #Loading model
    model =UNet(in_channels=3, num_classes=1) 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
        
    
    def train_loop(dataloader, model, criterion, optimizer):
        #size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    
    
        model.train()
        for idx,img_mask in enumerate(tqdm(train_dataloader)):
            img = img_mask[0].float()
            mask = img_mask[1].float()
        # Compute prediction and loss
            pred = model(img)
            loss = criterion(pred, mask)
            train_loss = loss.item()
            
        # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        trained_loss = train_loss / idx+1    
        print(f"Train loss each epochs :{trained_loss:.4f}")    
                #loss, current = loss.item()+idx * BATCH_SIZE + len(img_mask)
                #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
        model.eval()
        test_loss, correct = 0, []
        model.train()
        with torch.no_grad():
            for idx,img_mask in enumerate(tqdm(test_dataloader)):
                img = img_mask[0].float()
                mask = img_mask[1].float()
                pred_img = model(img)
                correct.appen(pred_img)
                test_loss += loss_fn(pred_img, mask).item()
            tested_loss = test_loss /idx+1
            print(f"Train loss each epochs :{tested_loss:.4f}")  

    #less begin training      
    for t in tqdm(range(EPOCHS)):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(dataloader=train_dataloader, model=model,criterion=criterion , optimizer=optimizer)
        test_loop(dataloader=test_dataloader, model=model,criterion=criterion)
        
        print("Done!")
    torch.save(model, MODEL_SAVE_PATH)
    
    
    
   
   

       