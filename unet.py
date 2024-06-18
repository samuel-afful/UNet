import torch.nn as nn

from unet_part import DoubleConv, DownSample, UpSample


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)
        
        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
       down_1, p1 = self.down_convolution_1(x)
       down_2, p2 = self.down_convolution_2(p1)
       down_3, p3 = self.down_convolution_3(p2)
       down_4, p4 = self.down_convolution_4(p3)

       b = self.bottle_neck(p4)

       up_1 = self.up_convolution_1(b, down_4)
       up_2 = self.up_convolution_2(up_1, down_3)
       up_3 = self.up_convolution_3(up_2, down_2)
       up_4 = self.up_convolution_4(up_3, down_1)

       out = self.out(up_4)
       return out
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
if __name__ == "main":
    import torch
    import torch.nn as nn
    from torchinfo import summary
    from torchvision import transforms
    from PIL import Image
    import shutil
    from unet_part import DoubleConv
    import matplotlib.pyplot as plt
    from unet import UNet
    
    model = UNet(in_channels=3,num_classes=1)
    model.load_state_dict(torch.load('models/unet.pth'))
    transform = transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor()])
    img = transform(Image.open('data/train/0ed6904e1004_01.jpg'))
    img_sq = img.unsqueeze(0).cpu().detach()
    img_sq.shape 
    
    pred = model(img_sq)
    pred.shape 
    pred = pred.squeeze(0).cpu().detach()
    pred_per = pred.permute(1,2,0)
    pred_per.shape    
    
    pred_per[pred_per>0] = 1
    pred_per[pred_per<0] = 0

    plt.imshow(pred_per,cmap='gray')
    
    img_per = img.permute(1,2,0)
    img_per.shape
    
    plt.imshow(img_per)
        
    double_conv = DoubleConv(3,10)
    input_image = torch.rand((9,3,512,512))
    double_conv(input_image)
    
    m = nn.Conv3d(in_channels=16, out_channels=3, kernel_size=3)
    summary(model=m, input_size=(1,16,9,240,240)) 
    
    summary(model=double_conv,input_size=(1,3,256,256,1))    
   
    sq = input_image.squeeze()
    
    sq.shape    
    model = UNet(in_channels=2,num_classes=3)
    #output = model(input_image,)
    summary(model,input_size=(1, 2, 240, 240),col_names=["input_size", "output_size", "num_params"]) 
    
    print(model)
    
    
    import os
    train = []
    mask  = []
    
    images = os.listdir(path='pick_up')
    images    
    
    for i ,image in enumerate(images):
        if image.endswith('.jpg'):
            src = os.path.join('pick_up', image)
            des = os.path.join('train', image)
            shutil.copy(src, des)
        else:
            src_mask = os.path.join('pick_up', image)
            des_mask = os.path.join('train_mask', image)
            shutil.copy(src_mask, des_mask)
            
        
        
        
        
        
        
        
        
    
        if i%2==0:
            shutil.move(src, dst)
            img = Image.open(image)
            train.append(image)
        else:
            mask.append(image)
  
    
  
    for image in train:
        #print(image)
        shutil.move(image,'train')
 
    
    
    
    
    
    
