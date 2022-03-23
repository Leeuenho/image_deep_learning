# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from matplotlib import pyplot as plt
import numpy as np
import random
import cv2
import os
from PIL import Image
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import gc

#GPU 이용
gc.collect()
torch.cuda.empty_cache()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
print(device)

data_folder = 'C:/Users/user/OneDrive/바탕 화면/새 폴더/x-ray_segmentation'
data_path = 'C:/Users/user/Downloads/data_set_2.zip'


size = 128 #이미지 크기

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        
        def conv_block(in_channels, out_channels):
            
            block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                            kernel_size=3, stride=1, padding=1, bias = True),
                                  nn.BatchNorm2d(num_features=out_channels),
                                  nn.ReLU())    
            return block
    
        self.num_classes = num_classes
        #contracting path
        self.enc_11 = conv_block(in_channels= 1, out_channels= 64)
        self.enc_12 = conv_block(in_channels= 64, out_channels= 64)
        self.pool_1 = nn.MaxPool2d(kernel_size= 2)
        
        self.enc_21 = conv_block(in_channels= 64, out_channels= 128)
        self.enc_22 = conv_block(in_channels= 128, out_channels= 128)
        self.pool_2 = nn.MaxPool2d(kernel_size= 2)
        
        self.enc_31 = conv_block(in_channels= 128, out_channels= 256)
        self.enc_32 = conv_block(in_channels= 256, out_channels= 256)
        self.pool_3 = nn.MaxPool2d(kernel_size= 2)
        
        self.enc_41 = conv_block(in_channels= 256, out_channels= 512)
        self.enc_42 = conv_block(in_channels= 512, out_channels= 512)
        self.pool_4 = nn.MaxPool2d(kernel_size= 2)
        
        self.enc_51 = conv_block(in_channels= 512, out_channels= 1024)
        
        #expansive path
        
        self.dec_51 = conv_block(in_channels= 1024, out_channels= 512)
        
        self.uppool_4 = nn.ConvTranspose2d(in_channels= 512, out_channels= 512, 
                                           kernel_size= 2, stride= 2, padding= 0, bias= True)
        self.dec_42 = conv_block(in_channels= 2*512, out_channels= 512) #skip_connection으로 이어지는 추가 볼륨이 존재하기 때문에 input channel 2배
        self.dec_41 = conv_block(in_channels= 512, out_channels= 256)
        
        self.uppool_3 = nn.ConvTranspose2d(in_channels= 256, out_channels= 256, 
                                           kernel_size= 2, stride= 2, padding= 0, bias= True)
        self.dec_32 = conv_block(in_channels= 2*256, out_channels= 256)
        self.dec_31 = conv_block(in_channels= 256, out_channels= 128)
        
        self.uppool_2 = nn.ConvTranspose2d(in_channels= 128, out_channels= 128, 
                                           kernel_size= 2, stride= 2, padding= 0, bias= True)
        self.dec_22 = conv_block(in_channels= 2*128, out_channels= 128)
        self.dec_21 = conv_block(in_channels= 128, out_channels= 64)
        
        self.uppool_1 = nn.ConvTranspose2d(in_channels= 64, out_channels= 64, 
                                           kernel_size= 2, stride= 2, padding= 0, bias= True)
        self.dec_12 = conv_block(in_channels= 2*64, out_channels= 64)
        self.dec_11 = conv_block(in_channels= 64, out_channels= 64)
        
        self.output = nn.Conv2d(in_channels=64, out_channels= self.num_classes,
                                 kernel_size= 1, stride= 1, padding= 0, bias= True) 
        



    
    def forward(self, X): #X = input image
        enc_11 = self.enc_11(X)
        enc_12 = self.enc_12(enc_11)
        pool_1 = self.pool_1(enc_12)
        
        enc_21 = self.enc_21(pool_1)
        enc_22 = self.enc_22(enc_21)
        pool_2 = self.pool_2(enc_22)
        
        enc_31 = self.enc_31(pool_2)
        enc_32 = self.enc_32(enc_31)
        pool_3 = self.pool_3(enc_32)
        
        enc_41 = self.enc_41(pool_3)
        enc_42 = self.enc_42(enc_41)
        pool_4 = self.pool_4(enc_42)
        
        enc_51 = self.enc_51(pool_4)
        
        dec_51 = self.dec_51(enc_51)
        
        uppool_4 = self.uppool_4(dec_51)
        cat_4 = torch.cat((uppool_4, enc_42), dim = 4) #skip connection과 uppooling 된 데이터 채널 합
        dec_42 = self.dec_42(cat_4)
        dec_41 = self.dec_41(dec_42)
        
        uppool_3 = self.uppool_3(dec_41)
        cat_3 = torch.cat((uppool_3, enc_32), dim = 1)
        dec_32 = self.dec_32(cat_3)
        dec_31 = self.dec_31(dec_32)
        
        uppool_2 = self.uppool_2(dec_31)
        cat_2 = torch.cat((uppool_2, enc_22), dim = 1)
        dec_22 = self.dec_22(cat_2)
        dec_21 = self.dec_21(dec_22)
        
        uppool_1 = self.uppool_1(dec_21)
        cat_1 = torch.cat((uppool_1, enc_12), dim = 1)
        dec_12 = self.dec_12(cat_1)
        dec_11 = self.dec_11(dec_12)
        
        output = self.output(dec_11)
        
        return output

# 좌우 반전 및 노이즈 이미지 처리
class Seg_Aug_Dataset(Dataset):
        def __init__(self, img_dir, num):
          self.img_dir = img_dir
          self.img_fns = os.listdir(img_dir+"/CXR_png/")
          self.num = num
          
        def __len__(self):
            return len(self.img_fns)
        
        def __getitem__(self, index):
            img_fn = self.img_fns[index]
            if(img_fn.endswith("png")):
    
                p = os.path.join(self.img_dir+"/CXR_png/{}".format(img_fn))
                image = Image.open(p).convert("L")
    
                m_p = os.path.join(self.img_dir+"/Mask/{}".format(img_fn))
                mask = Image.open(m_p).convert("L")
                
                #이미지 증폭 방식 3가지 중 1개 랜덤 진행
                """
                if(self.num == 1):

                    deg = random.randint(-60, 60)
                    image = image.rotate(deg)
                    mask = mask.rotate(deg)
               """

                image = self.transform(image)    
                mask = self.transform(mask)

            
                return image, mask
    
    
        def transform(self, image) :
            transform_ops = transforms.Compose([
                    transforms.Resize((size, size)),
                    transforms.ToTensor()
             ])
            return transform_ops(image)
        '''
for i in range(0, 2):
    #num = random.randint(1,3) #어떤 방식으로 이미지를 증폭할지 랜덤으로 선택(위 함수 if문에 들어갈 수)
    train_aug = []    
    test_aug = []
    train = []
    test = []
    
    data_augmentation_set = Seg_Aug_Dataset(data_folder, i) # 함수 호출
    train_aug, test_aug = train_test_split(data_augmentation_set, test_size = 0.02, shuffle=True, random_state=(random.randint(0, 10000)))
    train.extend(train_aug)
    test.extend(test_aug)
        '''


#네트워크 저장  
def save(pth_dir, model, optim, epoch):
    if not os.path.exists(pth_dir):
        os.makedir(pth_dir)
        
    torch.save({'model : ': model.state_dict(), 'optim': optim.state_dict()},
               "./%s/model_epoch%d.pth" %(pth_dir, epoch))
 
def load(pth_dir,model, optim):
    if not os.path.exists(pth_dir):
        epoch = 0
        return model, optim, epoch
    
    pth_list = os.listdir(pth_dir)
    print(pth_list[-1])
    pth_list.sort(key= lambda f: int(float(''.join(filter(str.isdigit, f)))))
    
    dict_model = torch.load("./%s?%s" % (pth_dir, pth_list[-1]))
    
    model.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(pth_list[-1].split('epoch')[1].split('pth')[0])
    
    return model, optim, epoch

   
from torch.utils.data.sampler import RandomSampler

start_epoch = 0
num_epoch = 10
lr = 1e-3

data_loader_train = DataLoader(train, batch_size=10)

#네트워크 설정
model = UNet(num_classes = 1).to(device = device, dtype=torch.double) 
#criterion = nn.MSELoss().to(device)
criterion = nn.BCEWithLogitsLoss().to(device) #손실함수
optimizer = optim.Adam(model.parameters(), lr = lr)


step_losses = []
epoch_losses = []

pth_dir = 'C:/Users/user/study/test/model'

model, optim, start_epoch = load(pth_dir = pth_dir, model = model, optim = optim)

for epoch in tqdm(range(start_epoch+1, num_epoch+1)) :
  epoch_loss = 0
  model.train()
  for X, Y in tqdm(data_loader_train, total = len(data_loader_train)) :
    X, Y = X.to( device = device, dtype = torch.double), Y.to(device = device, dtype = torch.double)
    Y_pred = model(X)
    #back propagation
    optimizer.zero_grad()
    loss = criterion(Y_pred, Y)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
    step_losses.append(loss.item())
  epoch_losses.append(epoch_loss/len(data_loader_train))
  
  save(pth_dir = pth_dir, model = model, optim=optim, epoch=epoch)
  
