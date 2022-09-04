# Name: src.py
# Description: The tools to to train and validate the model
# Author: Behzad Valipour Sh. <behzad.valipour@outlook.com>
# Date: 04.09.2022

'''
lines (17 sloc)  1.05 KB
MIT License
Copyright (c) 2022 Behzad Valipour Sh.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy as np
import pandas as pd
from PIL import Image

import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn, optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader



class LoadImageData(Dataset):
    def __init__(self,dataset_path,transform=T.Compose([T.ToTensor()])):
        # data loading
        self.dataset = pd.read_csv(dataset_path)
        self.n_samples = len(self.dataset)
        # Agumentation
        self.transform = transform
    def __len__(self):
        return self.n_samples

    def __getitem__(self,idx):
        self.X = Image.open((self.dataset.loc[idx,'PATH']))
        self.y = self.dataset.loc[idx,'label']
        
        if self.transform:
            self.X = self.transform(self.X)
            self.y = torch.as_tensor(self.y.astype(np.float32))
        
        return self.X,self.y.unsqueeze(0)
    
    
# Model
class TurbinModelNet(pl.LightningModule):
    def __init__(self,learning_rate = 0.001):
        super(TurbinModelNet,self).__init__()
        
        self.cnn_layers = nn.Sequential(
                                        #Input has a shape 128*128
                                        nn.Conv2d(in_channels=3, out_channels=16*3, kernel_size=3, stride=1),
                                        # 126*126
                                        nn.BatchNorm2d(16*3),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=3, stride=2),
                                        # 62*62
                                        nn.Conv2d(16*3, 32*3, kernel_size=3, stride=1),
                                        # 60*60
                                        nn.BatchNorm2d(32*3),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2, stride=2)
                                        # 30*30
                                        )
        
        self.linear_layers = nn.Sequential(
                                            nn.Linear(32*3*30*30, 120),
                                            nn.ReLU(),
                                            nn.Linear(120, 84),
                                            nn.ReLU(),
                                            nn.Linear(84, 1),
                                            nn.Sigmoid()
                                        )
        
        self.learning_rate = learning_rate
        self.loss = nn.BCELoss()
        self.Accuracy = torchmetrics.F1Score()
        self.Accuracy_val = torchmetrics.F1Score()
        self.Accuracy_test = torchmetrics.F1Score()
        self.save_hyperparameters()
        
    def forward(self,input):
        # print("INPUT:", input.shape)
        x = self.cnn_layers(input)
        # print("FIRST:", x.shape)
        x = x.view(-1,32*3*30*30)
        # print("SECOND:", x.shape)
        output = self.linear_layers(x)
        return output

    def training_step(self, batch, batch_idx):
        x,y = batch
        
        x = self(x)
        loss = self.loss(x,y)
        acc = self.Accuracy(x,y.int())
        self.log('Train_Loss', loss, on_epoch=True, on_step=True)
        self.log('Train_F1_Step',acc)
        return loss
    
    def training_epoch_end(self, outputs):
        self.log('train_F1_epoch', self.Accuracy.compute(),prog_bar=True)
        self.Accuracy.reset()
    
    def validation_step(self, batch, batch_idx):
        x,y = batch
        
        x = self(x)
        loss = self.loss(x,y)
        acc = self.Accuracy_val(x,y.int())
        self.log('validation_Loss', loss, on_epoch=True, on_step=True)
        self.log('validation_F1_Step',acc)
        return loss
    
    def validation_epoch_end(self, outputs):
        self.log('val_F1_epoch', self.Accuracy_val.compute(),prog_bar=True)
        self.Accuracy_val.reset()
        
    def test_step(self, batch, batch_idx):
        x,y = batch
        
        x = self(x)
        loss = self.loss(x,y)
        acc = self.Accuracy_test(x,y.int())
        self.log('Test_Loss', loss, on_epoch=True, on_step=True)
        self.log('Test_F1_Step',acc)
        return loss
    
    def test_epoch_end(self, outputs):
        self.log('test_F1_epoch', self.Accuracy_test.compute())
        self.Accuracy_test.reset()
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.learning_rate)
        return optimizer