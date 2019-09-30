import os
import sys 
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import utils.dataloader as dl
import utils.opencv_utils as cv
import utils.tools as tl

from masker_train import test_masker
from masker_train import COCODataSet
from painter_train import test_painter

img_size=512
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def slice_img(img):
    '''
    img, torch 3 x ih x iw
    remove several objects in the scene and inpaint the gap
    '''
    masks=test_masker(img) # k x 1 x ih x iw
    #randomly sample some mask and put them together
    k=masks.shape[0]
    print(masks.shape)
    rand_index=torch.randint(k,size=(min(3,k),),device=device)
    print(rand_index)
    sub_masks=masks.index_select(dim=0,index=rand_index)
    mask=sub_masks.sum(dim=0).gt(0.0).float() # 1 x ih x iw
    pred=test_painter(img,mask)
    return pred


def test_slice():
    dataset=COCODataSet()
    dataloader=DataLoader(dataset,batch_size=1,shuffle=True,num_workers=1)
    for i,img_batch in enumerate(dataloader):
        imgs=img_batch['img'] # 1 x 3 x ih x iw
        imgs=imgs.to(device)
        pred=slice_img(imgs[0])
        orig_img=tl.recover_img(imgs[0])
        cv.display_img(orig_img)
        pred_orig=tl.recover_img(pred)
        cv.display_img(pred_orig)

if __name__=='__main__':
    test_slice()
