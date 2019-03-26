'''
A network similar to Mask R-CNN using Feature Pyramid Networks (FPN)
It takes in an square image and generates K object masks (each mask is of img size)
'''

import os
import sys
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import utils.opencv_utils as cv
import utils.tools as tl

k=8 #maximum number of masks from one img
img_size=256 #size of input img

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''
MSCOCO loader
'''
coco_folder='dataset/MSCOCO/'
ann_folder=coco_folder+'annotations/'
data_type='train2017'
data_folder=coco_folder+data_type+'/'

ann_file=ann_folder+'instances_'+data_type+'.json'

from utils.pycocotools.coco import COCO

def seg_to_pts(seg):
    '''turn the list of points from coco annotation to np pts (Nx2)'''
    pts=np.asarray(pts,dtype=np.int32)
    pts=pts.reshape((-1,2))

class COCODataSet(Dataset):
    '''
    COCO Dataset (only return segmentation and bbox)
    '''
    def __init__(self):
        self.coco=COCO(ann_file)
        self.imgIds=self.coco.getImgIds()

    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self,idx):
        imgId=self.imgIds[idx]
        imgData=coco.loadImgs([imgId])[0]
        img=cv.load_img(data_folder+imgData['file_name'])
        annIds=coco.getAnnIds(imgIds=[imgId])
        anns=coco.loadAnns(annIds)
        bboxs=list()
        masks=list()
        height=img.shape[0]
        width=img.shape[1]
        img=img.astype(np.float32)
        img=img/128.-1
        img=img.transpose(2,0,1)
        for ann in anns:
            pts=seg_to_pts(ann['segmentation'])
            mask=cv.generate_polygon_mask(height,width,pts)
            bbox=np.asarray(ann['bbox'],dtype=np.float32)
            masks.append(mask)
            bboxs.append(bbox)
        bboxs=np.asarray(bboxs)
        masks=np.asarray(masks)
        img=torch.from_numpy(img) #c*h*w float32
        bboxs=torch.from_numpy(bboxs) #N*4 float32
        masks=torch.from_numpy(masks) #N*h*w float32
        sample={'img':img,'bboxs':bboxs,'masks':masks}
        return sample 
