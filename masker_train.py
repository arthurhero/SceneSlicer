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
batch_size=2

MAX_ANN_PER_IMG=30

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
    pts=np.asarray(seg,dtype=np.int32)
    pts=pts.reshape((-1,2))
    return pts

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
        coco=self.coco
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
            if ann['iscrowd'] == 1:
                continue
            segs=ann['segmentation']
            mask=np.zeros((height,width,1))
            for seg in segs:
                pts=seg_to_pts(seg)
                mask+=cv.generate_polygon_mask(height,width,pts)
            mask=mask.clip(0,1)
            bbox=np.asarray(ann['bbox'],dtype=np.float32)
            masks.append(mask)
            bboxs.append(bbox)
        bboxs=np.stack(bboxs)
        masks=np.stack(masks)
        masks=masks.transpose(0,3,1,2)
        img=torch.FloatTensor(img) #c*h*w float32
        bboxs=torch.FloatTensor(bboxs) #N*4 float32
        masks=torch.FloatTensor(masks) #N*1*h*w float32
        #trim or pad gt to make them fixed size
        cur_len=bboxs.shape[0]
        if cur_len>MAX_ANN_PER_IMG:
            bboxs=bboxs[:MAX_ANN_PER_IMG]
            masks=masks[:MAX_ANN_PER_IMG]
        elif cur_len<MAX_ANN_PER_IMG:
            bboxs_pad=torch.zeros(MAX_ANN_PER_IMG,4)
            bboxs_pad[:cur_len]=bboxs
            bboxs=bboxs_pad
            masks_pad=torch.zeros(MAX_ANN_PER_IMG,1,height,width)
            masks_pad[:cur_len]=masks
            masks=masks_pad
        #scale img and gt to square
        x_scale=img_size/float(width)
        y_scale=img_size/float(height)
        img=F.interpolate(img.view(1,-1,height,width),size=(img_size,img_size))
        img=img.view(-1,img_size,img_size)
        masks=F.interpolate(masks,size=(img_size,img_size))
        x=bboxs[:,0]
        x=x*x_scale
        y=bboxs[:,1]
        y=y*y_scale
        w=bboxs[:,2]
        w=w*x_scale
        h=bboxs[:,3]
        h=h*y_scale
        bboxs[:,0]=x
        bboxs[:,1]=y
        bboxs[:,2]=w
        bboxs[:,3]=h
        sample={'img':img,'num_ann':min(len(anns),MAX_ANN_PER_IMG),'bboxs':bboxs,'masks':masks}
        return sample 

def test_coco_dataset():
    dataset=COCODataSet()
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=batch_size)
    for i,img_batch in enumerate(dataloader):
        imgs=img_batch['img'].split(1)
        bboxss=img_batch['bboxs'].split(1)
        maskss=img_batch['masks'].split(1)
        num_anns=img_batch['num_ann'].split(1)
        for img,bboxs,masks,num_ann in zip(imgs,bboxss,maskss,num_anns):
            img=img[0]
            bboxs=bboxs[0]
            masks=masks[0]
            num_ann=num_ann[0]
            orig_img=tl.recover_img(img)
            cv.display_img(orig_img)
            bboxs=bboxs.split(1)
            masks=masks.split(1)
            bboxs=bboxs[:num_ann]
            masks=masks[:num_ann]
            for bbox,mask in zip(bboxs,masks):
                bbox=bbox[0]
                mask=mask[0]
                mask_img=img*(1-mask)+(0.2*mask)*img
                mask_img=tl.recover_img(mask_img)
                x=int(round(bbox[0]))
                y=int(round(bbox[1]))
                w=int(round(bbox[2]))
                h=int(round(bbox[3]))
                mask_img=cv.bbox_img(mask_img,x,y,w,h)
                cv.display_img(mask_img)
        if i>3:
            break

test_coco_dataset()
