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

import masker_ops as mo

img_size=512 #size of input img
batch_size=2
epoch=5
lr=0.02
sgd_momentum=0.9
weight_decay=0.0001

MAX_ANN_PER_IMG=30

k=200 #maximum number of masks from one img
nms_threshold=0.5
pos_threshold=0.5
sample_size=512
fh=14 #input feature patch size to mask generator
fw=14
mh=28 #output mask size of mask generator
mw=28

masker_ckpt_path='logs/masker.ckpt'

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
        sample=None
        while True:
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
            if len(img.shape)==2:
                img = np.dstack((img,img,img))
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
            if len(bboxs)==0:
                idx=idx+1
                continue
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
            x1=bboxs[:,0]
            x1=x1*x_scale
            y1=bboxs[:,1]
            y1=y1*y_scale
            w=bboxs[:,2]
            w=w*x_scale
            h=bboxs[:,3]
            h=h*y_scale
            bboxs[:,0]=y1
            bboxs[:,1]=x1
            bboxs[:,2]=y1+h
            bboxs[:,3]=x1+w
            sample={'img':img,'num_ann':min(len(anns),MAX_ANN_PER_IMG),'bboxs':bboxs,'masks':masks}
            break
        return sample 

'''
Test COCO Dataloader
'''
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
                y1=int(round(bbox[0]))
                x1=int(round(bbox[1]))
                y2=int(round(bbox[2]))
                x2=int(round(bbox[3]))
                mask_img=cv.bbox_img(mask_img,y1,x1,y2,x2)
                cv.display_img(mask_img)
        if i>3:
            break

'''
Train the masker
'''
def train_masker():
    in_channels=3
    #load dataset
    dataset=COCODataSet()
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=batch_size)

    #modules
    mask_rcnn=mo.MaskRCNN(k,nms_threshold,img_size,img_size,fh,fw,mh,mw,device).to(device)
    mask_rcnn.train()
    label_assigner=mo.AssignClsLabel(pos_threshold).to(device)
    optimizer = torch.optim.SGD(mask_rcnn.parameters(),lr=lr,momentum=sgd_momentum,weight_decay=weight_decay)

    #load ckpt
    if os.path.isfile(masker_ckpt_path):
        mask_rcnn.load_state_dict(torch.load(masker_ckpt_path))
        print("Loaded masker ckpt!")

    for e in range(epoch):
        step=0
        for i,img_batch in enumerate(dataloader):
            imgs=img_batch['img'] # B x 3 x ih x iw
            gt_bboxess=img_batch['bboxs'] # B x A x 4
            gt_maskss=img_batch['masks'] # B x A x 1 x ih x iw
            gt_counts=img_batch['num_ann'] # B x 1
            imgs=imgs.to(device)
            gt_bboxess=gt_bboxess.to(device)
            gt_maskss=gt_maskss.to(device)
            gt_counts=gt_counts.to(device)

            scoress,bboxess,anchorss,bboxess_f,maskss_f,prop_counts,prop_idxs=mask_rcnn(imgs)
            labels=label_assigner(bboxess,gt_bboxess,gt_counts)
            if labels is None:
                #print("got 0 pos bbox!")
                continue
            sample_idxs,sample_counts=mo.sample_proposals(labels,sample_size)
            l_cls=mo.calc_cls_score_loss(scoress,sample_idxs,sample_counts)
            l_bbox=mo.calc_bbox_loss(bboxess,anchorss,gt_bboxess,gt_counts,sample_idxs,sample_counts)
            loss=l_cls+l_bbox

            l_mask=l_bbox.new(1).zero_()
            if bboxess_f is not None:
                #print("got "+str(prop_counts[0].item())+" proposal!")
                labels_f=label_assigner(bboxess_f,gt_bboxess,gt_counts,prop_counts) #get labels for filtered proposals
                if labels_f is not None:
                    sample_idxs_f,sample_counts_f=mo.sample_proposals(labels_f,sample_size,prop_counts) #sample pos from fil
                    l_mask=mo.calc_mask_loss(bboxess_f,maskss_f,prop_counts,gt_bboxess,gt_maskss,gt_counts,sample_idxs_f,sample_counts_f,visualize=True)
                    loss=loss+l_mask
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step%10==0:
                torch.save(mask_rcnn.state_dict(), masker_ckpt_path)
                print('Epoch [{}/{}] , Step {}, Loss: {:.4f}, l_cls: {:.4f}, l_bbox: {:.4f}, l_mask: {:.4f}'
                            .format(e+1,epoch,step,loss.item(),l_cls.item(),l_bbox.item(),l_mask.item()))
            step+=1

#test_coco_dataset()
train_masker()
