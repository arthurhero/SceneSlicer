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

import utils.dataloader as dl
import utils.opencv_utils as cv
import utils.tools as tl

import masker_ops as mo

img_size=512 #size of input img
epoch=20
lr=0.000005
sgd_momentum=0.9
weight_decay=0.0005

MAX_ANN_PER_IMG=30

batch_size=2 # num of img per batch
r=64 #batch size for proposals from one img
pos_threshold=0.5
sample_size=512
fh=64 #input feature patch size to mask generator
fw=64
mh=64 #output mask size of mask generator
mw=64

k=16 #maximum number of proposals from one img (used during testing)
nms_threshold=0.5 #for filtering proposals (used during testing)

l_cls_alpha=1
l_bbox_alpha=1

fpn_ckpt_path='logs/masker_fpn.ckpt'
rpn_ckpt_path='logs/masker_rpn.ckpt'
class_ckpt_path='logs/class.ckpt'
masker_ckpt_path='logs/masker.ckpt'

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

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
    def __init__(self,min_ratio=0.0,max_ratio=1.0,random_mask=False):
        self.coco=COCO(ann_file)
        self.imgIds=self.coco.getImgIds()
        self.min_ratio=min_ratio
        self.max_ratio=max_ratio
        self.random_mask=random_mask

    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self,idx):
        coco=self.coco
        sample=None
        while True:
            imgId=self.imgIds[idx]
            imgData=coco.loadImgs([imgId])[0]
            img=cv.load_img(data_folder+imgData['file_name'])
            if self.random_mask:
                img=dl.process_img(img,crop_size=img_size,resize=False,sample_num=1,alpha=False,normalize=True,pytorch=True,random_mask=False,ones_boundary=False)
                img=img[0]
                img=torch.FloatTensor(img)
                pts=cv.generate_random_vertices(img_size,img_size,dis_max_ratio=self.max_ratio)
                mask=cv.generate_polygon_mask(img_size,img_size,pts)
                mask=mask.transpose(2,0,1)
                mask=torch.FloatTensor(mask)
                img_mask=torch.cat([img,mask],dim=0) # 4 x ih x iw
                return img_mask
            height=img.shape[0]
            width=img.shape[1]
            img=img.astype(np.float32)
            img=img/128.-1
            if len(img.shape)==2:
                img = np.dstack((img,img,img))
            img=img.transpose(2,0,1)
            annIds=coco.getAnnIds(imgIds=[imgId])
            anns=coco.loadAnns(annIds)
            bboxs=list()
            masks=list()
            labels=list()
            for ann in anns:
                if ann['iscrowd'] == 1:
                    continue
                bbox=np.asarray(ann['bbox'],dtype=np.float32)
                # throw away bbox that are too small
                area=bbox[2]*bbox[3]
                if area/(height*width)<self.min_ratio or area/(height*width)>self.max_ratio:
                    continue
                segs=ann['segmentation']
                mask=np.zeros((height,width,1))
                for seg in segs:
                    pts=seg_to_pts(seg)
                    mask+=cv.generate_polygon_mask(height,width,pts)
                mask=mask.clip(0,1)
                masks.append(mask)
                bboxs.append(bbox)
                labels.append(ann['category_id'])
            if len(bboxs)==0:
                idx=idx+1
                continue
            num_ann=len(bboxs)
            bboxs=np.stack(bboxs)
            masks=np.stack(masks)
            labels=np.stack(labels)
            masks=masks.transpose(0,3,1,2)
            img=torch.FloatTensor(img) #c*h*w float32
            bboxs=torch.FloatTensor(bboxs) #N*4 float32
            masks=torch.FloatTensor(masks) #N*1*h*w float32
            labels=torch.LongTensor(labels) #N
            #trim or pad gt to make them fixed size
            cur_len=bboxs.shape[0]
            if cur_len>MAX_ANN_PER_IMG:
                bboxs=bboxs[:MAX_ANN_PER_IMG]
                masks=masks[:MAX_ANN_PER_IMG]
                labels=labels[:MAX_ANN_PER_IMG]
            elif cur_len<MAX_ANN_PER_IMG:
                bboxs_pad=torch.zeros(MAX_ANN_PER_IMG,4)
                bboxs_pad[:cur_len]=bboxs
                bboxs=bboxs_pad
                masks_pad=torch.zeros(MAX_ANN_PER_IMG,1,height,width)
                masks_pad[:cur_len]=masks
                masks=masks_pad
                labels_pad=torch.zeros(MAX_ANN_PER_IMG)
                labels_pad[:cur_len]=labels
                labels=labels_pad.long()
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
            sample={'img':img,'num_ann':min(num_ann,MAX_ANN_PER_IMG),'bboxs':bboxs,'masks':masks,'labels':labels}
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
        labelss=img_batch['labels'].split(1)
        for img,bboxs,masks,num_ann,labels in zip(imgs,bboxss,maskss,num_anns,labelss):
            img=img[0]
            bboxs=bboxs[0]
            masks=masks[0]
            num_ann=num_ann[0]
            labels=labels[0]
            orig_img=tl.recover_img(img)
            cv.display_img(orig_img)
            bboxs=bboxs.split(1)
            masks=masks.split(1)
            labels=labels.split(1)
            bboxs=bboxs[:num_ann]
            masks=masks[:num_ann]
            labels=labels[:num_ann]
            for bbox,mask,label in zip(bboxs,masks,labels):
                bbox=bbox[0]
                mask=mask[0]
                mask_img=img*(1-mask)+(0.2*mask)*img
                mask_img=tl.recover_img(mask_img)
                y1=int(round(bbox[0].item()))
                x1=int(round(bbox[1].item()))
                y2=int(round(bbox[2].item()))
                x2=int(round(bbox[3].item()))
                mask_img=cv.bbox_img(mask_img,y1,x1,y2,x2)
                cv.display_img(mask_img)
                print(label)
        '''
        if i>3:
            break
            '''

'''
Train the masker
'''
def train_masker(step=1,fresh_fpn=True,visualize=False): # 4 step training
    in_channels=3
    #load dataset
    dataset=COCODataSet()
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=batch_size)

    #modules
    mask_rcnn=mo.MaskRCNN(r,pos_threshold,img_size,img_size,fh,fw,mh,mw,device).to(device)
    mask_rcnn.train()
    if not fresh_fpn:
        if os.path.isfile(fpn_ckpt_path):
            mask_rcnn.fpn.load_state_dict(torch.load(fpn_ckpt_path))
            print("Loaded fpn ckpt!")

    if step==1 or step==2:
    #if step==1:
        mask_rcnn.fpn.apply(tl.unfreeze_params)
    else:
        mask_rcnn.fpn.apply(tl.freeze_params)

    if os.path.isfile(rpn_ckpt_path):
        mask_rcnn.rpn.load_state_dict(torch.load(rpn_ckpt_path))
        print("Loaded rpn ckpt!")
    if os.path.isfile(class_ckpt_path):
        mask_rcnn.class_generator.load_state_dict(torch.load(class_ckpt_path))
        print("Loaded class ckpt!")
    if os.path.isfile(masker_ckpt_path):
        mask_rcnn.mask_generator.load_state_dict(torch.load(masker_ckpt_path))
        print("Loaded masker ckpt!")
    if step==1 or step==3:
        mask_rcnn.rpn.apply(tl.unfreeze_params)
        mask_rcnn.class_generator.apply(tl.freeze_params)
        mask_rcnn.mask_generator.apply(tl.freeze_params)
    else:
        mask_rcnn.rpn.apply(tl.freeze_params)
        mask_rcnn.class_generator.apply(tl.unfreeze_params)
        mask_rcnn.mask_generator.apply(tl.unfreeze_params)

    label_assigner=mo.AssignClsLabel(pos_threshold).to(device)
    optimizer = torch.optim.Adam(mask_rcnn.parameters(),lr=lr,betas=(0.5,0.9))

    for e in range(epoch):
        j=0
        l_cls_sum=0.0
        l_bbox_sum=0.0
        l_class_sum=0.0
        acc_sum=0.0
        l_mask_sum=0.0

        for i,img_batch in enumerate(dataloader):
            imgs=img_batch['img'] # B x 3 x ih x iw
            gt_bboxess=img_batch['bboxs'] # B x A x 4
            gt_maskss=img_batch['masks'] # B x A x 1 x ih x iw
            gt_counts=img_batch['num_ann'] # B x 1
            gt_labelss=img_batch['labels'] # B x A (all 91 classes, 1-index)
            imgs=imgs.to(device)
            gt_bboxess=gt_bboxess.to(device)
            gt_maskss=gt_maskss.to(device)
            gt_counts=gt_counts.to(device)
            gt_labelss=gt_labelss.to(device)

            if step==1 or step==3: # train rpn
                scoress,bboxess,anchorss=mask_rcnn(imgs,True)
                labels=label_assigner(anchorss,gt_bboxess,gt_counts)
                sample_idxs,sample_counts=mo.sample_proposals(labels,sample_size)
                if sample_idxs is None:
                    print("no positive anchors")
                    continue
                l_cls=mo.calc_cls_score_loss(scoress,sample_idxs,sample_counts)
                l_bbox=mo.calc_bbox_loss(bboxess,anchorss,gt_bboxess,gt_counts,sample_idxs,sample_counts)
                loss=l_cls_alpha*l_cls+l_bbox_alpha*l_bbox
                l_cls_sum+=l_cls.item()
                l_bbox_sum+=l_bbox.item()
            else:
                classes,maskss=mask_rcnn(imgs,False,gt_bboxess,gt_counts,gt_labelss)
                l_class,acc=mo.calc_class_loss(classes,gt_labelss,gt_counts)
                l_mask=mo.calc_mask_loss(maskss,gt_maskss,gt_counts,visualize=visualize)
                #loss=l_class
                loss=l_mask+l_class/10.0
                l_class_sum+=l_class.item()
                l_mask_sum+=l_mask.item()
                acc_sum+=acc
            optimizer.zero_grad()
            loss.backward()
            '''
            print("loss:",loss)
            print("sum:",l_class_sum)
            '''
            #print(mask_rcnn.class_generator.conv3[0].weight.grad.data)
            '''
            if step==2:
                print(mask_rcnn.mask_generator.conv1[0].weight.grad.data)
                '''
            optimizer.step()
            if j%100==0:
                if step==1 or step==2:
                    torch.save(mask_rcnn.fpn.state_dict(), fpn_ckpt_path)
                if step==1 or step==3:
                    torch.save(mask_rcnn.rpn.state_dict(), rpn_ckpt_path)
                    print('Epoch [{}/{}] , Step {}, l_cls: {:.4f}, l_bbox: {:.4f}'
                            .format(e+1,epoch,j,l_cls_sum/100.0,l_bbox_sum/100.0))
                    l_cls_sum=0.0
                    l_bbox_sum=0.0
                else:
                    torch.save(mask_rcnn.class_generator.state_dict(), class_ckpt_path)
                    torch.save(mask_rcnn.mask_generator.state_dict(), masker_ckpt_path)
                    print('Epoch [{}/{}] , Step {}, l_class: {:.4f}, accuracy: {:.4f}, l_mask: {:.4f}'
                            .format(e+1,epoch,j,l_class_sum/100.0,acc_sum/100.0,l_mask_sum/100.0))
                    l_class_sum=0.0
                    l_mask_sum=0.0
                    acc_sum=0.0

            j+=1

def test_masker():
    in_channels=3
    #load dataset
    dataset=COCODataSet()
    dataloader=DataLoader(dataset,batch_size=1,shuffle=True,num_workers=1)

    #modules
    mask_rcnn=mo.MaskRCNN(r,pos_threshold,img_size,img_size,fh,fw,mh,mw,device).to(device)
    mask_rcnn.eval()

    if os.path.isfile(fpn_ckpt_path):
        mask_rcnn.fpn.load_state_dict(torch.load(fpn_ckpt_path))
        print("Loaded fpn ckpt!")
    if os.path.isfile(rpn_ckpt_path):
        mask_rcnn.rpn.load_state_dict(torch.load(rpn_ckpt_path))
        print("Loaded rpn ckpt!")
    if os.path.isfile(masker_ckpt_path):
        mask_rcnn.mask_generator.load_state_dict(torch.load(masker_ckpt_path))
        print("Loaded masker ckpt!")

    for i,img_batch in enumerate(dataloader):
        imgs=img_batch['img'] # 1 x 3 x ih x iw
        imgs=imgs.to(device)
        '''
        gt_bboxess=img_batch['bboxs'] # 1 x A x 4
        gt_maskss=img_batch['masks'] # 1 x A x 1 x ih x iw
        gt_counts=img_batch['num_ann'] # 1 x 1
        gt_bboxess=gt_bboxess.to(device)
        gt_maskss=gt_maskss.to(device)
        gt_counts=gt_counts.to(device)
        '''

        scoress,bboxess,anchorss=mask_rcnn(imgs,True)
        # filter out k proposals
        proposal_filter=mo.ProposalFilter(k,nms_threshold)
        bboxess_f,counts_f=proposal_filter(scoress,bboxess) # 1 x K x 4, 1 x 1
        # predict masks
        maskss=mask_rcnn.predict_mask(bboxess_f,counts_f) # 1 x K x 1 x ih x iw
        visualize_masks(imgs[0],bboxess_f[0][:counts_f[0]],maskss[0][:counts_f[0]])
        if i==1:
            break

def visualize_masks(img,bboxs,masks):
    '''
    img - 3 x ih x iw
    bboxs - K x 4
    masks - K x 1 x ih x iw
    '''
    orig_img=tl.recover_img(img)
    cv.display_img(orig_img)
    bboxs=bboxs.split(1)
    masks=masks.split(1)
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

#test_coco_dataset()
#train_masker(2,fresh_fpn=False,visualize=False)
