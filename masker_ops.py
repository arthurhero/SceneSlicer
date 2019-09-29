'''
Layers and operations for masker
including FPN(ResNet50 backbone), RPN head, ROIAlign, mask generator, losses calculator, etc.
'''

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torchvision.models as models

import utils.opencv_utils as cv
import utils.tools as tl

class FeaturePyramidNet(nn.Module):
    def __init__(self):
        super(FeaturePyramidNet, self).__init__()
        resnet50=models.resnet50(pretrained=True)
        self.conv1=resnet50.conv1
        self.bn1=resnet50.bn1
        self.relu=resnet50.relu
        self.maxpool=resnet50.maxpool
        self.conv2=resnet50.layer1 #256 x 1/4 x 1/4
        self.conv3=resnet50.layer2 #512 x 1/8 x 1/8
        self.conv4=resnet50.layer3 #1024 x 1/16 x 1/16
        self.conv5=resnet50.layer4 #2048 x 1/32 x 1/32

        self.reduce5=nn.Conv2d(2048,256,kernel_size=1)
        self.reduce4=nn.Conv2d(1024,256,kernel_size=1)
        self.reduce3=nn.Conv2d(512,256,kernel_size=1)
        self.reduce2=nn.Conv2d(256,256,kernel_size=1)

        self.smooth5=nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.smooth4=nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.smooth3=nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.smooth2=nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.getp6=nn.Conv2d(256,256,kernel_size=1,stride=2)

    def forward(self, x):
        '''
        x is the batch of input square imgs
        '''
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.conv2(x)
        c2=x
        x=self.conv3(x)
        c3=x
        x=self.conv4(x)
        c4=x
        x=self.conv5(x)
        c5=x

        m5=self.reduce5(c5)
        m5up=F.interpolate(m5,scale_factor=2.,mode='bilinear')
        c4reduce=self.reduce4(c4)
        m4=c4reduce+m5up
        m4up=F.interpolate(m4,scale_factor=2.,mode='bilinear')
        c3reduce=self.reduce3(c3)
        m3=c3reduce+m4up
        m3up=F.interpolate(m3,scale_factor=2.,mode='bilinear')
        c2reduce=self.reduce2(c2)
        m2=c2reduce+m3up

        p5=self.smooth5(m5)
        p4=self.smooth4(m4)
        p3=self.smooth3(m3)
        p2=self.smooth2(m2)
        p6=self.getp6(p5)
        
        return p6,p5,p4,p3,p2 #size: 1/64,1/32,1/16,1/8,1/4

class RPNHead(nn.Module):
    def __init__(self):
        super(RPNHead, self).__init__()
        '''
        3 anchors per feature level for aspect ratio 1:2,1:1,2:1
        img size on p6,p5,p4,p3,p2 are ih,ih/2,ih/4,ih/8,ih/16 respectively
        '''
        self.rpn_conv=nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(256)
        self.rpn_cls_score=nn.Conv2d(256,(3*2),kernel_size=1) #3 anchors, 2 cls
        self.bnc=nn.BatchNorm2d(3*2)
        self.rpn_bbox_pred=nn.Conv2d(256,(3*4),kernel_size=1) #3 anchors, 4 coord
        self.bnb=nn.BatchNorm2d(3*4)

    def forward(self, p):
        '''
        p - some level in the feature pyramid
        '''
        p=self.rpn_conv(p)
        p=self.bn1(p)
        p=F.relu(p)
        p_cls_score=self.rpn_cls_score(p) # B x (3*2) x h x w
        p_cls_score=self.bnc(p_cls_score) # B x (3*2) x h x w
        p_bbox_pred=self.rpn_bbox_pred(p) # B x (3*4) x h x w
        p_bbox_pred=self.bnb(p_bbox_pred) # B x (3*4) x h x w
        #bbox returns dy, dx, log(h), log(w) wrt anchor
        return p_cls_score,p_bbox_pred 

class AggregateLevels(nn.Module):
    def __init__(self,img_h,img_w,device=None):
        super(AggregateLevels, self).__init__()
        '''
        Given B x (3*2) x h x w class scores per feature level,
        and B x (3*4) x h x w coord predictions per feature level,
        (where h and w are different among feature levels) 
        aggregate proposals from all levels

        return scoress shape: B x N x 2 (N is the total num of proposals across all levels)
               bboxess shape: B x N x 4
               anchorss shape: B x N x 4
        '''
        self.ih=img_h
        self.iw=img_w
        self.device=device

    def apply_deltas(self,anchors,deltas):
        '''
        Modified from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/model.py
        Applies the given deltas to the given anchors
        anchors - B x (3*h*w) x 4 (y_cener,x_center,h,w)
        deltas - B x (3*h*w) x 4 (dy,dx,logh,logw)
 
        return refined boxes B x (3*h*w) x 4 (y1,x1,y2,x2)
               and a set of indices of valid boxes B x v
        '''
        B=anchors.shape[0]
        N=anchors.shape[1]
        center_y = anchors[:,:,0]
        center_x = anchors[:,:,1]
        height = anchors[:,:,2]
        width = anchors[:,:,3]
        # Apply deltas
        center_y = center_y.clone() + (deltas[:,:,0]*height)
        center_x = center_x.clone() + (deltas[:,:,1]*width)
        height = height.clone() * torch.exp(deltas[:,:,2])
        width = width.clone() * torch.exp(deltas[:,:,3])
        # Convert to y1, x1, y2, x2
        '''
        y1 = (center_y-0.5*height).clamp(min=0.0,max=self.ih)
        x1 = (center_x-0.5*width).clamp(min=0.0,max=self.iw)
        y2 = (center_y+0.5*height).clamp(min=0.0,max=self.ih)
        x2 = (center_x+0.5*width).clamp(min=0.0,max=self.iw)
        '''
        y1 = (center_y-0.5*height)
        x1 = (center_x-0.5*width)
        y2 = (center_y+0.5*height)
        x2 = (center_x+0.5*width)
        y1=y1.unsqueeze(2)
        x1=x1.unsqueeze(2)
        y2=y2.unsqueeze(2)
        x2=x2.unsqueeze(2)
        ret=torch.cat([y1,x1,y2,x2],dim=2)
        return ret

    def enlarge_coord(self,x):
        '''
        get a 1d tensor as coords on one img,
        return a 1d tensor representing corresponding coords on a 2x orig img
        0.5 is for continuity correction
        '''
        ret=x.clone()
        ret=(2*ret)+0.5
        return ret 

    def forward(self,css,bps,img_h,img_w):
        '''
        css - (cs2,cs3,cs4,cs5,cs6)
        bps - (bp2,bp3,bp4,bp5,bp6)
        cs - class scores, shape: B x (3*2) x h x w
        bp - bbox predictions, shape: B x (3*4) x h x w
        '''
        bboxess=list() #for all levels
        scoress=list() #for all levels
        anchorss=list() #for all levels
        for i in range(2,7):
            cs=css[i-2]
            bp=bps[i-2]
            B=bp.shape[0]
            h=bp.shape[2]
            w=bp.shape[3]
            cs=cs.clone().view(-1,3,2,h,w)
            bp=bp.clone().view(-1,3,4,h,w)
            cs=cs.clone().permute(0,1,3,4,2)
            cs=cs.clone().view(-1,3*(h*w),2)
            bp=bp.clone().permute(0,1,3,4,2)
            bp=bp.clone().view(-1,3*(h*w),4)
            #get anchors
            coord_y=torch.arange(h).float()
            coord_x=torch.arange(w).float()
            for j in range(i):
                coord_y=self.enlarge_coord(coord_y)
                coord_x=self.enlarge_coord(coord_x)
            ha=float(img_h)/(2**(6-i)) #get anchor box height, 
                                       #e.g. for level 2, ha is 32 for input size 512
            wa=float(img_w)/(2**(6-i)) #get anchor box width
            ha2=ha*2
            wa2=wa*2
            anchors=cs.new(B,3,h,w,4).float()
            anchors[:,0,:,:,2]=ha
            anchors[:,0,:,:,3]=wa2
            anchors[:,1,:,:,2]=ha
            anchors[:,1,:,:,3]=wa
            anchors[:,2,:,:,2]=ha2
            anchors[:,2,:,:,3]=wa
            for j in range(w):
                anchors[:,:,:,j,0]=coord_y
            anchors[:,:,:,:,1]=coord_x
            anchors=anchors.clone().view(B,3*(h*w),4) #finished populating anchors!
            bbs=self.apply_deltas(anchors,bp) #B x (3*h*w) x 4
            anchorss.append(anchors)
            bboxess.append(bbs)
            scoress.append(cs)
        scoress=torch.cat(scoress,dim=1) #B x N x 2 
        bboxess=torch.cat(bboxess,dim=1) #B x N x 4
        anchorss=torch.cat(anchorss,dim=1) #B x N x 4

        return scoress,bboxess,anchorss 


class ROIAlign(nn.Module):
    def __init__(self):
        super(ROIAlign, self).__init__()
        '''
        Given bboxess B x K x 4 sampled pos bboxes or proposed by ProposalFilter (during testing)
        and counts B x 1 indicating k of K valid bboxess
        and feature levels p2 - p6

        return cropped features from the feature levels B x K x 256 x oh x ow
        '''

    def calc_level(self,area,img_h,img_w):
        '''
        Given area, calculate which level this bbox belongs to
        '''
        img_area=area.new(1).float()
        img_area[0]=img_h*img_w
        ret=6+(area.pow(0.5)/img_area.pow(0.5)).log2()
        #use p5 for all p6 level features
        ret=ret.clone().clamp(2,5).round().long()
        return ret


    def forward(self,imgs,bboxess,counts,ps,img_h,img_w,output_h=16,output_w=16):
        '''
        imgs - B x 3 x ih x iw
        p - B x 256 x h x w
        ps - (p2,p3,p4,p5,p6)
        '''
        channel=ps[0].shape[1] #get channel number
        croppeds=list()
        img_batch=torch.split(imgs,1)
        bboxes_batch=torch.split(bboxess,1)
        count_batch=torch.split(counts,1)
        ps_batch=list()
        p2_batch=torch.split(ps[0],1)
        p3_batch=torch.split(ps[1],1)
        p4_batch=torch.split(ps[2],1)
        p5_batch=torch.split(ps[3],1)
        p6_batch=torch.split(ps[4],1)
        for i in range(len(p2_batch)):
            p=(p2_batch[i],p3_batch[i],p4_batch[i],p5_batch[i],p6_batch[i])
            ps_batch.append(p)
        for img,bboxes,count,pp in zip(img_batch,bboxes_batch,count_batch,ps_batch):
            bboxes=bboxes[0]
            count=count[0]
            #an empty tensor to be filled, K x c x oh x ow
            cropped=bboxes.new(bboxess.shape[1],channel+3,output_h,output_w).zero_()
            bboxes=bboxes[:count] # k x 4
            cropped_list=list()
            y1=bboxes[:,0]
            x1=bboxes[:,1]
            y2=bboxes[:,2]
            x2=bboxes[:,3]
            area=(y2-y1)*(x2-x1)
            level=self.calc_level(area,img_h,img_w)
            bbox_batch=torch.split(bboxes,1)
            l_batch=torch.split(level,1)
            for bbox,l in zip(bbox_batch,l_batch):
                bbox=bbox[0]
                l=l[0]
                p=pp[l-2] #corresponding feature level, 1 x 256 x h x w 
                p_large=F.interpolate(p,size=(img_h,img_w),mode='bilinear')
                bbox_int=bbox.round().long()
                y1=bbox_int[0].clamp(0,img_h-1)
                x1=bbox_int[1].clamp(0,img_w-1)
                y2=bbox_int[2].clamp(y1.item()+1,img_h)
                x2=bbox_int[3].clamp(x1.item()+1,img_w)
                p_c=p_large[:,:,y1:y2,x1:x2]
                img_c=img[:,:,y1:y2,x1:x2]
                c=torch.cat([img_c,p_c],dim=1)
                c=F.interpolate(c,size=(output_h,output_w),mode='bilinear') # 1 x 259 x oh x ow
                cropped_list.append(c)
            cropped_valid=torch.cat(cropped_list) # k x 259 x oh x ow
            cropped[:count]=cropped_valid # K x 259 x oh x ow

            cropped=cropped.unsqueeze(0)
            croppeds.append(cropped)
        croppeds=torch.cat(croppeds) # B x K x 259 x oh x ow
        return croppeds

class ClassGenerator(nn.Module):
    def __init__(self):
        super(ClassGenerator, self).__init__()
        '''
        The classification head of mask-RCNN
        Given cropped features B x K x 256 x fh x fw
        and counts B x 1
        return predicted classes B x K x 91
        '''
        self.conv1=nn.Sequential(
                nn.Conv2d(259,256,kernel_size=3,stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU())
        self.conv2=nn.Sequential(
                nn.Conv2d(256,128,kernel_size=3,stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU())
        self.conv3=nn.Sequential(
                nn.Conv2d(128,64,kernel_size=3,stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU())
        self.conv4=nn.Sequential(
                nn.Conv2d(64,64,kernel_size=3,stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU())
        self.fc1=nn.Linear(1024,1024)
        self.fc2=nn.Linear(1024,91)

    def forward(self,featuress,counts,fh=14,fw=14):
        features_batch=torch.split(featuress,1)
        count_batch=torch.split(counts,1)
        classes=list()
        for features,count in zip(features_batch,count_batch):
            features=features[0]
            count=count[0]
            features=features[:count] # k x 259 x fh x fw
            #print(features.shape)
            #features=F.interpolate(features,size=(7,7),mode='bilinear') # k x 259 x 7 x 7
            #print(features.shape)
            features=self.conv1(features)
            features=self.conv2(features)
            features=self.conv3(features)
            features=self.conv4(features)
            features=features.clone().view(count,-1)
            features=self.fc1(features)
            features=self.fc2(features) # k x 91
            cls=features.new(featuress.shape[1],91).zero_()
            cls[:count]=features
            cls=cls.unsqueeze(0)
            classes.append(cls)
            #print(self.fc1.weight)
        classes=torch.cat(classes,dim=0) # B x K x 91
        return classes

class MaskGenerator(nn.Module):
    def __init__(self):
        super(MaskGenerator, self).__init__()
        '''
        The masker head of mask-RCNN
        Given cropped features B x K x 256 x fh x fw
        and counts B x 1
        return predicted masks B x K x 1 x mh x mw
        '''
        self.conv1=nn.Sequential(
                nn.Conv2d(259,256,kernel_size=3,stride=1, padding=1),
                nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1),
                nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1),
                nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU())
        self.conv2=nn.Sequential(
                nn.Conv2d(259,256,kernel_size=3,stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU())
        self.upsample=nn.ConvTranspose2d(256,256,kernel_size=2,stride=2)
        self.conv3=nn.Sequential(
                nn.Conv2d(259,256,kernel_size=3,stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU())
        self.conv4=nn.Sequential(
                nn.Conv2d(256,91,kernel_size=1,stride=1, padding=0))

    def forward(self,featuress,counts,labelss,fh=14,fw=14,mh=28,mw=28):
        features_batch=torch.split(featuress,1)
        count_batch=torch.split(counts,1)
        labels_batch=torch.split(labelss,1)
        maskss=list()
        for features,count,labels in zip(features_batch,count_batch,labels_batch):
            masks=features.new(featuress.shape[1],1,mh,mw).zero_().float() # K x 1 x mh x mw
            features=features[0]
            count=count[0]
            labels=labels[0]
            features=features[:count] # k x 259 x fh x fw
            labels=labels[:count] # k
            imgs = features[:,:3] # k x 3 x fh x fw
            
            '''
            for i in range(imgs.shape[0]):
                cv.display_torch_img(imgs[i])
                '''

            features=self.conv1(features)
            features=torch.cat([imgs,features],dim=1)
            features=self.conv2(features)
            #features=self.upsample(features)
            features=torch.cat([imgs,features],dim=1)
            features=self.conv3(features)
            features=self.conv4(features) # k x 91 x fh x fw

            labels=labels.view(-1,1,1,1).expand(-1,-1,features.shape[2],features.shape[3]) # k x 1 x fh x fw
            features=features.gather(dim=1,index=labels) # k x 1 x fh x fw

            features=torch.sigmoid(features)
            masks_valid=F.interpolate(features,size=(mh,mw),mode='bilinear') # k x 1 x mh x mw
            masks[:count]=masks_valid
            masks=masks.unsqueeze(0)
            maskss.append(masks)
        maskss=torch.cat(maskss) # B x K x 1 x mh x mw
        return maskss

class WholeMask(nn.Module):
    def __init__(self):
        super(WholeMask, self).__init__()
        '''
        Given bboxess B x K x 4 
        and counts B x 1 indicating k of K valid bboxess
        and maskss B x K x 1 x mh x mw

        resize the mask to corresponding bbox, and pad it to ih and iw

        return whole maskss B x K x 1 x ih x iw
        '''

    def forward(self,bboxess,counts,maskss,img_h,img_w):
        bboxes_batch=torch.split(bboxess,1)
        count_batch=torch.split(counts,1)
        masks_batch=torch.split(maskss,1)
        masks_list=list()
        for bboxes,count,masks in zip(bboxes_batch,count_batch,masks_batch):
            bboxes=bboxes[0]
            count=count[0]
            masks=masks[0]
            bboxes=bboxes[:count] # k x 4
            masks=masks[:count] # k x 1 x mh x mw
            bbox_batch=torch.split(bboxes,1)
            mask_batch=torch.split(masks,1)
            masks_whole=list()
            for bbox,mask in zip(bbox_batch,mask_batch):
                bbox=bbox[0]
                bbox_int=bbox.round().long()
                y1=bbox_int[0].clamp(0,img_h-1)
                x1=bbox_int[1].clamp(0,img_w-1)
                y2=bbox_int[2].clamp(y1.item()+1,img_h)
                x2=bbox_int[3].clamp(x1.item()+1,img_w)
                mask_whole=mask.new(1,1,img_h,img_w).zero_().float()
                h=y2-y1
                w=x2-x1
                mask_resized=F.interpolate(mask,size=(h,w)) # 1 x 1 x h x w
                mask_whole[:,:,y1:y2,x1:x2]=mask_resized # 1 x 1 x ih x iw
                masks_whole.append(mask_whole)
            masks_whole=torch.cat(masks_whole) # k x 1 x ih x iw
            masks_whole_pad=masks_whole.new(maskss.shape[1],1,img_h,img_w).zero_().float()
            masks_whole_pad[:count]=masks_whole
            masks_whole_pad=masks_whole_pad.unsqueeze(0)
            masks_list.append(masks_whole_pad)
        maskss_ret=torch.cat(masks_list) # B x K x 1 x ih x iw
        return maskss_ret

class MaskRCNN(nn.Module):
    def __init__(self,r,pos_threshold,img_h,img_w,cf_h,cf_w,mh,mw,device=None):
        super(MaskRCNN, self).__init__()
        self.fpn=FeaturePyramidNet()
        self.rpn=RPNHead()
        self.level_aggregator=AggregateLevels(img_h,img_w,device=device)
        self.roi_align=ROIAlign()
        self.class_generator=ClassGenerator()
        self.mask_generator=MaskGenerator()
        self.whole_mask=WholeMask()

        self.rpn.apply(tl.init_weights)
        self.class_generator.apply(tl.init_weights)
        self.mask_generator.apply(tl.init_weights)

        self.img_h=img_h
        self.img_w=img_w
        self.cf_h=cf_h # height of cropped feature for roi align output
        self.cf_w=cf_w # width of cropped feature for roi align output
        self.mh=mh # resolution of output of mask generator
        self.mw=mw
        self.label_assigner=AssignClsLabel(pos_threshold).to(device)
        self.r=r # number of pos proposals sampled


    def forward(self,x,train_RPN=True,gt_bboxess=None,gt_counts=None,gt_labelss=None):
        '''
        x - a batch of imgs, B x 3 x ih x iw
        '''
        p6,p5,p4,p3,p2=self.fpn(x) #size: 1/64,1/32,1/16,1/8,1/4
        ps=(p2,p3,p4,p5,p6)
        self.ps=ps

        if train_RPN:
            p2cs,p2bp=self.rpn(p2)
            p3cs,p3bp=self.rpn(p3)
            p4cs,p4bp=self.rpn(p4)
            p5cs,p5bp=self.rpn(p5)
            p6cs,p6bp=self.rpn(p6)
            css=(p2cs,p3cs,p4cs,p5cs,p6cs)
            bps=(p2bp,p3bp,p4bp,p5bp,p6bp)
            scoress,bboxess,anchorss=self.level_aggregator(css,bps,self.img_h,self.img_w)
            return scoress,bboxess,anchorss

        else:
            '''
            labels=self.label_assigner(bboxess,gt_bboxess,gt_counts,use_anchor=False) #get labels for proposals
            if labels is not None:
                sample_idxs,sample_counts=sample_proposals(labels,self.r,sample_proposal=True) #sample only pos from proposals
                sample_bboxess=select_bbox(bboxess,sample_idxs,sample_counts)
                '''
            cropped_featuress=self.roi_align(x,gt_bboxess,gt_counts,ps,self.img_h,self.img_w,self.cf_h,self.cf_w)
            classes=self.class_generator(cropped_featuress,gt_counts) # B x K x 91
            if gt_labelss is None:
                _,gt_labelss=classes.max(2) # B x K
            maskss_small=self.mask_generator(cropped_featuress,gt_counts,gt_labelss,fh=self.cf_h,fw=self.cf_w,mh=self.mh,mw=self.mw)
            maskss=self.whole_mask(gt_bboxess,gt_counts,maskss_small,self.img_h,self.img_w)
            return classes,maskss
            '''
            return classes,None
            '''

def select_bbox(bboxess,idxs,counts):
    '''
    bboxess - B x N x 4
    idxs - B x R
    counts - B x 1

    return sampled bboxes - B x R x 4
    '''
    bboxes_batch=torch.split(bboxess,1)
    idx_batch=torch.split(idxs,1)
    count_batch=torch.split(counts,1)
    bboxess_ret=list()
    for bboxes,idx,count in zip(bboxes_batch,idx_batch,count_batch):
        bboxes=bboxes[0]
        idx=idx[0]
        count=count[0]
        idx=idx[:count].clone() # get idx of positive anchors
        bboxes=torch.index_select(bboxes.clone(),0,idx)
        bboxes_pad=bboxes.new(idxs.shape[1],4).zero_() # R x 4
        bboxes_pad[:count]=bboxes
        bboxes_pad=bboxes_pad.unsqueeze(0)
        bboxess_ret.append(bboxes_pad)
    bboxess_ret=torch.cat(bboxess_ret,dim=0)
    return bboxess_ret

class ProposalFilter(nn.Module):
    def __init__(self,k,nms_threshold):
        super(ProposalFilter, self).__init__()
        '''
        Used during testing
        Given B x N x 2 class scores
        and B x N x 4 coord predictions
        filter out top k proposals according to class score

        return ret_bboxess shape: B x K x 4
               counts      shape: B x 1
        '''
        self.k=k
        self.nms_threshold=nms_threshold

    def nms(self,boxess,scoress,overlap=0.5,top_k=200):
        # Original author: Francisco Massa:
        # https://github.com/fmassa/object-detection.torch
        # Ported to PyTorch by Max deGroot (02/01/2017)
        # Modified by Ziwen Chen to support batch operation
        """Apply non-maximum suppression to avoid detecting too many
        overlapping bounding boxes for a given object.
        Args:
            boxess: (tensor) The location preds for the img, B x N x 4
            scoress: (tensor) The class predscores for the img, B x N
            overlap: (float) The overlap thresh for suppressing unnecessary boxes.
            top_k: (int) The Maximum number of box preds to consider.
        Return:
            The indices of the kept boxes with respect to N.
        """
        N=scoress.shape[1]
        top_k=min(top_k,N)

        boxes_batch=boxess.split(1)
        scores_batch=scoress.split(1)
        keep_list=list()
        count_list=list()
        for boxes,scores in zip(boxes_batch,scores_batch):
            boxes=boxes[0] # N x 4
            scores=scores[0] # N
            y1 = boxes[:,0]
            x1 = boxes[:,1]
            y2 = boxes[:,2]
            x2 = boxes[:,3]
            area = (x2-x1)*(y2-y1) # N
            _,idx = scores.sort(0) # sort in ascending order
            keep = idx.new(top_k,).zero_().long() #a new long tensor of size top_k
            count = idx.new(1).zero_().long() #a new long tensor of size 1
            while count<top_k and idx.shape[0]>0:
                i = idx[-1]  #index of current largest val
                idx=idx[:-1]  #remove kept element
                keep[count]=i
                count += 1
                #get the remaining bboxes
                rem_y1=torch.index_select(y1,0,idx)
                rem_x1=torch.index_select(x1,0,idx)
                rem_y2=torch.index_select(y2,0,idx)
                rem_x2=torch.index_select(x2,0,idx)
                #clip all bboxes to the current bbox
                rem_y1=torch.clamp(rem_y1, min=y1[i].item(),max=y2[i].item())
                rem_x1=torch.clamp(rem_x1, min=x1[i].item(),max=x2[i].item())
                rem_y2=torch.clamp(rem_y2, min=y1[i].item(),max=y2[i].item())
                rem_x2=torch.clamp(rem_x2, min=x1[i].item(),max=x2[i].item())
                h=rem_y2-rem_y1
                w=rem_x2-rem_x1
                inter=w*h
                #IoU = inter / (area(a) + area(b) - inter)
                rem_areas=torch.index_select(area,0,idx) #remaining areas
                union=(rem_areas+area[i])-inter
                IoU=inter/union
                #keep only elements with an IoU <= overlap
                idx=idx[IoU.le(overlap)]
            count=count.unsqueeze(0)
            keep=keep.unsqueeze(0)
            count_list.append(count)
            keep_list.append(keep)
        counts=torch.cat(count_list,dim=0) # B x 1
        keeps=torch.cat(keep_list,dim=0) # B x top_k
        return keeps, counts

    def forward(self,scoress,bboxess):
        scoress=F.softmax(scoress.clone(),dim=2)
        scoress=scoress[:,:,0] #only get the score for positive object cls
        #apply non-maximum suppression
        keeps,counts=self.nms(bboxess,scoress,overlap=self.nms_threshold,top_k=self.k)
        keep_batch=torch.split(keeps,1)
        count_batch=torch.split(counts,1)
        bboxes_batch=torch.split(bboxess,1)
        ret_bboxess=list()
        for keep,count,bboxes in zip(keep_batch,count_batch,bboxes_batch):
            keep=keep[0]
            count=count[0]
            bboxes=bboxes[0]
            keep=keep[:count] #get rid of zero-padding at the end
            rem_bboxes=torch.index_select(bboxes,0,keep) #select bboxes according to keep indices
            ret_bboxes=rem_bboxes.new(self.k,4).zero_()
            ret_bboxes[:count]=rem_bboxes
            ret_bboxes=ret_bboxes.unsqueeze(0)
            ret_bboxess.append(ret_bboxes)
        ret_bboxess=torch.cat(ret_bboxess) #B x top_k x 4
        return ret_bboxess,counts

class AssignClsLabel(nn.Module):
    def __init__(self,pos_threshold=0.5):
        super(AssignClsLabel, self).__init__()
        '''
        Given anchorss B x N x 4, (y_cener,x_center,h,w)
        and gt bboxess B x A x 4 (ground truth),
        and gt count B x 1 (indicate how many a of A are valid gt)
        Assign each anchor a cls label of either pos or other (1 or 0)
        
        return labels B x N x 1
        '''
        self.pos_threshold=pos_threshold

    def forward(self,anchorss,gt_bboxess,gt_counts,use_anchor=True):
        labels=list()
        anchors_batch=torch.split(anchorss,1)
        gt_bboxes_batch=torch.split(gt_bboxess,1)
        gt_count_batch=torch.split(gt_counts,1)
        for anchors,gt_bboxes,gt_count in zip(anchors_batch,gt_bboxes_batch,gt_count_batch):
            anchors=anchors[0]
            gt_bboxes=gt_bboxes[0]
            gt_count=gt_count[0]
            gt_bboxes=gt_bboxes[:gt_count] # a x 4
            
            label=anchors.new(anchors.shape[0],1).zero_() # N x 1
            if use_anchor:
                y=anchors[:,0]
                x=anchors[:,1]
                h=anchors[:,2]
                w=anchors[:,3]
                y1=y-h/2
                y2=y1+h
                x1=x-w/2
                x2=x1+w
                area=h*w
            else:
                bboxes=anchors
                y1=bboxes[:,0]
                x1=bboxes[:,1]
                y2=bboxes[:,2]
                x2=bboxes[:,3]
                y=(y1+y2)/2.
                x=(x1+x2)/2.
                h=y2-y1
                w=x2-x1
                area=h*w
            gt_bbox_batch=torch.split(gt_bboxes,1)
            for gt_bbox in gt_bbox_batch:
                gt_bbox=gt_bbox[0]
                gy1=gt_bbox[0]
                gx1=gt_bbox[1]
                gy2=gt_bbox[2]
                gx2=gt_bbox[3]
                ga=(gy2-gy1)*(gx2-gx1)
                yy1=y1.clamp(min=gy1.item(),max=gy2.item())
                xx1=x1.clamp(min=gx1.item(),max=gx2.item())
                yy2=y2.clamp(min=gy1.item(),max=gy2.item())
                xx2=x2.clamp(min=gx1.item(),max=gx2.item())
                inter=(yy2-yy1)*(xx2-xx1)
                union=(area+ga)-inter
                iou=inter/union
                cls_score=iou.ge(self.pos_threshold).float()
                cls_score=cls_score.clone().view(-1,1)
                label=label.clone()+cls_score
            label=label.clone().ge(1.).long()
            if not use_anchor:
                pos_idx=label.view(-1).nonzero()
                pos_count=pos_idx.shape[0]
                if pos_count == 0:
                    return None
            label=label.unsqueeze(0)
            labels.append(label)
        labels=torch.cat(labels) # B x N x 1, long tensor
        return labels

def sample_proposals(labels,S,sample_proposal=False):
    '''
    Given class labels, B x N x 1
    Sample S/4 positive anchors, 3*S/4 neg anchors 
    If sample_proposal, sample S positive proposals

    return the idxs B x S where first quarter (or less) idx is pos and the rest is neg
    and pos counts B x 1
    '''
    idxs=list()
    pos_counts=list()
    label_batch=torch.split(labels,1)
    for label in label_batch:
        label=label[0]
        label=label.clone().view(-1)
        idx_pos=label.nonzero()
        idx_neg=(label==0).nonzero()
        pos_num=idx_pos.shape[0]
        neg_num=idx_neg.shape[0]
        pos_count=idx_pos.new(1).long()
        if sample_proposal:
            p=min(pos_num,S)
            n=0
        else:
            p=min(pos_num,S//4)
            n=min(neg_num,S-p)
        if p == 0:
            return None,None
        pos_count[0]=p
        idx=labels.new(S).zero_().long()
        idxx_pos=torch.randperm(idx_pos.shape[0])
        idxx_pos=idxx_pos[:p].clone()
        idxx_pos_=idx_pos.new(idxx_pos.shape).long()
        idxx_pos_[:]=idxx_pos
        idxx_pos=idxx_pos_.clone()
        idx_pos_selected=torch.index_select(idx_pos,0,idxx_pos)
        idx_pos_selected=idx_pos_selected.clone().view(-1)
        idx[:p]=idx_pos_selected
        if not sample_proposal:
            idxx_neg=torch.randperm(idx_neg.shape[0])
            idxx_neg=idxx_neg[:n].clone()
            idxx_neg_=idx_neg.new(idxx_neg.shape).long()
            idxx_neg_[:]=idxx_neg
            idxx_neg=idxx_neg_.clone()
            idx_neg_selected=torch.index_select(idx_neg,0,idxx_neg)
            idx_neg_selected=idx_neg_selected.clone().view(-1)
            idx[p:p+n]=idx_neg_selected
        idx=idx.unsqueeze(0)
        pos_count=pos_count.unsqueeze(0)
        idxs.append(idx)
        pos_counts.append(pos_count)
    idxs=torch.cat(idxs)
    pos_counts=torch.cat(pos_counts)
    return idxs,pos_counts

'''
methods for calculating losses
'''
            
def calc_cls_score_loss(scoress,idxs,counts):
    '''
    Given scoress (not softmaxed) B x N x 2
    sample idxs B x S (first p for pos, rest for neg)
    pos counts B x 1

    Return the l_cls log loss, a scalar
    '''
    scores_batch=torch.split(scoress,1)
    idx_batch=torch.split(idxs,1)
    count_batch=torch.split(counts,1)
    loss=scoress.new(1).zero_().float()
    for scores,idx,count in zip(scores_batch,idx_batch,count_batch):
        scores=scores[0]
        idx=idx[0]
        count=count[0]
        idx_pos=idx[:count]
        idx_neg=idx[count:]
        scores=F.log_softmax(scores,dim=1)
        scores_pos=torch.index_select(scores,0,idx_pos)
        scores_neg=torch.index_select(scores,0,idx_neg)
        scores_pos=scores_pos[:,0].clone()
        scores_neg=scores_neg[:,1].clone()
        scores_pos_mean=scores_pos.mean()
        scores_neg_mean=scores_neg.mean()
        loss=loss-(scores_pos_mean+scores_neg_mean)
    return loss

def calc_bbox_loss(bboxess,anchorss,gt_bboxess,gt_counts,idxs,counts):
    '''
    Given bboxess B x N x 4, (y1,x1,y2,x2)
          anchorss B x N x 4, (y_cener,x_center,h,w)
          gt_boxess B x N x 4, (y1,x1,y2,x2)
          gt_counts B x 1
          sample idxs B x S
          sample counts B x 1

    Return the l_bbox smooth l1 loss, a scalar
    '''
    bboxes_batch=torch.split(bboxess,1)
    anchors_batch=torch.split(anchorss,1)
    gt_bboxes_batch=torch.split(gt_bboxess,1)
    gt_count_batch=torch.split(gt_counts,1)
    idx_batch=torch.split(idxs,1)
    count_batch=torch.split(counts,1)
    l_bbox=bboxess.new(1).zero_().float()
    for bboxes,anchors,gt_bboxes,gt_count,idx,count in zip(bboxes_batch,anchors_batch,gt_bboxes_batch,gt_count_batch,idx_batch,count_batch):
        bboxes=bboxes[0]
        anchors=anchors[0]
        gt_bboxes=gt_bboxes[0]
        gt_count=gt_count[0]
        idx=idx[0]
        count=count[0]
        idx=idx[:count].clone() # get idx of positive anchors
        bboxes=torch.index_select(bboxes.clone(),0,idx)
        anchors=torch.index_select(anchors.clone(),0,idx)
        gt_bboxes=gt_bboxes[:gt_count].clone() # a x 4
        gt_y1=gt_bboxes[:,0]
        gt_x1=gt_bboxes[:,1]
        gt_y2=gt_bboxes[:,2]
        gt_x2=gt_bboxes[:,3]
        gt_area=(gt_y2-gt_y1)*(gt_x2-gt_x1)
        bbox_batch=torch.split(bboxes,1)
        anchor_batch=torch.split(anchors,1)
        tis=list()
        gtis=list()
        for bbox,anchor in zip(bbox_batch,anchor_batch):
            #find the gt bbox that has the largest iou with the current anchor 
            bbox=bbox[0]
            anchor=anchor[0]
            ay=anchor[0]
            ax=anchor[1]
            ah=anchor[2]
            aw=anchor[3]
            ay1=ay-ah/2
            ay2=ay1+ah
            ax1=ax-aw/2
            ax2=ax1+aw
            area=ah*aw
            gt_yy1=gt_y1.clamp(min=ay1.item(),max=ay2.item())
            gt_xx1=gt_x1.clamp(min=ax1.item(),max=ax2.item())
            gt_yy2=gt_y2.clamp(min=ay1.item(),max=ay2.item())
            gt_xx2=gt_x2.clamp(min=ax1.item(),max=ax2.item())
            inter=(gt_yy2-gt_yy1)*(gt_xx2-gt_xx1)
            union=(gt_area+area)-inter
            iou=inter/union
            _,max_idx=iou.max(dim=0)
            #calculate l_bbox
            gt_bbox=gt_bboxes[max_idx]
            gy1=gt_bbox[0]
            gx1=gt_bbox[1]
            gy2=gt_bbox[2]
            gx2=gt_bbox[3]
            gy=(gy1+gy2)/2.
            gx=(gx1+gx2)/2.
            gh=gy2-gy1
            gw=gx2-gx1
            y1=bbox[0]
            x1=bbox[1]
            y2=bbox[2]
            x2=bbox[3]
            y=(y1+y2)/2.
            x=(x1+x2)/2.
            h=y2-y1
            w=x2-x1
            ti=y1.new(4).zero_().float()
            gti=y1.new(4).zero_().float()
            ti[0]=(y-ay)/ah
            ti[1]=(x-ax)/aw
            ti[2]=h.log()-ah.log()
            ti[3]=w.log()-aw.log()
            gti[0]=(gy-ay)/ah
            gti[1]=(gx-ax)/aw
            gti[2]=gh.log()-ah.log()
            gti[3]=gw.log()-aw.log()
            ti=ti.unsqueeze(0)
            gti=gti.unsqueeze(0)
            tis.append(ti)
            gtis.append(gti)
        tis=torch.cat(tis) # s x 4
        gtis=torch.cat(gtis) # s x 4
        lb=F.smooth_l1_loss(tis,gtis)
        l_bbox+=(lb/count.float())
    return l_bbox

def calc_class_loss(classes,gt_labelss,gt_counts):
    '''
    classes - B x K x 91
    gt_labelss - B x K
    gt counts B x 1

    Return the l_class cross entropy loss, a scalar
    '''
    classes_batch=torch.split(classes,1)
    gt_labels_batch=torch.split(gt_labelss,1)
    gt_count_batch=torch.split(gt_counts,1)

    l_class=classes.new(1).zero_().float()
    correct=0
    total=0

    for cls,gt_labels,gt_count in zip(classes_batch,gt_labels_batch,gt_count_batch):
        cls=cls[0]
        gt_labels=gt_labels[0]
        gt_count=gt_count[0]

        cls=cls[:gt_count] # k x 91
        gt_labels=gt_labels[:gt_count] # k

        lc=F.cross_entropy(cls,gt_labels)
        l_class+=lc

        _, predicted = cls.max(1)
        correct+=predicted.eq(gt_labels).sum().item()
        total+=gt_labels.size(0)
    return l_class*100,correct/total

def calc_mask_loss(maskss,gt_maskss,gt_counts,visualize=False):
    '''
    whole maskss B x R x 1 x ih x iw
    ground-truth maskss B x A x 1 x ih x iw
    gt counts B x 1

    Return the l_mask binary cross entropy loss, a scalar
    '''
    masks_batch=torch.split(maskss,1)
    gt_masks_batch=torch.split(gt_maskss,1)
    gt_count_batch=torch.split(gt_counts,1)

    l_mask=gt_maskss.new(1).zero_().float()

    for masks,gt_masks,gt_count in zip(masks_batch,gt_masks_batch,gt_count_batch):
        masks=masks[0]
        gt_masks=gt_masks[0]
        gt_count=gt_count[0]

        masks=masks[:gt_count].clone() # a x 1 x ih x iw
        gt_masks=gt_masks[:gt_count].clone() # a x 1 x ih x iw

        lm=F.binary_cross_entropy(masks,gt_masks,reduction='mean')
        l_mask+=(lm/gt_count.float())
        if visualize:
            for i in range(gt_count.item()):
                mask_visualizer(None,masks[i][0],None,gt_masks[i][0])
    return l_mask*1000

def mask_visualizer(bbox,mask,gt_bbox,gt_mask):
    #bbox=bbox.cpu().detach()
    mask=mask.cpu().detach()
    #gt_bbox=gt_bbox.cpu().detach()
    gt_mask=gt_mask.cpu().detach()
    #bbox=bbox.round().long()
    #gt_bbox=gt_bbox.round().long()
    mask=torch.where(mask>0.5,torch.Tensor([1]),torch.Tensor([0]))
    gt_mask=torch.where(gt_mask>0.5,torch.Tensor([1]),torch.Tensor([0]))
    mask=mask.long()
    gt_mask=gt_mask.long()
    mask=mask*100
    gt_mask=gt_mask*200
    #bbox=bbox.numpy()
    #gt_bbox=gt_bbox.numpy()
    mask=mask.numpy()
    gt_mask=gt_mask.numpy()
    img=np.dstack((mask,mask,mask))
    img+=np.dstack((gt_mask,gt_mask,gt_mask))
    img=img.clip(0,255)
    img=img.astype(np.uint8)
    #img=cv.bbox_img(img,bbox[0],bbox[1],bbox[2],bbox[3],(255,0,0))
    #img=cv.bbox_img(img,gt_bbox[0],gt_bbox[1],gt_bbox[2],gt_bbox[3],(0,255,0))
    cv.display_img(img)
