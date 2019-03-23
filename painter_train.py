'''
Training code for painter GAN
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
import utils.dataloader as dl

import painter_ops as po

bg_training_flist_file='/home/chenziwe/sceneslicer/SceneSlicer/dataset/MITindoor/Images/training.txt'
bg_validation_flist_file='/home/chenziwe/sceneslicer/SceneSlicer/dataset/MITindoor/Images/validation.txt'
bg_testing_flist_file='/home/chenziwe/sceneslicer/SceneSlicer/dataset/MITindoor/Images/testing.txt'
bg_gen_ckpt_path='/home/chenziwe/sceneslicer/SceneSlicer/logs/bg_gen256.ckpt'
bg_dis_ckpt_path='/home/chenziwe/sceneslicer/SceneSlicer/logs/bg_dis256.ckpt'
ob_training_flist_file=''
ob_validation_flist_file=''
ob_testing_flist_file=''
ob_gen_ckpt_path=''
ob_dis_ckpt_path=''

bg_in_channels=3
ob_in_channels=4
gan_iteration=5
batch_size=16
img_size=128
epoch=5
lr=1e-4
l1_alpha=1
patch_alpha=1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ImageDataSet(Dataset):
    '''
    Images dataset
    (MITindoor and Rendered 3D Models from ShapeNet)
    with free-form mask
    '''
    def __init__(self, flist_file, alpha):
        self.flist=dl.load_flist_file(flist_file)
        self.alpha=alpha

    def __len__(self):
        return len(self.flist)

    def __getitem__(self,idx):
        img = cv.load_img(self.flist[idx])
        if img is None:
            sys.exit("Error! Img is None at: "+self.flist[idx])
        img = dl.process_img(img,crop_size=img_size,resize=False,sample_num=1,alpha=self.alpha,normalize=True,pytorch=True,random_mask=True,ones_boundary=False)
        img=img[0]
        img=torch.from_numpy(img)
        return img

def train_bg_painter():
    #load dataset
    dataset=ImageDataSet(bg_training_flist_file,alpha=False)
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=8)
    gennet=po.PainterNet(bg_in_channels,device).to(device)
    gennet.train()
    disnet=po.SNPatchGAN(bg_in_channels,device).to(device)
    disnet.train()
    gen_optimizer = torch.optim.Adam(gennet.parameters(),lr=lr,betas=(0.5,0.9))
    dis_optimizer = torch.optim.Adam(disnet.parameters(),lr=lr,betas=(0.5,0.9))

    #load checkpoint
    if os.path.isfile(bg_gen_ckpt_path):
        gennet.load_state_dict(torch.load(bg_gen_ckpt_path))
        print("Loaded bg gen ckpt!")
    if os.path.isfile(bg_dis_ckpt_path):
        disnet.load_state_dict(torch.load(bg_dis_ckpt_path))
        print("Loaded bg dis ckpt!")

    for e in range(epoch):
        step=0
        for i,img_batch in enumerate(dataloader):
            actual_batch_size=img_batch.shape[0]
            img_batch=img_batch.to(device)
            imgs=img_batch[:,:bg_in_channels]
            masks=img_batch[:,bg_in_channels:]
            incomplete_imgs=imgs*(masks.eq(0.).float())

            #get predictions from generator
            x_coarse,x=gennet(incomplete_imgs,masks)
            predictions=x*masks+incomplete_imgs

            #get score from discriminator
            pos_neg_in=torch.cat([imgs,predictions],dim=0)
            pos_neg_score=disnet(pos_neg_in)
            pos_score=pos_neg_score[:actual_batch_size]
            neg_score=pos_neg_score[actual_batch_size:]

            #calculate losses
            scale_factor=pos_score.shape[2]/float(pos_neg_in.shape[2])
            mask_s=F.interpolate(masks, scale_factor=scale_factor, mode='bilinear')
            if step%(gan_iteration+1)!=gan_iteration:
                d_loss_pos=F.relu(torch.ones_like(pos_score)-pos_score)
                d_loss_neg=F.relu(torch.ones_like(neg_score)+neg_score)
                d_loss=(d_loss_pos*mask_s).mean()+(d_loss_neg*mask_s).mean()
                dis_optimizer.zero_grad()
                d_loss.backward()
                dis_optimizer.step()
            else:
                g_loss=-(neg_score*mask_s).mean()
                l1_loss=(predictions-imgs).abs().mean()
                loss=l1_loss*l1_alpha+g_loss*patch_alpha
                gen_optimizer.zero_grad()
                loss.backward()
                gen_optimizer.step()

                torch.save(gennet.state_dict(), bg_gen_ckpt_path)
                torch.save(disnet.state_dict(), bg_dis_ckpt_path)
                print('Epoch [{}/{}] , Step {}, Loss: {:.4f}'
                        .format(e+1, epoch, step, loss.item()))

            step+=1

train_bg_painter()
