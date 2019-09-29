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
#import utils.dataloader as dl

import painter_ops as po
from masker_train import COCODataSet

bg_training_flist_file='dataset/MITindoor/Images/training.txt'
bg_validation_flist_file='dataset/MITindoor/Images/validation.txt'
bg_testing_flist_file='dataset/MITindoor/Images/testing.txt'
bg_gen_ckpt_path='logs/bg_gen.ckpt'
bg_gen_coarse_ckpt_path='logs/bg_coarse_gen.ckpt'
bg_dis_ckpt_path='logs/bg_dis.ckpt'
ob_training_flist_file=''
ob_validation_flist_file=''
ob_testing_flist_file=''
ob_gen_ckpt_path=''
ob_gen_coarse_ckpt_path=''
ob_dis_ckpt_path=''

bg_in_channels=3
ob_in_channels=4
gan_iteration=5
batch_size=2
img_size=512
epoch=100
lr=0.000001
l1_alpha=1
l1_coarse_alpha=0.01
fm_alpha=0
patch_alpha=0.005

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

'''
class ImageDataSet(Dataset):
    Images dataset
    (MITindoor and Rendered 3D Models from ShapeNet)
    with free-form mask
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
    '''
    
def apply_masks(imgs,maskss,counts):
    '''
    imgs - B x 3 x ih x iw
    maskss - B x A x 1 x ih x iw
    counts - B x 1
    '''
    masks=list()
    ret_imgs=list()
    incomplete_imgs=list()
    img_batch=imgs.split(1)
    mask_batch=maskss.split(1)
    count_batch=counts.split(1)
    for img,mask,count in zip(img_batch,mask_batch,count_batch):
        mask=mask[0] # A x 1 x ih x iw
        count=count[0] # 1
        #print("count:",count)
        count=min(4,count)
        mask=mask[:count] # a x 1 x ih x iw
        img=img.expand(count,-1,-1,-1) # a x 3 x ih x iw
        incomplete_img=img*(mask.eq(0.).float())
        masks.append(mask)
        ret_imgs.append(img)
        incomplete_imgs.append(incomplete_img)
    ret_imgs=torch.cat(ret_imgs,dim=0)
    masks=torch.cat(masks,dim=0)
    incomplete_imgs=torch.cat(incomplete_imgs,dim=0)
    '''
    print(ret_imgs.shape)
    print(masks.shape)
    print(incomplete_imgs.shape)
    '''
    return ret_imgs,masks,incomplete_imgs

def train_painter(max_ratio=1,pretrain=False,fix_coarse=False,ob=False):
    '''
    pretrain - whether this is training the coarse part of generator or not
    ob - whether this is training an object painter instead of a bg painter
    '''
    in_channels=3
    gen_ckpt_path=bg_gen_ckpt_path
    gen_coarse_ckpt_path=bg_gen_coarse_ckpt_path
    dis_ckpt_path=bg_dis_ckpt_path
    training_flist_file=bg_training_flist_file
    if ob:
        in_channels=4
        gen_ckpt_path=ob_gen_ckpt_path
        gen_coarse_ckpt_path=ob_gen_coarse_ckpt_path
        dis_ckpt_path=ob_dis_ckpt_path
        training_flist_file=ob_training_flist_file
    #load dataset
    '''
    alpha=False
    if ob:
        alpha=True
    dataset=ImageDataSet(training_flist_file,alpha=alpha)
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=8)
    '''
    #use coco for now
    dataset=COCODataSet(max_ratio=max_ratio,random_mask=True)
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=batch_size)

    gennet=po.PainterNet(in_channels,pretrain,fix_coarse,device).to(device)
    gennet.train()
    disnet=po.SNPatchGAN(in_channels,device).to(device)
    disnet.train()
    gen_optimizer = torch.optim.Adam(gennet.parameters(),lr=lr,betas=(0.5,0.9))
    dis_optimizer = torch.optim.Adam(disnet.parameters(),lr=lr,betas=(0.5,0.9))

    #load checkpoint
    if pretrain:
        if os.path.isfile(gen_coarse_ckpt_path):
            gennet.load_state_dict(torch.load(gen_coarse_ckpt_path))
            print("Loaded coarse gen ckpt!")
    else:
        if os.path.isfile(gen_ckpt_path):
            gennet.load_state_dict(torch.load(gen_ckpt_path))
            print("Loaded gen ckpt!")
        else:
            if os.path.isfile(gen_coarse_ckpt_path):
                gennet.load_state_dict(torch.load(gen_coarse_ckpt_path))
                print("Loaded coarse gen ckpt! Training fine from coarse now!")
    if os.path.isfile(dis_ckpt_path):
        disnet.load_state_dict(torch.load(dis_ckpt_path))
        print("Loaded dis ckpt!")

    d_loss_sum=0.0
    g_loss_sum=0.0

    for e in range(epoch):
        step=0
        g_step=0
        d_step=0
        for i,img_batch in enumerate(dataloader):
            train_g=False
            if pretrain or step%(gan_iteration+1)==gan_iteration:
                #if step%(gan_iteration+1)==gan_iteration:
                train_g=True
                disnet.apply(tl.freeze_params)
                gennet.apply(tl.unfreeze_params)
            else:
                disnet.apply(tl.unfreeze_params)
                gennet.apply(tl.freeze_params)

            actual_batch_size=img_batch.shape[0]
            img_batch=img_batch.to(device)
            imgs=img_batch[:,:in_channels]
            masks=img_batch[:,in_channels:]
            incomplete_imgs=imgs*(masks.eq(0.).float())
            '''
            #load data from coco
            imgs=img_batch['img'] # B x 3 x ih x iw
            maskss=img_batch['masks'] # B x A x 1 x ih x iw
            counts=img_batch['num_ann'] # B x 1
            imgs=imgs.to(device)
            maskss=maskss.to(device)
            counts=counts.to(device)
            imgs,masks,incomplete_imgs=apply_masks(imgs,maskss,counts)
            actual_batch_size=imgs.shape[0]
            '''

            #get predictions from generator
            predictions=None
            x_coarse=None
            if pretrain:
                x_coarse=gennet(incomplete_imgs,masks)
                predictions=x_coarse
            else:
                x_coarse,x=gennet(incomplete_imgs,masks)
                predictions=x*masks+incomplete_imgs

            #get score from discriminator
            pos_neg_in=torch.cat([imgs,predictions],dim=0)
            pos_neg_score,pos_neg_feature=disnet(pos_neg_in,masks)
            pos_score=pos_neg_score[:actual_batch_size]
            neg_score=pos_neg_score[actual_batch_size:]
            pos_feature=pos_neg_feature[:actual_batch_size]
            neg_feature=pos_neg_feature[actual_batch_size:]

            #calculate losses
            if not train_g:
                d_loss_pos=F.relu(torch.ones_like(pos_score)-pos_score)
                d_loss_neg=F.relu(torch.ones_like(neg_score)+neg_score)
                d_loss=(d_loss_pos).mean()+(d_loss_neg).mean()
                dis_optimizer.zero_grad()
                d_loss.backward()
                dis_optimizer.step()
                d_loss_sum+=d_loss
                if d_step%500==499:
                    print('Epoch [{}/{}] , Step {}, D_Loss: {:.4f}'
                            .format(e+1, epoch, d_step, d_loss_sum/500.0))
                    d_loss_sum=0.0
                    torch.save(disnet.state_dict(), dis_ckpt_path)
                d_step+=1
            else:
                l1_loss=(predictions-imgs).abs().mean()
                feature_match_loss=(pos_feature-neg_feature).abs().mean()
                loss=l1_loss*l1_alpha+feature_match_loss*fm_alpha
                if not fix_coarse:
                    l1_coarse_loss=(x_coarse-imgs).abs().mean()
                    loss+=l1_coarse_loss*l1_coarse_alpha
                if not pretrain:
                    g_loss=-(neg_score).mean()
                    loss+=(g_loss*patch_alpha)
                gen_optimizer.zero_grad()
                loss.backward()
                #print(gennet.conv1.conv_layer[0].weight.grad)
                gen_optimizer.step()
                g_loss_sum+=loss
                if g_step%100==99:
                    print('Epoch [{}/{}] , Step {}, G_Loss: {:.4f}'
                            .format(e+1, epoch, g_step, g_loss_sum/100.0))
                    g_loss_sum=0.0
                    if pretrain:
                        torch.save(gennet.state_dict(), gen_coarse_ckpt_path)
                    else:
                        torch.save(gennet.state_dict(), gen_ckpt_path)
                g_step+=1

            '''
            sample_orig=tl.recover_img(imgs[0])
            sample_incomplete=tl.recover_img(incomplete_imgs[0])
            sample_coarse=tl.recover_img(x_coarse[0])
            sample_predicted=tl.recover_img(predictions[0])
            cv.display_img(sample_orig)
            cv.display_img(sample_incomplete)
            cv.display_img(sample_coarse)
            cv.display_img(sample_predicted)
            if step/gan_iteration>1:
                break
            '''
            step+=1

def test_painter(img,mask):
    '''
    img, an img with holes, 3 x ih x iw
    mask, 1 x ih x iw
    '''
    gennet=po.PainterNet(img.shape[0],False,False,device).to(device)
    gennet.eval()
    if os.path.isfile(bg_gen_ckpt_path):
        gennet.load_state_dict(torch.load(bg_gen_ckpt_path))
        print("Loaded gen ckpt!")
    imgs=img.unsqueeze(0)
    masks=mask.unsqueeze(0)
    _,x=gennet(imgs,masks)
    predictions=x*masks+imgs # 1 x 3 x ih x iw
    return predictions[0]

if __name__=='__main__':
    train_painter(max_ratio=0.20,pretrain=False,fix_coarse=False,ob=False)
