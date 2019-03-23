'''
Layers and operations for background painter (GAN) and object painter (GAN)
including spectral normalization, gated convolution, contextual attention, etc.
'''

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter 

import utils.opencv_utils as cv
import utils.tools as tl

def l2normalize(w, eps=1e-12):
    # normalize w where w is a pytorch tensor
    return w / (w.norm() + eps)

def init_weights(m):
    # Initialize parameters
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class SpectralNorm(nn.Module):
    '''
    Original Idea see paper by Miyato et. al.
    (https://arxiv.org/abs/1802.05957)
    Code provided by christiancosgrove
    (https://github.com/christiancosgrove/pytorch-spectral-normalization-gan)
    Documented and modified by me
    '''
    # module refers to the conv layer whose weights will be normalized here
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self.module.apply(init_weights)
        # if haven't made u,v before, initialize them
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            #calculate w^Tu
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            #calculate wv
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))
        #calculate u^Twv
        sigma = u.dot(w.view(height,-1).mv(v))
        #set normalized weight for module
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        #get the weight from the conv layer (module)
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        #flatten weight matrix except the batch axis
        width = w.view(height, -1).data.shape[1]

        #initialize random vectors from isotropic distribution
        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        #delete the original weight
        del self.module._parameters[self.name]

        #store the vectors into the module as parameters 
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        #update weight first and then feed forward the conv layer
        self._update_u_v()
        return self.module.forward(*args)


class ContextualAttention(nn.Module):
    '''
    Original idea and code in TensorFlow by Yu et. al.
    (https://github.com/JiahuiYu/generative_inpainting/blob/master/inpaint_ops.py)
    Transferred and modified by me
    '''
    def __init__(self,patch_size=3,rate=1,fuse_kernel_size=3,softmax_scale=10.,fuse=True,device=None):
        super(ContextualAttention, self).__init__()
        self.ps=patch_size
        self.r=rate
        self.fs=fuse_kernel_size
        self.ss=softmax_scale
        self.fuse=fuse
        self.device=device

    def forward(self, f_o, b_o, mask_o=None):
        '''
        f - foreground feature map (B x c x h x w)
        b - background feature map (B x c x h x w)
        mask - indicating patches not available (B x 1 x h x w)
        Assume all have the same height and width
        '''
        ps=self.ps
        r=self.r
        fs=self.fs
        ss=self.ss
        fuse=self.fuse

        pad=tl.calc_padding(ps,1)
        fuse_pad=tl.calc_padding(fs,1)

        B=f_o.shape[0]
        c=f_o.shape[1]
        h=f_o.shape[2]
        w=f_o.shape[3]

        #get background patches
        #note that hw/rr is the number of patches from one image
        bg_patches = F.unfold(b_o,kernel_size=ps,stride=r,padding=pad) # B x (c x ps x ps) x (hw/rr)
        bg_patches = bg_patches.clone().view(B,c,ps,ps,-1) # B x c x ps x ps x (hw/rr)
        bg_patches = bg_patches.clone().permute(0,4,1,2,3) # B x (hw/rr) x c x ps x ps

        #shrink f and b and mask by rate
        f = F.interpolate(f_o, scale_factor=1./r, mode='nearest') # B x c x h/r x w/r
        b = F.interpolate(b_o, scale_factor=1./r, mode='nearest') # B x c x h/r x w/r
        mask=mask_o
        if mask_o is not None:
            mask = F.interpolate(mask_o, scale_factor=1./r, mode='nearest') # B x 1 x h/r x w/r

        #shrinked size by rate
        hr=f.shape[2]
        wr=f.shape[3]

        #get shrinked background patches (to be matched with foreground)
        bg_patches_shrinked = F.unfold(b,kernel_size=ps,padding=pad) # B x (c x ps x ps) x (hw/rr)
        bg_patches_shrinked = bg_patches_shrinked.clone().view(B,c,ps,ps,-1) # B x c x ps x ps x (hw/rr)
        bg_patches_shrinked = bg_patches_shrinked.clone().permute(0,4,1,2,3) # B x (hw/rr) x c x ps x ps

        #get patches from mask
        if mask is None:
            mask = torch.zeros(B,1,hr,wr)
            if self.device is not None:
                mask=mask.to(self.device)
        mask_patches = F.unfold(mask,kernel_size=ps,padding=pad) # B x (1 x ps x ps) x (hw/rr)
        mask_patches = mask_patches.clone().view(B,1,ps,ps,-1) # B x 1 x ps x ps x (hw/rr)
        mask_patches = mask_patches.clone().permute(0,1,4,2,3) # B x 1 x (hw/rr) x ps x ps
        mask_patches = mask_patches.clone().mean(3,True) # B x 1 x (hw/rr) x 1 x ps
        mask_patches = mask_patches.clone().mean(4,True) # B x 1 x (hw/rr) x 1 x 1 
        mask_patches = mask_patches.clone().eq(0.).float() # invert the mask

        #create identity matrices for fusion
        fuse_w = torch.eye(fs).view(1,1,fs,fs) # 1 x 1 x fs x fs
        if self.device is not None:
            fuse_w=fuse_w.to(self.device)

        results = list()

        #split f, bg patches, shrinked bg patches along the batch axis
        f_batch = f.split(1)
        b_patch_batch = bg_patches.split(1)
        b_patch_shrinked_batch = bg_patches_shrinked.split(1)
        mask_patch_batch = mask_patches.split(1)

        for fi,bi,bsi,mi in zip(f_batch,b_patch_batch,b_patch_shrinked_batch,mask_patch_batch):
            fi=fi.clone().view(1,c,hr,wr) # 1 x c x h/r x w/r
            bi=bi.clone().view(hr*wr,c,ps,ps)
            bsi=bsi.clone().view(hr*wr,c,ps,ps)
            mi=mi.clone().view(1,hr*wr,1,1)

            bnorm=bsi.pow(2).sum(dim=(1,2,3)).pow(0.5)
            eps=torch.FloatTensor((1e-4,)).expand_as(bnorm)
            if self.device is not None:
                eps=eps.to(self.device)
            bnorm=bnorm.max(eps)
            bnorm=bnorm.view(-1,1,1,1)
            bsi=bsi.clone()/bnorm # (hw/rr) x c x ps x ps
            score=F.conv2d(fi,bsi,stride=1,padding=pad) # 1 x (hw/rr) x h/r x w/r

            '''
            Please refer to Yu et. al.'s paper for explanation
            (https://arxiv.org/abs/1801.07892)
            '''
            if fuse:
                score=score.clone().view(1,1,hr*wr,hr*wr)
                score=F.conv2d(score,fuse_w,stride=1,padding=fuse_pad) # 1 x 1 x (hw/rr) x (hw/rr)
                score=score.clone().view(1,hr,wr,hr,wr)
                score=score.clone().permute(0,2,1,4,3)
                score=score.clone().contiguous().view(1,1,hr*wr,hr*wr)
                score=F.conv2d(score,fuse_w,stride=1,padding=fuse_pad)
                score=score.clone().view(1,wr,hr,wr,hr)
                score=score.clone().permute(0,2,1,4,3)
                score=score.clone().contiguous().view(1,hr*wr,hr,wr)

            #apply mask to zero-out invalid score
            score = score.clone()*mi
            #apply softmax to get actual score (scale is for making score sharper)
            score = F.softmax(score.clone()*ss, dim=1)
            score = score.clone()*mi

            #final step! Copy and paste from bg patches according to the score
            result = F.conv_transpose2d(score,bi,stride=r,padding=pad,output_padding=1)/4. # 1 x c x h x w
            results.append(result)

        out = torch.cat(results)
        return out

def test_contextual_attention(fname1,fname2):
    '''
    Testing the CA layer using two 256x256 RGB images
    Result should be the first image with style transferred from the second image
    '''
    img = cv.load_img(fname1)
    img2 = cv.load_img(fname2)
    img=img[:256,:256]
    img2=img2[:256,:256]
    cv.display_img(img)
    cv.display_img(img2)
    img = img.astype(np.float32)
    img2 = img2.astype(np.float32)
    img = img.transpose(2,0,1)
    img2 = img2.transpose(2,0,1)
    img = torch.FloatTensor(img)
    img2 = torch.FloatTensor(img2)
    img=img.view(1,256,256,3)
    img2=img2.view(1,256,256,3)
    ca_layer = ContextualAttention(rate=2,fuse=True)
    img = ca_layer(img,img2)
    img = img.numpy()[0]
    imp = np.clip(img,0,255)
    img = img.transpose(1,2,0)
    img = img.astype(np.uint8)
    cv.display_img(img)

class GatedConv2d(nn.Module):
    '''
    Original idea by Yu et. al.
    (https://arxiv.org/abs/1806.03589)
    '''
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True):
        super(GatedConv2d, self).__init__()
        self.conv_layer=nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias),
                nn.LeakyReLU()
                )
        self.gate_layer=nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias),
                nn.Sigmoid()
                )
        self.conv_layer.apply(init_weights)
        self.gate_layer.apply(init_weights)

    def forward(self, x):
        x1=self.conv_layer(x)
        x2=self.gate_layer(x)
        out = x1*x2
        return out

class PainterNet(nn.Module):
    def __init__(self,in_channels,device=None):
        super(PainterNet, self).__init__()

        #stage 1 (coarse)
        pad=tl.calc_padding(5,1)
        # +2 is for the ones_boundary and mask
        self.conv1=GatedConv2d(in_channels+2,32,5,stride=1,padding=pad)
        pad=tl.calc_padding(3,1)
        self.conv2=GatedConv2d(32,64,3,stride=2,padding=pad)
        self.conv3=GatedConv2d(64,64,3,stride=1,padding=pad)
        self.conv4=GatedConv2d(64,128,3,stride=2,padding=pad)
        self.conv5=GatedConv2d(128,128,3,stride=1,padding=pad)
        self.conv6=GatedConv2d(128,128,3,stride=1,padding=pad)
        pad=tl.calc_padding(3,2)
        self.conv7=GatedConv2d(128,128,3,stride=1,padding=pad,dilation=2)
        pad=tl.calc_padding(3,4)
        self.conv8=GatedConv2d(128,128,3,stride=1,padding=pad,dilation=4)
        pad=tl.calc_padding(3,8)
        self.conv9=GatedConv2d(128,128,3,stride=1,padding=pad,dilation=8)
        pad=tl.calc_padding(3,16)
        self.conv10=GatedConv2d(128,128,3,stride=1,padding=pad,dilation=16)
        pad=tl.calc_padding(3,1)
        self.conv11=GatedConv2d(128,128,3,stride=1,padding=pad)
        self.conv12=GatedConv2d(128,128,3,stride=1,padding=pad)
        self.conv13=GatedConv2d(128,64,3,stride=1,padding=pad)
        self.conv14=GatedConv2d(64,64,3,stride=1,padding=pad)
        self.conv15=GatedConv2d(64,32,3,stride=1,padding=pad)
        self.conv16=GatedConv2d(32,16,3,stride=1,padding=pad)
        self.conv17=nn.Conv2d(16,in_channels,3,stride=1,padding=pad)
        self.conv17.apply(init_weights)

        #stage 2 (fine)
        #conv branch
        pad=tl.calc_padding(5,1)
        self.xconv1=GatedConv2d(in_channels+2,32,5,stride=1,padding=pad)
        pad=tl.calc_padding(3,1)
        self.xconv2=GatedConv2d(32,32,3,stride=2,padding=pad)
        self.xconv3=GatedConv2d(32,64,3,stride=1,padding=pad)
        self.xconv4=GatedConv2d(64,64,3,stride=2,padding=pad)
        self.xconv5=GatedConv2d(64,128,3,stride=1,padding=pad)
        self.xconv6=GatedConv2d(128,128,3,stride=1,padding=pad)
        pad=tl.calc_padding(3,2)
        self.xconv7=GatedConv2d(128,128,3,stride=1,padding=pad,dilation=2)
        pad=tl.calc_padding(3,4)
        self.xconv8=GatedConv2d(128,128,3,stride=1,padding=pad,dilation=4)
        pad=tl.calc_padding(3,8)
        self.xconv9=GatedConv2d(128,128,3,stride=1,padding=pad,dilation=8)
        pad=tl.calc_padding(3,16)
        self.xconv10=GatedConv2d(128,128,3,stride=1,padding=pad,dilation=16)
        #attention branch
        pad=tl.calc_padding(5,1)
        self.pmconv1=GatedConv2d(in_channels+2,32,5,stride=1,padding=pad)
        pad=tl.calc_padding(3,1)
        self.pmconv2=GatedConv2d(32,32,3,stride=2,padding=pad)
        self.pmconv3=GatedConv2d(32,64,3,stride=1,padding=pad)
        self.pmconv4=GatedConv2d(64,128,3,stride=2,padding=pad)
        self.pmconv5=GatedConv2d(128,128,3,stride=1,padding=pad)
        self.pmconv6=GatedConv2d(128,128,3,stride=1,padding=pad)
        if device is None:
            self.ca=ContextualAttention(patch_size=3,rate=2,fuse_kernel_size=3,softmax_scale=10.,fuse=True)
        else:
            self.ca=ContextualAttention(patch_size=3,rate=2,fuse_kernel_size=3,softmax_scale=10.,fuse=True,device=device).to(device)
        self.pmconv7=GatedConv2d(128,128,3,stride=1,padding=pad)
        self.pmconv8=GatedConv2d(128,128,3,stride=1,padding=pad)

        #final stage
        self.fconv1=GatedConv2d(256,128,3,stride=1,padding=pad)
        self.fconv2=GatedConv2d(128,128,3,stride=1,padding=pad)
        self.fconv3=GatedConv2d(128,64,3,stride=1,padding=pad)
        self.fconv4=GatedConv2d(64,64,3,stride=1,padding=pad)
        self.fconv5=GatedConv2d(64,32,3,stride=1,padding=pad)
        self.fconv6=GatedConv2d(32,16,3,stride=1,padding=pad)
        self.fconv7=nn.Conv2d(16,in_channels,3,stride=1,padding=pad)
        self.fconv7.apply(init_weights)

    def forward(self, x, mask):
        '''
        x - B x c x h x w
        mask - B x 1 x h x w
        '''
        xin = x
        ones_boundary = torch.ones_like(mask)
        x=torch.cat([x,ones_boundary,mask],dim=1)
        size=x.shape[2]
        mask_s = F.interpolate(mask, scale_factor=0.25, mode='nearest') 
        #stage 1
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.conv6(x)
        x=self.conv7(x)
        x=self.conv8(x)
        x=self.conv9(x)
        if size>=256:
            x=self.conv10(x)
        x=self.conv11(x)
        x=self.conv12(x)
        x=F.interpolate(x, scale_factor=2, mode='nearest')
        x=self.conv13(x)
        x=self.conv14(x)
        x=F.interpolate(x, scale_factor=2, mode='nearest')
        x=self.conv15(x)
        x=self.conv16(x)
        x=self.conv17(x)
        x=x.clamp(-1.,1.)
        x_coarse=x*mask+xin*(mask.eq(0.).float())

        #stage 2
        x=x_coarse
        x_cur=torch.cat([x,ones_boundary,mask],dim=1)
        #conv branch
        x=self.xconv1(x_cur)
        x=self.xconv2(x)
        x=self.xconv3(x)
        x=self.xconv4(x)
        x=self.xconv5(x)
        x=self.xconv6(x)
        x=self.xconv7(x)
        x=self.xconv8(x)
        x=self.xconv9(x)
        if size>=256:
            x=self.xconv10(x)
        x_conv=x
        #attention branch
        x=self.pmconv1(x_cur)
        x=self.pmconv2(x)
        x=self.pmconv3(x)
        x=self.pmconv4(x)
        x=self.pmconv5(x)
        x=self.pmconv6(x)
        f=x
        b=x
        x=self.ca(f,b,mask_s)
        x=self.pmconv7(x)
        x=self.pmconv8(x)
        x_attention=x

        #last stage
        x=torch.cat([x_conv,x_attention],dim=1)
        x=self.fconv1(x)
        x=self.fconv2(x)
        x=F.interpolate(x, scale_factor=2, mode='nearest')
        x=self.fconv3(x)
        x=self.fconv4(x)
        x=F.interpolate(x, scale_factor=2, mode='nearest')
        x=self.fconv5(x)
        x=self.fconv6(x)
        x=self.fconv7(x)
        x=x.clamp(-1.,1.)

        return x_coarse,x


class SNPatchGAN(nn.Module):
    def __init__(self,in_channels,device=None):
        super(SNPatchGAN, self).__init__()
        pad=tl.calc_padding(5,1)
        self.conv1=SpectralNorm(nn.Conv2d(in_channels,64,5,stride=1,padding=pad))
        self.conv2=SpectralNorm(nn.Conv2d(64,128,5,stride=2,padding=pad))
        self.conv3=SpectralNorm(nn.Conv2d(128,256,5,stride=2,padding=pad))
        self.conv4=SpectralNorm(nn.Conv2d(256,256,5,stride=2,padding=pad))
        self.conv5=SpectralNorm(nn.Conv2d(256,256,5,stride=2,padding=pad))
        self.conv6=SpectralNorm(nn.Conv2d(256,256,5,stride=2,padding=pad))

    def forward(self, x):
        size=x.shape[2]

        x=self.conv1(x)
        x=F.leaky_relu(x)
        x=self.conv2(x)
        x=F.leaky_relu(x)
        x=self.conv3(x)
        x=F.leaky_relu(x)
        x=self.conv4(x)
        x=F.leaky_relu(x)
        x=self.conv5(x)
        x=F.leaky_relu(x)
        if size>=256:
            x=self.conv6(x)
            x=F.leaky_relu(x)
        return x