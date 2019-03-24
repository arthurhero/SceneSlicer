'''
convenient tools for conv layers
'''

import numpy as np
import torch

def calc_padding(ker_size,dilate_rate):
    '''
    calculate how much padding is needed for 'SAME' padding
    assume square square kernel
    assume odd kernel size
    '''
    ker_size=(ker_size-1)*(dilate_rate-1)+ker_size
    margin=(ker_size-1)//2
    return margin

def recover_img(tensor):
    '''
    Recover an RGB(A) from a pytorch tensor
    '''
    img=tensor.cpu().detach()
    img=(img+1)*128
    img=img.clamp(0,255)
    img=img.permute(1,2,0)
    img=img.numpy()
    img=img.astype(np.uint8)
    return img
