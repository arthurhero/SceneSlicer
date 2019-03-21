import numpy as np
import time
from multiprocessing import Pool
import sys
import cv2
import subprocess
import random
import os
import opencv_utils as cv

def under_prob(prob):
    x=random.randint(0,9999)
    return x<prob*10000

def img_process(img,crop_size=256,alpha=True,pytorch=True):
    '''
    resize img
    crop the img to crop_size
    normalize img
    randomly flip img
    add alpha channel if requested
    transpose img for pytorch
    '''
    height, width, channel= img.shape
    if height > width:
        img=cv.resize_img(img,crop_size,int(round(height/width*crop_size)))
    else:
        img=cv.resize_img(img,int(round(width/height*crop_size)),crop_size)

    height, width, _= img.shape
    x_start=random.randint(0,width-crop_size)
    y_start=random.randint(0,height-crop_size)
    img = img[x_start:x_start+crop_size, y_start:y_start+crop_size]
    if under_prob(0.5):
        img = cv.flip_img_h(img)
    img = img.astype(np.float32)
    img = img/128
    img = img-1
    if alpha and channel == 3:
        alpha_channel = np.ones((crop_size,crop_size,1), dtype=np.float32)
        np.append(img,alpha_channel)
    if pytorch:
        img = img.transpose((2, 0, 1))
    return img

def create_file_list(root_path):
    '''
    create a list of file paths for all imgs files inside root_path
    '''
    flist = list()
    queue = list() 
    queue.append((root_path,False))
    cur_path = "" 
    while len(queue) != 0:
        last_node = queue[-1]
        last = last_node[0]
        expanded = last_node[1]
        if last[-4:]==".jpg" or last[-4:]==".png":
            fpath=cur_path+last
            flist.append(fpath)
            del queue[-1]
        else:
            if not expanded:
                cur_path+=last
                if cur_path[-1]!="/":
                    cur_path+="/"
                queue[-1]=(last,True)
                cmd = "ls "+cur_path
                process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()
                files=output.split()
                for f in files:
                    queue.append((f,False))
            else:
                cur_path=cur_path[:-(len(last))]
                if len(cur_path)!=0:
                    cur_path=cur_path[:-1]
                del queue[-1]
    return flist

def load_all_from_disk(flist,data,img_size=256):
    '''
    flist - a list of file path
    data - an empty list to be appended
    img_size - height and width of square img data
    '''
    #start=time.time()
    for f in flist:
        img = cv.load_img(f)
        img = img_process(img,img_size)
        data.append(img)
    #end=time.time()
    #print "Used "+str(end-start)+" secs to load all imgs!"
