import time
from threading import Thread,Lock
import subprocess
import random
import numpy as np
import opencv_utils as cv

def under_prob(prob):
    x=random.randint(0,9999)
    return x<prob*10000

def process_img(img,crop_size=256,resize=False,sample_num=1,
        alpha=True,pytorch=True,random_mask=False,ones_boundary=False):
    '''
    resize img so the shorter edge equal to crop_size if requested or if image too small
    sample sample_num crops from the img with crop_size
    randomly flip img horizontally
    add alpha channel if requested
    add randomly mask channel if requested
    normalize img to [-1,1]
    transpose img for pytorch if requested
    put an all-ones channel between img and mask as boundary if requested
    '''
    batch = list()
    height, width, channel=img.shape
    if resize or min(height,width)<crop_size:
        if height > width:
            img=cv.resize_img(img,crop_size,int(round(height/width*crop_size)))
        else:
            img=cv.resize_img(img,int(round(width/height*crop_size)),crop_size)
        height, width, _= img.shape
    original_img = img
    for i in range(sample_num):
        x_start=random.randint(0,width-crop_size)
        y_start=random.randint(0,height-crop_size)
        img = original_img[x_start:x_start+crop_size, y_start:y_start+crop_size]
        if under_prob(0.5):
            img = cv.flip_img_h(img)
        #cv.display_img(img)
        img = img.astype(np.float32)
        if alpha and channel == 3:
            alpha_channel = np.ones((crop_size,crop_size,1), dtype=np.float32)
            img = np.dstack((img,alpha_channel))
        if random_mask:
            mask = cv.generate_polygon_mask(crop_size)
            if ones_boundary:
                boundary = np.ones((crop_size,crop_size,1), dtype=np.float32)
                img = np.dstack((img,boundary))
            img = np.dstack((img, mask))
        #normalize img to [-1,1]
        img = img/128
        img = img-1
        if pytorch:
            #put the channel axis to the front
            img = img.transpose((2, 0, 1))
        batch.append(img)
    return batch

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

def save_flist_file(fname,flist,append=False):
    f = None
    if append:
        f = open(fname, "a")
    else:
        f = open(fname, "w")
    for l in flist:
        f.write(l+'\n')
    f.close()

def load_all_from_disk(flist,data,img_size=256,resize=False,sample_num=1,alpha=True,pytorch=True,random_mask=False,ones_boundary=False,multi=False,lock=None):
    '''
    flist - a list of file path
    data - an empty list to be appended
    img_size - height and width of square img data
    the rest refer to the documentation of process_img
    '''
    for f in flist:
        img = cv.load_img(f)
        imgs = process_img(img,img_size,resize,sample_num,alpha,pytorch,random_mask,ones_boundary) 
        if multi:
            lock.acquire()
            data.extend(imgs)
            lock.release()
        else:
            data.extend(imgs)

def multi_load_all_from_disk(flist,data,worker_num=1,img_size=256,resize=False,sample_num=1,alpha=True,pytorch=True,random_mask=False,ones_boundary=False):
    '''
    multiprocess version
    '''
    #start=time.time()
    lock = Lock()
    ts=list()
    interval = len(flist)//worker_num
    for i in range(worker_num):
        start = i*interval
        end = start+interval
        if i==worker_num-1:
            end = len(flist)
        args = (flist[start:end],data,img_size,resize,sample_num,alpha,pytorch,random_mask,ones_boundary,True,lock)
        t=Thread(target=load_all_from_disk,args=args)
        t.start()
        ts.append(t)
    for i in range(worker_num):
        ts[i].join()
    #end=time.time()
    #print("Used "+str(end-start)+" secs to load all imgs!")

'''
TEST CODE
flist=create_file_list("/Users/arthurhero/Desktop/Research/sceneslicer/dataset/ShapeNetRendered")
#save_flist_file('test.txt',flist,append=False)
data = list()
multi_load_all_from_disk(flist,data,worker_num=5,sample_num=2,alpha=True,pytorch=True,random_mask=True,ones_boundary=True)
'''
