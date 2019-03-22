import cv2
import numpy as np
import random

def add_alpha(img):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2RGBA)
    return img

def flip_img_b(img):
    img = cv2.flip(img, -1)
    return img

def resize_img(img,x,y):
    img = cv2.resize(img,(x,y))
    return img

def flip_img_v(img):
    img = cv2.flip(img, 0)
    return img

def flip_img_h(img):
    img = cv2.flip(img, 1)
    return img

def load_img(fname):
    img = cv2.imread(fname,cv2.IMREAD_UNCHANGED)
    return img

def save_img(fname,img):
    cv2.imwrite(fname,img)

def display_img(img):
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def display_img_file(fname):
    img = cv2.imread(fname)
    cv2.imshow(fname,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def generate_random_vertices(size=128,num_min=3,num_max=6,dis_max_ratio=0.5):
    '''
    generates an array of points for a polygon to be drawn on a square
    mask of size size.
    number of points restricted by num_min and num_max
    dis_max_ratio indicates the maximum distance between the coordinates of 
    two consecutive points w.r.t size
    '''
    dis_max = round(dis_max_ratio*size)
    num = random.randint(0,num_max-num_min+1)+num_min
    pts = list()
    x=random.randint(0,size+1)
    y=random.randint(0,size+1)
    pts.append([x,y])
    for i in range(num):
        x_off=random.randint(0,dis_max*2+1)-dis_max
        x+=x_off
        y_off=random.randint(0,dis_max*2+1)-dis_max
        y+=y_off
        pts.append([x,y])
    pts = np.asarray(pts, dtype=np.int32)
    return pts

def generate_polygon_mask(size=128):
    '''
    generates a random polygon mask on canvas of size size
    0 for non-mask area and 1 for mask
    '''
    mask = np.zeros((size,size,1),np.float32)
    pts = generate_random_vertices(size)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(mask, [pts], 1)
    return mask
