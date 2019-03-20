import cv2
import numpy as np

def display_img(fname):
    img = cv2.imread(fname)
    cv2.imshow(fname,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def generate_polygon_mask(size=128):
    mask = np.zeros((1,size,size,1),np.float32)

