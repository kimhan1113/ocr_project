import cv2 
import numpy as np
import matplotlib.pyplot as plt

def resize_and_pad(image_name):

    eps = 1e-14
    img = cv2.imread(image_name)
    vid_h,vid_w,_= img.shape

    if vid_h > vid_w :
        new_h = 512
        new_w = int(vid_w*(new_h)/(vid_h+eps))
    else:
        new_w = 512
        new_h = int(vid_h*(new_w)/(vid_w+eps))

    img_resize_ = cv2.resize(img,(new_w,new_h))

    img_resize = np.zeros((512,512,3))
    
    if vid_h > vid_w :
        padding =int( (512 - new_w)/2)
        img_resize[:,padding : padding + new_w,:] = img_resize_
        padding = -1*padding
    else:
        padding = int((512 - new_h)/2 )
        img_resize[padding:padding+new_h,:,:] = img_resize_


    return np.array(img_resize).astype(np.uint8),padding,img.shape

def transform_box(box,padding,shape):
    eps = 0.00000001
    x1,y1,x2,y2 = np.array(box).astype(np.float32)
    padding = padding / 512.0

    w  = shape[1]
    h  = shape[0]

    
    if padding < 0 :
        
        # h > w --> padded in w direction
        x1 = x1 - np.abs(padding)
        x2 = x2 - np.abs(padding)
        w = h
    else :
        # w > h --> padded in h direction

        y1 = y1 - np.abs(padding)
        y2 = y2 - np.abs(padding)
        h = w

    x1 = x1 * w
    x2 = x2 * w
    y1 = y1 * h
    y2 = y2 * h
    
    return (x1,y1,x2,y2)

def to_array(bbox,padding,im_shape):
    bbox_final = []
    for box_ in bbox:
        box = transform_box(box_,padding,im_shape)
        bbox_final.append(box)

    return bbox_final
