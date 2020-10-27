import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from ..utils.vis import plot_box
from ..crnn.crnn_data import crop_words
from ..utils.bboxes import rbox3_to_polygon 
from ..datasets.ssd_data import preprocess
from ..crnn.crnn_model import CRNN
from ..crnn.crnn_utils import alphabet87 as alphabet
from ..crnn.crnn_utils import decode
from ..tbpp.tbpp_model import TBPP512, TBPP512_dense
from ..tbpp.tbpp_utils import PriorUtil
from ..ssd.ssd_utils import non_maximum_suppression

MEAN = np.array([104,117,123])

def resize_and_pad(image_name,cfgs=None):

    eps = cfgs.EPS
    
    img = cv2.imread(image_name)
    vid_h,vid_w,_= img.shape

    if vid_h > vid_w :
        new_h = 512
        new_w = int(vid_w*(new_h)/(vid_h+eps))
    else:
        new_w = 512
        new_h = int(vid_h*(new_w)/(vid_w+eps))

    img = cv2.resize(img,(new_w,new_h))

    img_resize = 255*np.zeros((512,512,3))
    
    if vid_h > vid_w :
        padding =int( (512 - new_w)/2)
        img_resize[:,padding : padding + new_w,:] = img
    else:
        padding = int((512 - new_h)/2 )
        img_resize[padding:padding+new_h,:,:] = img


    return np.array(img_resize[:,:,(2,1,0)]).astype(np.uint8)

def is_inside(detection_poly,body_poly):
    center = np.sum(detection_poly,axis=0)/4.0
    #print(center)
    xmin,xmax = np.amin(body_poly[:,0]),np.amax(body_poly[:,0])
    ymin,ymax = np.amin(body_poly[:,1]),np.amax(body_poly[:,1])

    x_in = xmin < center[0] and xmax > center[0]
    y_in = ymin < center[1] and ymax > center[1]
    
    #print(x_in,y_in)
    
    return x_in and y_in

def filter(quad, body) : 
    '''
    filter out detections whose center lies outside of body 

    # Arguments 
        detection_results: List of corresponding detection polygonal with 
            shape (objects, 4 x xy)
        body : (objects, 4 x xy)
    detection
    '''
    quad_polys_ = [np.reshape(quad[j,:], (-1, 2)) for j in range(len(quad))]
    body_polys = [np.reshape(body[0], (4, 2))]
    
    quad_polys = []
    idxs = []
    for i in range(len(quad_polys_)): # samples

        if is_inside(quad_polys_[i],body_polys[0]) : 
            quad_polys.append(quad_polys_[i])
            idxs.append(i)

    quad = [np.reshape(quad_polys[i],(-1)) for i in range(len(quad_polys))]
    return quad,idxs

def quad2bbox(quads):
    bboxs = []
    for quad in quads : 
        l,t,r,_,_,b,_,_ = quad
        box = np.array([l,t,r,b],dtype=np.float32)
        bboxs.append(box)
    return np.array(bboxs)


class eval_util():
    def __init__(self,gtu,
                 confidence_threshold,
                 args,
                 tbpp_model_path=None,
                 input_size=(512,512),
                 with_crnn=False,
                 crnn_model_path=None):

        self.confidence_threshold = confidence_threshold
        self.mean = MEAN
        self.gtu = gtu
        self.num_images = len(self.gtu.image_names)
        self.index = 0 
        self.height = 512
        self.width = 512
        self.nms_thresh = 0.4
        self.nms_top_k = 400
        self.overlap_thresh = 1.0

        # attributes related to tbpp
        self.input_size = input_size
        self.tbpp_graph = tf.Graph()
        self.tbpp_model_path =  tbpp_model_path
        
        # attributes related to crnn
        self.with_crnn = with_crnn
        self.crnn_model_path = crnn_model_path

        
        # define tbpp graph
        with self.tbpp_graph.as_default():
            self.tbpp_sess= tf.Session()
            with self.tbpp_sess.as_default():
                self.tbpp_model = TBPP512_dense((512,512,3))
                self.prior_util = PriorUtil(self.tbpp_model)
                self.tbpp_model.load_weights(self.tbpp_model_path, by_name=True)

        # define crnn graph
        if self.with_crnn and self.crnn_model_path != None : 
            self.crnn_graph = tf.Graph()
            with self.crnn_graph.as_default():
                self.crnn_session = tf.Session()
                with self.crnn_session.as_default():
                    self.crnn_model = CRNN((256,32, 1), len(alphabet), prediction_only=True, gru=False,training=0,bidirectional=True)
                    self.crnn_model.load_weights(crnn_model_path, by_name=True)

    def sample_image(self,sample_body=True,num_rots=0):
        inputs,data,body = self.gtu.sample_batch(1,self.index,sample_body=True,preserve_aspect_ratio=True)
        zero_padded_image,data,body = self.gtu.sample_batch(1,self.index,sample_body=True,preserve_aspect_ratio=True,zero=True)

        # rotate image
        inputs = np.rot90(np.squeeze(inputs),k=num_rots)
        zero_padded_image = np.rot90(np.squeeze(zero_padded_image),k=num_rots)
        
        body = np.array(body)
        body = [np.squeeze(body)[:-1]]

        if num_rots != 0 :
            body = self.rotate_quad([np.squeeze(body)],num_rots=num_rots)

        return inputs[None,:,:,:],zero_padded_image,body


    def get_coord(self,quad):
        return min(quad[::2]),max(quad[::2]),min(quad[1::2]),max(quad[1::2])
    def get_quad(self,xmin,xmax,ymin,ymax):
        return [xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax]

    def rotate_quad(self,quads,num_rots=0):
        num_rots = num_rots % 4

        if num_rots == 0 : 
            return quads

        quads_rotated = []
        for quad in quads : 
            xmin,xmax,ymin,ymax = self.get_coord(quad)

            if num_rots == 3: 
                xmin,xmax,ymin,ymax = 1-ymax,1-ymin,xmin,xmax

            elif num_rots == 2 : 
                xmin,xmax,ymin,ymax = 1-xmax,1-xmin,1-ymax,1-ymin

            elif num_rots == 1 :
                xmin,xmax,ymin,ymax = ymin,ymax,1-xmax,1-xmin


            quad = self.get_quad(xmin,xmax,ymin,ymax)
            quads_rotated.append(quad)

        return quads_rotated

    def filter_by_overlap(self,bbox):
        print(bbox)
        eps = 1e-15
        bbox = np.multiply(bbox,512).astype(np.int)
        map = np.zeros((512,512))

        for box in bbox : 
            xmin,ymin,xmax,ymax = box 
            map[ymin:ymax,xmin:xmax] += 1 
        
        map = np.clip(map,0,2)
        bbox_filtered = []

        for box in bbox : 
            xmin,ymin,xmax,ymax = box 
            area = (xmax-xmin) * (ymax - ymin)

            patch = map[ymin:ymax,xmin:xmax]
            patch_sum = np.sum(patch)  - area

            if (patch_sum / (area+eps)) < self.overlap_thresh : 
                print(patch_sum / (area+eps))
                bbox_filtered.append(box)

        return bbox_filtered

    def evaluate(self):

        for i in range(self.num_images):
            quads = [] 
            detected = False

            for ang in range(4):
                inputs,zero_padded_image,body = self.sample_image(num_rots=ang)

                if ang == 0 :
                    plt.figure(figsize=[8]*2)
                    raw_image = np.array(np.squeeze(zero_padded_image)+self.mean[None,None,:])
                    plt.imshow(raw_image.astype(np.uint8))


                with self.tbpp_graph.as_default():
                    with self.tbpp_sess.as_default():
                        preds = self.tbpp_model.predict(inputs,batch_size=1)
                res = self.prior_util.decode(preds[0], self.confidence_threshold, fast_nms=False)

                if len(res) == 0 :
                    continue

                quad,conf= res[:,4:12],res[:,(-2)]

                if len(conf.shape) == 2 : 
                    conf = np.squeeze(conf)


                quad_filtered,idxs = filter(quad,body)
                quad_rotated = self.rotate_quad(quad_filtered,num_rots=4-ang)
                conf_filtered = conf[idxs]

                quads += quad_rotated

                if detected : 
                    confs = np.concatenate((confs.copy(),conf_filtered.copy()))
                else : 
                    confs = conf_filtered
                    detected = True



            bbox = quad2bbox(quads)
            try : 
                assert bbox.shape[0] == confs.shape[0] 
            except : 
                print(confs.shape)
                assert 1==0

            idx = non_maximum_suppression(
                    bbox,confs, 
                    self.nms_thresh, self.nms_top_k)

            bbox = self.filter_by_overlap(bbox[idx])

            for box in bbox:
                plot_box(box, box_format='xyxy', color='r')

            plt.show()
            plt.close()
            self.index = np.random.randint(0,self.num_images)
            

#
#            
#            
#            if self.with_crnn :
#                input_width = 256
#                input_height = 32
#
#                words = crop_words(raw_image,np.array(bbox), input_height, width=input_width, grayscale=True)
#                words = np.asarray([w.transpose(1,0,2) for w in words])
#                
#                if len(words) > 0:
#                    with crnn_graph.as_default():
#                        with crnn_session.as_default():
#                            res_crnn = crnn_model.predict(words)
#
#                for i in range(len(words)):
#                    chars = [alphabet[c] for c in np.argmax(res_crnn[i], axis=1)]
#                    res_str = decode(chars)
#                    #cv2.imwrite('croped_word_%03i.png' % (i), words[i])
#                    x,y= np.multiply(bbox[i][:2],512).astype(np.int)
#                    plt.text(x,y, res_str,bbox=dict(facecolor='white', alpha=0.5))
#
#            #ax = plt.gca()
#            
#
#            #plt.show()
#            #plt.close()
#            self.index += 1




