import numpy as np
import os
import glob
import sys
import time
# import fastss

this_dir = os.path.dirname(__file__)
sys.path.insert(0, this_dir + '/..')

from tbpp_model import TBPP512_dense
from tbpp_utils import PriorUtil
from lib.annotate import Annotater

from crnn_model import CRNN
from crnn_utils import alphabet87 as alphabet


if __name__ == '__main__':

    weights_path = './checkpoints/tbpp/202007031335_ky_tbpp/weights.030.h5'
    weights_path_crnn = './checkpoints/crnn/202006231539_crnn_lstm_ky/weights.030.h5'
    data_path = './data/ky_test'
    dataset_name = 'AGM_GOOD'

    confidence_threshold = 0.25
    image_names = np.sort(glob.glob(os.path.join(data_path,dataset_name,'*','*.bmp')))
    print(len(image_names))
    annotation_path = os.path.join(data_path,dataset_name,'Annotations')

    if not os.path.exists(annotation_path):
        os.makedirs(annotation_path)

    # TextBoxes++ + DenseNet
    tbpp = TBPP512_dense(softmax=False)
    # model = TBPP512(softmax=False)
    prior_util = PriorUtil(tbpp)
    checkdir = os.path.dirname(weights_path) 

    input_width = 256
    input_height = 32
    crnn = CRNN((input_width,input_height,1),len(alphabet),prediction_only=True,gru=False)

    print("started loading model weights")
    tic = time.time()
    tbpp.load_weights(weights_path)
    crnn.load_weights(weights_path_crnn)
    print(time.time() - tic)
    print("finished loading model weights")

    anno_tool = Annotater(tbpp,crnn,prior_util,image_names,annotation_path)
    anno_tool.annotate()



