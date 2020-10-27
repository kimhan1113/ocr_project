import numpy as np
import matplotlib.pyplot as plt
import os
import editdistance
import pickle
import time
import keras
import time
import argparse
import sys
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint


this_dir = os.path.dirname(__file__)
sys.path.insert(0, this_dir + '/..')

# sys.path.append('C:\\Users\\jsson\\OneDrive\\바탕 화면\\coding\\ocr_project\\data\\models\\crnn_lstm\\')


from config.config import cfg,get_output_dir
from lib.crnn.crnn_model import CRNN
from lib.crnn.crnn_data import InputGenerator
from lib.crnn.crnn_utils import decode
from lib.utils.training import Logger, ModelSnapshot
from lib.datasets.factory import get_dataset
from lib.crnn.crnn_utils import alphabet87 as alphabet


# ### Data

# In[ ]:


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='evaluate tbpp model')
    parser.add_argument('--weights', dest='model_path',
                        help='initialize with pretrained model weights',
                        default='crnn_lstm/weights.300000.h5', type=str)
    parser.add_argument('--exp',dest='exp_name',
                        help='will be used to determine name of directory for picture storage',
                        default='crnn_large',type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset to train on',
                        default='ky', type=str)
    parser.add_argument('--gru', dest='gru',
                        help='use gru?',choices=[True,False],
                        default=False, type=str)
    parser.add_argument('--name', dest='name',
                        help='name of experiment',
                        default='unidirectional', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    #if len(sys.argv) == 1:
    #    parser.print_help()
    #    # sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    print('Called with args:')
    print(args)

    # gt_util =  get_dataset(name='ky',phase='train_extend')
    gt_util = get_dataset(name='ky',path=None)
    gt_util_synth = None



    
    # if args.dataset == 'ky' :
    #     gt_util_synth =  get_dataset(name=args.dataset,phase='train_synth')
    #     print('ky')
    # if args.dataset == 'coco':
    #     gt_util_synth =  get_dataset(name=args.dataset,phase='train')
    #     gt_util_synth_val =  get_dataset(name=args.dataset,phase='val')
    #     gt_util_synth  = gt_util_synth.merge(gt_util_synth_val)
    #     print('coco')
    # else :
    #     gt_util_synth = None



    weights_path = args.model_path
    
    output_dir = get_output_dir(gt_util,args.name)

    checkdir = os.path.join(output_dir,'checkpoints')
    print(checkdir)

    if os.path.exists(checkdir) == False:
        os.makedirs(checkdir)

    input_width = 256
    input_height = 32
    batch_size = 256
    input_shape = (input_width, input_height, 1)

    model, model_pred = CRNN(input_shape, len(alphabet), gru=args.gru,bidirectional=True)
    max_string_len = model_pred.output_shape[1]
    gen_train = InputGenerator(gt_util, batch_size, alphabet, input_shape[:2], 
                               grayscale=True, max_string_len=max_string_len,gt_util_synth=gt_util_synth)


    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    model.load_weights(weights_path,by_name=True)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)
    model.fit_generator(generator=gen_train.generate(), # batch_size here?
                        steps_per_epoch=gt_util.num_objects // batch_size,
                        epochs=40,
                        #validation_data=gen_val.generate(), # batch_size here?
                        #validation_steps=gt_util_val.num_objects // batch_size,
                        callbacks=[
                            ModelCheckpoint(output_dir+'/real_main.{epoch:03d}.h5', verbose=1, save_weights_only=True, period=5),
                            ModelSnapshot(checkdir, 10000),
                            Logger(checkdir),
                            keras.callbacks.TensorBoard(log_dir='./logs',
                                                                  batch_size=batch_size)

                        ],
                        initial_epoch=0)


