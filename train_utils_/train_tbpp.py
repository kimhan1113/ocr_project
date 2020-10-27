import numpy as np
import matplotlib.pyplot as plt
import keras
import os
import argparse
import sys

this_dir = os.path.dirname(__file__)
sys.path.insert(0, this_dir + '/..')

from config.config import cfg,get_output_dir0
from lib.tbpp.tbpp_model import get_network 
from lib.tbpp.tbpp_utils import PriorUtil
from lib.tbpp.train import  train_tbpp
from lib.datasets.factory import get_dataset
from lib.utils.model import load_weights

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='evaluate tbpp model')
    parser.add_argument('--weights', dest='model_path',
                        help='initialize with pretrained model weights',
                        default='data/models/tbpp/weights.018.h5', type=str)
    parser.add_argument('--exp',dest='exp_name',
                        help='will be used to determine name of directory for picture storage',
                        default='default',type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset to test on',
                        default='ky', type=str)
    parser.add_argument('--network', dest='network',
                        help='name of the network',choices=['dense','vgg16'],
                        default='dense', type=str)
    parser.add_argument('--name', dest='name',
                        help='name of experiment',
                        default='tbpp_large', type=str)
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

    gt_util =  get_dataset(name=args.dataset,phase='train_extend')
    print(gt_util.num_samples)
    model,freeze = get_network(args.network)
    weights_path = args.model_path 
    prior_util = PriorUtil(model)
    output_dir = get_output_dir(gt_util,args.name)

    if weights_path is not None:
        model.load_weights( weights_path,by_name=True)
        print('successfully loaded model weights')

    train_tbpp(model,gt_util,prior_util,cfg,output_dir,freeze=freeze)

    






