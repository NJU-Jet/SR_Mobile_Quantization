import os
import argparse
import cv2
import numpy as np
from options import parse
from solvers import Solver
from data import DIV2K
import matplotlib as mpl
import matplotlib.pyplot as plt
import shutil
import os
import os.path as osp
from tensorboardX import SummaryWriter
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FSRCNN Demo')
    parser.add_argument('--opt', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--scale', default=3, type=int)
    parser.add_argument('--ps', default=48, type=int, help='patch_size')
    parser.add_argument('--bs', default=16, type=int, help='batch_size')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--gpu_ids', default=None)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--resume_path', default=None)
    parser.add_argument('--qat', action='store_true', default=False)
    parser.add_argument('--qat_path', default=None)

    args = parser.parse_args()
    args, lg = parse(args)

    # Tensorboard save directory
    resume = args['solver']['resume']
    tensorboard_path = 'Tensorboard/{}'.format(args['name'])

    if resume==False:
        if osp.exists(tensorboard_path):
            shutil.rmtree(tensorboard_path, True)
            lg.info('Remove dir: [{}]'.format(tensorboard_path))
    writer = SummaryWriter(tensorboard_path)

    # create dataset
    train_data = DIV2K(args['datasets']['train'])
    lg.info('Create train dataset successfully!')
    lg.info('Training: [{}] iterations for each epoch'.format(len(train_data)))
        
    val_data = DIV2K(args['datasets']['val'])
    lg.info('Create val dataset successfully!')
    lg.info('Validating: [{}] iterations for each epoch'.format(len(val_data)))
        
    # create solver
    lg.info('Preparing for experiment: [{}]'.format(args['name']))
    solver = Solver(args, train_data, val_data, writer)

    # train
    lg.info('Start training...')
    solver.train()
