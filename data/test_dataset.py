import h5py
import numpy as np
import tensorflow as tf
import cv2
import random
import os
import os.path as osp
import pickle


class Test(tf.keras.utils.Sequence):
    def __init__(self, opt):
        self.dataroot_hr = opt['dataroot_HR']
        self.dataroot_lr = opt['dataroot_LR']
        self.filename_path = opt['filename_path']

        self.img_list = []
        with open(self.filename_path, 'r') as f:
            filenames = f.readlines()
        for line in filenames:
            self.img_list.append(line.strip())


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        lr, hr = self.get_image_pair(idx)
        lr_batch, hr_batch = np.expand_dims(lr, 0), np.expand_dims(hr, 0)

        #return (lr_batch).astype(np.uint8), (hr_batch).astype(np.uint8)
        return lr_batch.astype(np.float32), hr_batch.astype(np.float32)

    def get_image_pair(self, idx):
        hr_path = osp.join(self.dataroot_hr, self.img_list[idx])
        base, ext = osp.splitext(self.img_list[idx])
        lr_basename = base + 'x3' + '.pt'
        lr_path = osp.join(self.dataroot_lr, lr_basename)
        
        # load img
        hr = self.read_img(hr_path)
        lr = self.read_img(lr_path)

        return lr, hr

    def read_img(self, img_path):
        with open(img_path, 'rb') as f:
            img = pickle.load(f)

        return img
