import h5py
import numpy as np
import tensorflow as tf
import cv2
import random
import os
import os.path as osp
import pickle


class DIV2K(tf.keras.utils.Sequence):
    def __init__(self, opt):
        self.dataroot_hr = opt['dataroot_HR']
        self.dataroot_lr = opt['dataroot_LR']
        self.filename_path = opt['filename_path']
        self.scale = opt['scale']
        self.split = opt['split']
        self.patch_size = opt['patch_size']
        self.batch_size = opt['batch_size']
        self.flip = opt['flip']
        self.rot = opt['rot']
        self.enlarge_times = opt['enlarge_times']

        self.img_list = []
        with open(self.filename_path, 'r') as f:
            filenames = f.readlines()
        for line in filenames:
            self.img_list.append(line.strip())

    def shuffle(self):
        random.shuffle(self.img_list)


    def __len__(self):
        if self.split == 'train':
            return int(len(self.img_list)*self.enlarge_times/self.batch_size)
            
        else:
            return len(self.img_list)

    def __getitem__(self, idx):
        start = (idx*self.batch_size) 
        end = start + self.batch_size
        if self.split == 'train':
            lr_batch = np.zeros((self.batch_size, self.patch_size, self.patch_size, 3), dtype=np.float32)
            hr_batch = np.zeros((self.batch_size, self.patch_size*self.scale, self.patch_size*self.scale, 3), dtype=np.float32)
            for i in range(start, end):
                lr, hr = self.get_image_pair(i%len(self.img_list))
                lr_batch[i-start] = lr
                hr_batch[i-start] = hr
        else:
            lr, hr = self.get_image_pair(idx)
            lr_batch, hr_batch = np.expand_dims(lr, 0), np.expand_dims(hr, 0)

        return (lr_batch).astype(np.float32), (hr_batch).astype(np.float32)

    def get_image_pair(self, idx):
        hr_path = osp.join(self.dataroot_hr, self.img_list[idx])
        base, ext = osp.splitext(self.img_list[idx])
        lr_basename = base + 'x{}'.format(self.scale) + '.pt'
        lr_path = osp.join(self.dataroot_lr, lr_basename)
        
        # load img
        hr = self.read_img(hr_path)
        lr = self.read_img(lr_path)

        if self.split == 'train':
            lr_patch, hr_patch = self.get_patch(lr, hr, self.patch_size, self.scale)
            lr, hr = self.augment(lr_patch, hr_patch, self.flip, self.rot)
        
        return lr, hr

    def read_img(self, img_path):
        with open(img_path, 'rb') as f:
            img = pickle.load(f)

        return img

    def get_patch(self, lr, hr, ps, scale):
        lr_h, lr_w = lr.shape[:2]
        hr_h, hr_w = hr.shape[:2]

        lr_x = random.randint(0, lr_w - ps)
        lr_y = random.randint(0, lr_h - ps)
        hr_x = lr_x * scale
        hr_y = lr_y * scale

        lr_patch = lr[lr_y:lr_y+ps, lr_x:lr_x+ps, :]
        hr_patch = hr[hr_y:hr_y+ps*scale, hr_x:hr_x+ps*scale, :]

        return lr_patch, hr_patch

    def augment(self, lr, hr, flip, rot):
        hflip = flip and random.random() < 0.5
        vflip = flip and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        if hflip:
            lr = np.ascontiguousarray(lr[:, ::-1, :])
            hr = np.ascontiguousarray(hr[:, ::-1, :])
        if vflip:
            lr = np.ascontiguousarray(lr[::-1, :, :])
            hr = np.ascontiguousarray(hr[::-1, :, :])
        if rot90:
            lr = lr.transpose(1, 0, 2)
            hr = hr.transpose(1, 0, 2)
    
        return lr, hr
