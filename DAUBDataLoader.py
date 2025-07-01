import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from numpy import *
import numpy as np
import scipy.io as scio
import cv2
from match_images import matching


#load image
class DAUB_TrainSetLoader(Dataset):
    def __init__(self, root, fullSupervision=True, align=True):
        # 加载图像路径 #align表示是否进行图像对齐
        txtpath = ('./dataset/DAUB_DTUM/train_DAUB.txt')
        txt = np.loadtxt(txtpath, dtype=str)  # 直接读取字符串类型
        self.imgs_arr = txt
        self.root = root
        self.fullSupervision = fullSupervision
        self.align = align
        self.train_mean = 127.29936981201172
        self.train_std = 27.624435424804688
        self.target_size = (256,256)
        if self.align==True :
            print('是否进行对齐：是')
        if self.align==False:
            print('是否进行对齐：否')  

    def __getitem__(self, index):
        # 获取完整的图像路径
  
        img_path = self.imgs_arr[index]
        
        # 从路径中解析序列名和帧号
        base_name = os.path.basename(img_path)
        frame = int(os.path.splitext(base_name)[0])  # 帧号是从文件名提取的
        seq_dir = os.path.dirname(img_path)  # 序列名是从路径中提取的

        # 读取当前帧
        img_ori = cv2.imread(img_path)
        if np.ndim(img_ori) == 3:
            img_ori = img_ori[:,:,0]
        img_ori = self.process_image(img_ori, self.target_size)
        img = np.expand_dims(img_ori.astype(np.float32), axis=0)


        for i in range(1,5):
            img_hispath = os.path.join(seq_dir, str(max(frame-i, 0)) + '.bmp')
            img_his = cv2.imread(img_hispath)
            if np.ndim(img_his) == 3:
                img_his= img_his[:,:,0]
            if self.align:
                img_his = matching(img_his, img_ori)
            img_his = self.process_image(img_his, self.target_size)
            img_his = np.expand_dims(img_his.astype(np.float32), axis=0)
            img= np.concatenate((img_his, img), axis=0)


        # 读取标签
        label_path = img_path.replace('images', 'masks').replace('.bmp', '.png')
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = self.process_image(label, self.target_size, interpolation=cv2.INTER_NEAREST)
        label = label.astype(np.float32) / 255.0

        # 预处理
        img = (img - self.train_mean) / self.train_std
        img = torch.unsqueeze(torch.from_numpy(img), 0)
        label = torch.unsqueeze(torch.from_numpy(label), 0)

        [_, m, n] = label.shape
        return img, label, m, n
    def process_image(self, image, target_size, interpolation=cv2.INTER_NEAREST):
        # 缩放到目标尺寸
        image = cv2.resize(image, target_size, interpolation=interpolation)
        return image

    def __len__(self):
        return len(self.imgs_arr)



class DAUB_TestSetLoader(Dataset):
    def __init__(self, root, fullSupervision=True, align=True):
        # 加载图像路径 #align表示是否进行图像对齐
        txtpath = ('./dataset/DAUB_DTUM/test_DAUB.txt')
        txt = np.loadtxt(txtpath, dtype=str)  # 直接读取字符串类型
        self.imgs_arr = txt
        self.root = root
        self.fullSupervision = fullSupervision
        self.align = align
        self.train_mean = 127.29936981201172
        self.train_std = 27.624435424804688
        self.target_size = (256,256)

    def __getitem__(self, index):
        # 获取完整的图像路径
        img_path = self.imgs_arr[index]

        # 从路径中解析序列名和帧号
        base_name = os.path.basename(img_path)
        frame = int(os.path.splitext(base_name)[0])  # 帧号是从文件名提取的
        seq_dir = os.path.dirname(img_path)  # 序列目录是从路径中提取的
        # 读取当前帧
        img_ori = cv2.imread(img_path)
        if np.ndim(img_ori) == 3:
            img_ori = img_ori[:,:,0]
    
        img_ori = self.process_image(img_ori, self.target_size)
        img = np.expand_dims(img_ori.astype(np.float32), axis=0)

        for i in range(1,5):
            img_hispath = os.path.join(seq_dir, str(max(frame-i, 0)) + '.bmp')
            img_his = cv2.imread(img_hispath)
            if np.ndim(img_his) == 3:
                img_his= img_his[:,:,0]
            if self.align:
                img_his = matching(img_his, img_ori)
            img_his = self.process_image(img_his, self.target_size)
            img_his = np.expand_dims(img_his.astype(np.float32), axis=0)
            img= np.concatenate((img_his, img), axis=0)
        # 读取标签
        label_path = img_path.replace('images', 'masks').replace('.bmp', '.png')
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = self.process_image(label, self.target_size, interpolation=cv2.INTER_NEAREST)
        label = label.astype(np.float32) / 255.0

        # 预处理
        img = (img - self.train_mean) / self.train_std
        img = torch.unsqueeze(torch.from_numpy(img), 0)
        label = torch.unsqueeze(torch.from_numpy(label), 0)

        [_, m, n] = label.shape
        return img, label, m, n
    def process_image(self, image, target_size, interpolation=cv2.INTER_NEAREST):
        # 缩放到目标尺寸
        image = cv2.resize(image, target_size, interpolation=interpolation)
        return image
    
    def __len__(self):
        return len(self.imgs_arr)
