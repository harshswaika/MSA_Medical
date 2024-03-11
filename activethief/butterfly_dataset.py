import os
import glob
import random

from PIL import Image
import numpy as np
from skimage import color
import scipy.io as sio
import tqdm
import pickle
from collections import OrderedDict, defaultdict

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import multiprocessing
from torch.multiprocessing import Pool
import cv2
import math

"""
source: https://github.com/983632847/USCL/blob/main/train_USCL/data_aug/dataset_wrapper_Ultrasound_Video_Mixup.py
"""
    
<<<<<<< HEAD
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
    
=======

>>>>>>> ba1060430d61521041e28b47a409dc50ce80da3d
class ButterflyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, LabelList=None, DataList=None):
        """
        Ultrasound self-supervised training Dataset, only choose one image from a video
        :param data_dir: str
        :param transform: torch.transform
        """
        # self.data_info = self.get_img_info(data_dir)
        self.transform = transform
        self.LabelList = LabelList
        self.DataList = DataList
<<<<<<< HEAD
        self.loader = pil_loader
        self.samples = self.get_img_info(data_dir)

    def __getitem__(self, index):
        sample, target = self.samples[index]  # list
        # print("index", self.samples[index])
        # path_img = random.sample(path_imgs, 1)  # random choose one image
        # img1 = Image.open(path_imgs).convert('RGB')  # 0~255
        # img2 = Image.open(path_imgs).convert('RGB')  # 0~255
        # label1 = 0 if path_img[0].lower()[64:].find("cov") > -1 else (1 if path_img[0].lower()[64:].find("pneu") > -1 else 2)

        # if self.transform is not None:
        #     img1, img2 = self.transform((img1, img2))  # transform

        if self.transform is not None:
            sample = self.loader(sample)
            sample = self.transform(sample)
        
        return sample, target, index

    def __len__(self):  # len
        return len(self.samples)
=======

        self.samples = self.get_img_info(data_dir)

    def __getitem__(self, index):
        path_imgs, samples = self.data_info[index]  # list
        path_img = random.sample(path_imgs, 1)  # random choose one image
        img1 = Image.open(path_img[0]).convert('RGB')  # 0~255
        img2 = Image.open(path_img[0]).convert('RGB')  # 0~255
        label1 = 0 if path_img[0].lower()[64:].find("cov") > -1 else (1 if path_img[0].lower()[64:].find("pneu") > -1 else 2)

        if self.transform is not None:
            img1, img2 = self.transform((img1, img2))  # transform

        return img1, label1, img2, label1

    def __len__(self):  # len
        return len(self.data_info)
>>>>>>> ba1060430d61521041e28b47a409dc50ce80da3d
    
    @staticmethod
    def get_img_info(data_dir):
        # data_info = list()
        samples = []
        video_count = 0
        for root, dirs, _ in os.walk(data_dir):
            for sub_dir in dirs:  # one video as one class
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg') or x.endswith('.png'), img_names))
                path_imgs = []
<<<<<<< HEAD
                # print("sub", sub_dir)
=======
>>>>>>> ba1060430d61521041e28b47a409dc50ce80da3d
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    path_imgs.append(path_img)
<<<<<<< HEAD
                    samples.append((path_img, video_count))
                # data_info.append(path_imgs)
                video_count += 1

        # print(samples)
=======
                    samples.extend((path_img, video_count))
                # data_info.append(path_imgs)
                video_count += 1

>>>>>>> ba1060430d61521041e28b47a409dc50ce80da3d
        return samples