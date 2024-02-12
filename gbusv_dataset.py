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


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
    

class GbVideoDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, return_all_video_frames=True, data_split='all'):
        self.transform = transform

        self.root = root
        self.transform = transform
        self.return_all_video_frames = return_all_video_frames ## True
        self.data_split = data_split
        # self.num_of_sampled_frames = num_of_sampled_frames ##1

        self._get_annotations()

        self.loader = pil_loader

    @staticmethod
    def get_video_name(name):
        return name.split("/")[-2]

    @staticmethod
    def get_frame_id(name):
        return int(name.split("/")[-1][:-4])

    def get_image_paths(self):
        print('path ############', self.data_basepath)
        return sorted(list(tqdm.tqdm(glob.iglob(os.path.join(self.data_basepath, "*/*.jpg")))))

    def get_image_name(self, key: str, ind: int):
        return os.path.join(self.data_split_path, key,  "%05d.jpg" % ind)

    def video_id_frame_id_split(self, name):
        return self.get_video_name(name), self.get_frame_id(name)

    def _get_single_frame(self, path_key, ind):
        return self.transform(self.loader(self.get_image_name(path_key, ind)))

    def _get_annotations(self):
        self.data_basepath = self.root
        self.data_split_path = os.path.join(self.data_basepath)

        # create a flattened list of all image paths
        pickle_path = os.path.join(self.data_basepath, "all_paths.pkl")
        if not os.path.exists(pickle_path):
            print('creat new cache')
            images = self.get_image_paths()
            samples = []
            video_names = []
            video_count = 0
            video_frames = sorted([self.video_id_frame_id_split(name) for name in images])
            for vid_id, ind in video_frames:
                if vid_id not in video_names:
                    video_names.append(vid_id)
                    video_count += 1
                path = self.get_image_name(vid_id, ind)
                label = video_count - 1
                samples.append((path, label))
            pickle.dump(samples, open(pickle_path, "wb"))
        self.samples = pickle.load(open(pickle_path, "rb"))
        print("Num of videos %d frames %d" % (len(set([e[1] for e in self.samples])), len(self.samples)))

        # # create a dict of video names and corresponding frames
        # pickle_path = os.path.join(self.data_basepath, self.data_split+ "_names.pkl")
        # if not os.path.exists(pickle_path):
        #     print('creat new cache')
        #     images = self.get_image_paths()
        #     path_info = OrderedDict()
        #     video_names = sorted([self.video_id_frame_id_split(name) for name in images])
        #     for vid_id, ind in video_names:
        #         if vid_id not in path_info:
        #             path_info[vid_id] = []
        #         path_info[vid_id].append(ind)
        #     path_info = sorted([(key, val) for key, val in path_info.items()])
        #     # os.makedirs(self.data_split_path, exist_ok=True)
        #     pickle.dump(path_info, open(pickle_path, "wb"))
        # self.path_info = pickle.load(open(pickle_path, "rb"))
        # num_frames = int(np.sum([len(p_info[1]) for p_info in self.path_info]))
        # print("Num for %s videos %d frames %d" % (self.data_split, len(self.path_info), num_frames))


    def __getitem__(self, index):
        path, target = self.samples[index]

        ## Loading the image at the chosen index 
        if self.transform is not None:
            sample = self.loader(path)
            sample = self.transform(sample)
        
        return sample, target, index
        

    def __len__(self):
        return len(self.samples)


    #     # Get the list of subdirectories (each subdirectory represents a class)
    #     self.classes = [d.name for d in os.scandir(main_directory) if d.is_dir()]

    #     # Create a list to store image paths and corresponding labels
    #     self.data = []
    #     self.samples = []

    #     # Iterate through subdirectories to collect image paths and labels
    #     for i, class_name in enumerate(self.classes):
    #         class_path = os.path.join(main_directory, class_name)
    #         image_paths = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith('.jpg')]
    #         self.data.extend([(img_path, i) for img_path in image_paths])
    #         self.samples.append((img_path, i) for img_path in image_paths)
    #     print(len(self.samples))

    # def __len__(self):
    #     return len(self.data)

    # def __getitem__(self, index):
    #     # Load and preprocess the image
    #     img_path, label = self.samples[index]
    #     image = Image.open(img_path)#.convert('RGB')

    #     if self.transform:
    #         image = self.transform(image)

    #     return image, label, index