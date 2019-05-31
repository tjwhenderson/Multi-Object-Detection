################################################################################
# ECE 285: Final Project - Multi-Object Detection
# Spring 2019
#
#
#
# Description:
# This code defines a custom PyTorch Dataset object suited for the
# 2012 PASCAL VOC dataset with 20 unique categories. This dataset contains
# approximately 100,000 images. This is a well-known dataset for object detection,
# classification, segmentation of objects and so on. The nominative
# labels are mapped to an integer between 0-19, which is later converted into
# an n-hot binary encoded label.
################################################################################

# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler
#from torchvision.datasets import VOCDetection
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, RandomVerticalFlip, ToTensor, Normalize


import os
import sys
import tarfile
import collections
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import xml.etree.ElementTree as ET

#%%
class VOCDetection(Dataset):
    def __init__(self, root, image_set='trainval', transforms=None):
        super(VOCDetection, self).__init__()
        self.root = root
        self.image_set = image_set
        self.transforms = transforms

        voc_root = self.root # roots to PASCALVOC2012
        image_dir = os.path.join(voc_root, 'JPEGImages')
        annotation_dir = os.path.join(voc_root, 'Annotations')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets/Main')

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val" or a valid'
                'image_set from the VOC ImageSets/Main folder.')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]
        assert (len(self.images) == len(self.annotations))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict


#%%
class PascalVOC2012Dataset(Dataset):
    """Custom Dataset class for the 2012 PASCAL VOC Dataset.

    The expected dataset is stored in the "/datasets/PascalVOC2012/" on ieng6
    """
    def __init__(self, root, transforms=None, mode='trainval'):
        super(PascalVOC2012Dataset, self).__init__()
        self.data = VOCDetection(root=root, image_set=mode)
        self.transforms = transforms
        
        self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", \
                        "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", \
                        "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    def __len__(self):
        """Returns the total number of samples in the dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Returns the image and its label at the index 'ind'
        (after applying transformations to the image, if specified).

        Params:
        -------
        - ind: (int) The index of the image to get

        Returns:
        --------
        - img: transformed tensor image at the specified index
        - label: corresponding label of the image as an array
                 with the values : [class, x, y, w, h]
        """

        img, label = self.data.__getitem__(idx)

        if self.transforms is not None:
            img = self.transforms(img)
            
        ## convert to compatible label for YOLO
        yolo_label = self.get_yolo_label(label)
        sample = {"image": img, "label": yolo_label}
        return sample

    def get_yolo_label(self, label):
        w = int(label['annotation']['size']['width'])
        h = int(label['annotation']['size']['height'])
        obj = label['annotation']['object']

        yolo_label = np.zeros((100, 5))
        if isinstance(obj, list): # contains multiple objects          
            for n, obj_i in enumerate(obj):
                clsid_i = self.classes.index( obj_i['name'] )
                b_i = (float(obj_i['bndbox']['xmin']), float(obj_i['bndbox']['xmax']), \
                       float(obj_i['bndbox']['ymin']), float(obj_i['bndbox']['ymax']))
                bbox_i = self.convert_bbox((w,h), b_i)
                yolo_label[n,0] = clsid_i
                yolo_label[n,1:] = bbox_i

        else: # contains single object
            clsid = self.classes.index( obj['name'] )
            b = (float(obj['bndbox']['xmin']), float(obj['bndbox']['xmax']), \
                 float(obj['bndbox']['ymin']), float(obj['bndbox']['ymax']))
            bbox = self.convert_bbox((w,h), b)
            yolo_label[0,0] = clsid
            yolo_label[0,1:] = bbox
        return yolo_label

    def convert_bbox(self, size, box):
        dw = 1./(size[0])
        dh = 1./(size[1])
        x = (box[0] + box[1])/2.0 - 1
        y = (box[2] + box[3])/2.0 - 1
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return (x,y,w,h)


#%%

def create_split_loaders(root_dir, batch_size,
                         p_val=0.1, p_test=0.2, shuffle=True,
                         show_sample=False, extras={}):
    """ Creates the DataLoader objects for the training, validation, and test sets.

    Params:
    -------
    - imgs_dir: directory containing the image files
    - labels_dir: directory containing the label files
    - batch_size: (int) mini-batch size to load at a time
    - seed: (int) Seed for random generator (use for testing/reproducibility)
    - transform: A torchvision.transforms object - transformations to apply to each image
                 (Can be "transforms.Compose([transforms])")
    - p_val: (float) Percent (as decimal) of dataset to use for validation
    - p_test: (float) Percent (as decimal) of the dataset to split for testing
    - shuffle: (bool) Indicate whether to shuffle the dataset before splitting
    - show_sample: (bool) Plot a mini-example as a grid of the dataset
    - extras: (dict)
        If CUDA/GPU computing is supported, contains:
        - num_workers: (int) Number of subprocesses to use while loading the dataset
        - pin_memory: (bool) For use with CUDA - copy tensors into pinned memory
                  (set to True if using a GPU)
        Otherwise, extras is an empty dict.

    Returns:
    --------
    - train_loader: (DataLoader) The iterator for the training set
    - val_loader: (DataLoader) The iterator for the validation set
    - test_loader: (DataLoader) The iterator for the test set
    """
    
    root_dir = './VOCdevkit/VOC2012'
    tf = Compose([
            Resize((416,416)),
#            transforms.RandomHorizontalFlip(),
#            transforms.RandomVerticalFlip(),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    dataset = PascalVOC2012Dataset(root=root_dir, mode='trainval', transforms=tf)
    
    # Dimensions and indices of training set
    dataset_size = dataset.__len__()
    all_indices = list(range(dataset_size))

    # Shuffle dataset before dividing into training & test sets
    if shuffle:
        np.random.seed(15)
        np.random.shuffle(all_indices)

    # Create the validation split from the full dataset
    val_split = int(np.floor(p_val * dataset_size))
    train_ind, val_ind = all_indices[val_split :], all_indices[: val_split]

    # Separate a test split from the training dataset
    test_split = int(np.floor(p_test * len(train_ind)))
    train_ind, test_ind = train_ind[test_split :], train_ind[: test_split]

    # Use the SubsetRandomSampler as the iterator for each subset
    sample_train = SubsetRandomSampler(train_ind)
    sample_test = SubsetRandomSampler(test_ind)
    sample_val = SubsetRandomSampler(val_ind)

    num_workers = 32
    pin_memory = False

    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]

    # Define the training, test, & validation DataLoaders
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=sample_train, num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(dataset, batch_size=batch_size,
                             sampler=sample_test, num_workers=num_workers,
                              pin_memory=pin_memory)

    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=sample_val, num_workers=num_workers,
                            pin_memory=pin_memory)


    # Return the training, validation, test DataLoader objects
    return train_loader, val_loader, test_loader

#%%
#if __name__ == '__main__':
#    root_dir = '../VOCdevkit/VOC2012'
#    
#    tf = Compose([
#            Resize((416,416)),
##            transforms.RandomHorizontalFlip(),
##            transforms.RandomVerticalFlip(),
#            ToTensor(),
#            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#        ])
#
#    dataset = PascalVOC2012Dataset(root=root_dir, mode='trainval', transforms=tf)
#    print(dataset.__len__())


    
