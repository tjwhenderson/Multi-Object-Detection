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
from torchvision.datasets import VOCDetection
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, RandomVerticalFlip, ToTensor, Normalize


# Other libraries for data manipulation and visualization
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import xml.etree.ElementTree as ET

class PascalVOC2012Dataset(Dataset):
    """Custom Dataset class for the 2012 PASCAL VOC Dataset.

    The expected dataset is stored in the "/datasets/PascalVOC2012/" on ieng6
    """
    def __init__(self, transform, root_dir, mode='trainval', download=False):
        super(PascalVOC2012Dataset, self).__init__()
        self.data = VOCDetection(root=root_dir, year='2012', \
                    transform=transform, image_set=mode, download=download)

        # self.classes = {0: "aeroplace", 1: "bicycle", 2: "bird", 3: "boat",
        #                 4: "bottle", 5: "bus", 6: "car", 7: "cat", 8: "chair",
        #                 9: "cow", 10: "diningtable", 11: "dog", 12: "horse",
        #                 13: "motorbike", 14: "person", 15: "pottedplant",
        #                 16: "sheep", 17: "sofa", 18: "train", 19: "tvmonitor"}

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

        ## convert to compatible label for YOLO
        yolo_label = self.get_yolo_label(label)
        return img, yolo_label

    def get_yolo_label(self, label):
        w = int(label['annotation']['size']['width'])
        h = int(label['annotation']['size']['height'])
        obj = label['annotation']['object']
        yolo_label = []

        if isinstance(obj, list): # contains multiple objects
            for obj_i in obj:
                clsid_i = self.classes.index( obj_i['name'] )
                b_i = (float(obj_i['bndbox']['xmin']), float(obj_i['bndbox']['xmax']), \
                       float(obj_i['bndbox']['ymin']), float(obj_i['bndbox']['ymax']))
                bbox_i = self.convert_bbox((w,h), b_i)
                yolo_label.append([clsid_i, bbox_i])

        else: # contains single object
            clsid = self.classes.index( obj['name'] )
            b = (float(obj['bndbox']['xmin']), float(obj['bndbox']['xmax']), \
                 float(obj['bndbox']['ymin']), float(obj['bndbox']['ymax']))
            bbox = self.convert_bbox((w,h), b)
            yolo_label.append([clsid, bbox])
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

    def convert_label(self, label, classes):
        """Convert the numerical label to n-hot encoding.

        Params:
        -------
        - label: a string of conditions corresponding to an image's class

        Returns:
        --------
        - binary_label: (Tensor) a binary encoding of the multi-class label
        """

        binary_label = torch.zeros(len(classes))
        for key, value in classes.items():
            if value in label:
                binary_label[key] = 1.0
        return binary_label

#%%

def create_split_loaders(root_dir, batch_size, seed=15,
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
    transform = Compose([Resize(416), RandomHorizontalFlip(),RandomVerticalFlip(),
                         ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = PascalVOC2012Dataset(root_dir=root_dir, transform=transform)

    # Dimensions and indices of training set
    dataset_size = dataset.__len__()
    all_indices = list(range(dataset_size))

    # Shuffle dataset before dividing into training & test sets
    if shuffle:
        np.random.seed(seed)
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

    num_workers = 0
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
    return (train_loader, val_loader, test_loader)

#%%
if __name__ == '__main__':
    root_dir = os.getcwd()
    
    transform = Compose([Resize(416), RandomHorizontalFlip(),RandomVerticalFlip(),
                         ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data = PascalVOC2012Dataset(root_dir=root_dir, transform=transform, download=False)
    img, label = data.__getitem__(0)
    print(label)

    dataloaders = create_split_loaders(root_dir=root_dir, batch_size=64)
