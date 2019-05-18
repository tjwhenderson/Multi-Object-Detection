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
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler

# Other libraries for data manipulation and visualization
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PascalVOC2012Dataset(Dataset):
    """Custom Dataset class for the 2012 PASCAL VOC Dataset.

    The expected dataset is stored in the "/datasets/PascalVOC2012/" on ieng6
    """

    def __init__(self, imgs_dir, labels_dir, img_size=416, mode='train'):
        self.img_data = []
        self.label_data = []
        self.img_size = img_size
        self.mode = mode
        self.imgs_dir = imgs_dir
        self.labels_dir = labels_dir

        self.classes = {0: "Person", 1: "Bird", 2: "Cat", 3: "Cow",
                        4: "Dog", 5: "Horse", 6: "Sheep", 7: "Aeroplane",
                        8: "Bicycle", 9: "Boat", 10: "Bus", 11: "Car", 12: "Motorbike",
                        13: "Train", 14: "Bottle", 15: "Chair", 16: "Dining Table",
                        17: "Potted Plant", 18: "Sofa", 19: "TV/Monitor"}

        for filename in sorted(os.listdir(self.labels_dir)):
            self.img_data.append(filename.replace("txt", "jpg"))

        for filename in sorted(os.listdir(self.labels_dir)):
            self.label_data.append(filename)

    def __len__(self):
        # Return the total number of data samples
        return len(self.img_data)

    def __getitem__(self, idx):
        """Returns the image and its label at the index 'ind'
        (after applying transformations to the image, if specified).

        Params:
        -------
        - ind: (int) The index of the image to get

        Returns:
        --------
        - A tuple (img, label)
        """

        img_path = os.path.join(self.imgs_dir,self.img_data[idx])
        img = Image.open(img_path).convert('RGB')
        transform = tv.transforms.Compose([
            tv.transforms.Resize(self.img_size),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomVerticalFlip(),
            tv.transforms.ToTensor()])

            # Do we need to normalize the tensor??
            # tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        label_path = os.path.join(self.labels_dir,self.label_data[idx])
        label = np.loadtxt(label_path).reshape(-1,5)
        return img, label

    # not sure if this is needed, so I left it
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



def create_split_loaders(imgs_dir, labels_dir, batch_size, seed, transform=transforms.ToTensor(),
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

    # Get create a ChestXrayDataset object
    dataset = PascalVOC2012Dataset(Dataset, imgs_dir, labels_dir)

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

if __name__ == '__main__':
    imgs_dir = './VOC2012/JPEGImages/'
    labels_dir = './VOC2012/labels/'
