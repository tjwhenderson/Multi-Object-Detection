from utils import compute_AP, NMS
from yolo_model import yoloModel

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, RandomVerticalFlip, ToTensor, Normalize

import PASCAL_Dataloader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import PIL

CLASS_ID = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", \
                "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", \
                "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def detect(test_loader, test_data, is_training=False):
    ## TODO: Load YOLO model

    print('Performing object detection!')

    use_cuda = torch.cuda.is_available()
    # Setup GPU optimization if CUDA is supported
    if use_cuda:
        computing_device = torch.device("cuda")
        print("CUDA is supported")
    else: # Otherwise, train on the CPU
        computing_device = torch.device("cpu")

    imgs = []
    img_detections = []  # Stores detections for each image index
    img_labels = []

    for batch_i, samples in enumerate(test_loader):
        images, labels = samples["image"], samples["label"]

        images = images.to(computing_device)
        labels = labels.to(computing_device)
        with torch.no_grad():
            detections = net(images)
            detections = NMS(detections)

        # save for plotting
        imgs.extend(images)
        img_detections.extend(detections)
        img_labels.extend(labels)
    return imgs, img_detections, img_labels

def plot_bbox(img, detections, label=None):
    """ img : the original pre-transformed image
        detections : (x1, y1, x2, y2, obj_conf, cls_score, cls_pred)
        label : ground truth label
    """

    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    ORI_H, ORI_W = img.shape[:2]
    H, W = 416, 416

    # Draw bounding boxes and labels of detections
    if detections is not None:
        unique_labels = detections[:, -1]  # get all cls_pred
        n_cls_preds = len(unique_labels)  # get unique num
        bbox_colors = random.sample(colors, n_cls_preds)  # color for each

        # These preds are sorted by classes, then by conf (in each class)
        cur_class = -1

        for pred_i, (x1, y1, x2, y2, obj_conf, cls_score, cls_pred) in enumerate(detections):
            cls_pred = cls_pred.item()
            cls_id = CLASS_ID[int(cls_pred)]

            # Rescale coordinates to original dimensions
            x = x1 * (ORI_W / W)
            y = y1 * (ORI_H / H)
            w = abs(x2 - x1) * (ORI_W / W)
            h = abs(y2 - y1) * (ORI_H / H)

            color = bbox_colors[int(cls_pred)]
            if cur_class != cls_pred:
                # This means the pred is the top one of that class
                cls_name = "*" + cls_id
                zorder = 20
                cur_class = cls_pred
            else:
                cls_name = cls_id
                zorder = 4

             # Create a Rectangle patch
            bbox = patches.Rectangle((x, y), w, h, linewidth=2,
                                     edgecolor=color, facecolor='none',
                                     zorder=zorder)

            # Add the bbox and label to the plot
            ax.add_patch(bbox)
            plt.text(x, y, s=cls_id, color='white',
                             verticalalignment='top',
                             bbox={'color': color, 'pad': 0},
                             fontsize=8, zorder=zorder)

    else:
        print('No objects detected!')

    plt.axis('off')
    # plt.gca().xaxis.set_major_locator(NullLocator())
    # plt.gca().yaxis.set_major_locator(NullLocator())

    # Save generated image with detections
    # pth = os.path.join(output_folder_path, '{}.png'.format(img_i))
    # plt.savefig(pth, bbox_inches='tight', pad_inches=0.0)
    # plt.close()
