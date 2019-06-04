from __future__ import division
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def bbox_iou(box1, box2, already_exact=True):
    """
    Returns the IOU of two specified bounding boxes
    """
    if not already_exact:
        # Transform x,y,w,h, --> xmin,ymin,xmax,ymax
        b1_xmin, b1_xmax = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_ymin, b1_ymax = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_xmin, b2_xmax = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_ymin, b2_ymax = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_xmin, b1_ymin, b1_xmax, b1_ymax = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        b2_xmin, b2_ymin, b2_xmax, b2_ymax = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # Coordinates of intersection rectangle
    rect_inter_x1 =  torch.max(b1_xmin, b2_xmin)
    rect_inter_y1 =  torch.max(b1_ymin, b2_ymin)
    rect_inter_x2 =  torch.min(b1_xmax, b2_xmax)
    rect_inter_y2 =  torch.min(b1_ymax, b2_ymax)
    
    # Are of intersection
    area_inter = torch.clamp(rect_inter_x2 - rect_inter_x1 + 1, min=0) * torch.clamp(rect_inter_y2 - rect_inter_y1 + 1, min=0)
    
    # Area of union
    area_b1 = (b1_xmax - b1_xmin + 1) * (b1_ymax - b1_ymin + 1)
    area_b2 = (b2_xmax - b2_xmin + 1) * (b2_ymax - b2_ymin + 1)

    iou = area_inter / (area_b1 + area_b2 - area_inter + 1e-16)

    return iou