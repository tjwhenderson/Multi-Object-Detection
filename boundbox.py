#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 16:53:22 2019

@author: jdeguzman
"""

class Boundbox:
    def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c = c
        self.classes = classes

def get_bbox_abscoords(bbox):
    """ converts x, y, w, h, to x.min, x.max, y.min, y.max """
    x, y, w, h, = bbox[:,0], bbox[:,2], bbox[:,1], bbox[:,3]
    xmin, xmax = (x - w/2), (x + w/2)
    ymin, ymax = (y - h/2), (y + h/2)
    bbox_abs = Boundbox(xmin, ymin, xmax, ymax)
    return bbox_abs


def IOU(bbox1, bbox2):
    bbox1_abs = get_bbox_abscoords(bbox1)
    bbox2_abs = get_bbox_abscoords(bbox1)
    intersect_w = bbox_intersect([bbox1_abs.xmin, bbox1_abs.xmax], [bbox2_abs.xmin, bbox2_abs.xmax])
    intersect_h = bbox_intersect([bbox1_abs.ymin, bbox1_abs.ymax], [bbox2_abs.ymin, bbox2_abs.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = bbox1[:,1], bbox1[:,3]
    w2, h2 = bbox2[:,1], bbox2[:,3]

    union = w1*h1 + w2*h2 - intersect
    return float(intersect) / union

def bbox_intersect(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3
