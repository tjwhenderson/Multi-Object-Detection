import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

"""
Modified version of the one on the YOLO repository

This parses the XML files for PASCALVOC2012. Original missed the years 2007 and 2012
and thus resulted in ~11000 labels. When this is run with the modifications,
it will generate ~17000 labels in the form of text files.
"""

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", \
           "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", \
           "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert(size, box):
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

def convert_annotation(year, image_id):
    in_file = open('./VOC%s/Annotations/%s.xml'%(year, image_id))
    out_file = open('./VOC%s/Labels/%s.txt'%(year, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

year = '2012'
if not os.path.exists('./VOC%s/Labels/'%(year)):
    os.makedirs('./VOC%s/Labels/'%(year))
image_ids = []

for filename in sorted(os.listdir('./VOC2012/Annotations/')):
    image_ids.append(filename.replace('.xml',''))

image_ids = image_ids[1:]
list_file = open('%s_files.txt'%(year), 'w')

for image_id in image_ids:
    list_file.write('%s/VOC%s/JPEGImages/%s\n'%(wd, year, image_id))
    convert_annotation(year, image_id)
list_file.close()

# for year, image_set in sets:
#     if not os.path.exists('./VOC%s/Labels/'%(year)):
#         os.makedirs('./VOC%s/Labels/'%(year))
#     image_ids = open('./VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
#
#     list_file = open('%s_%s.txt'%(year, image_set), 'w')
#
#     for image_id in image_ids:
#         list_file.write('%s/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
#         convert_annotation(year, image_id)
#     list_file.close()

# os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train.txt")
# os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt")
