#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 18:16:29 2019

@author: jdeguzman
"""

import xmltodict
import os
import pandas as pd
import pickle

class preprocessing():
    def __init__(self, folder):
        self.filelist = []
        self.filedir = folder
        for filename in sorted(os.listdir(folder)):
            self.filelist.append(filename)

    def parse_objects(self):
        IMGS = [] # list of filenames of images
        OBJECTS = {} # dictionary containing the objects with key being filename

        for f in self.filelist:
            with open(os.path.join(self.filedir, f)) as fd:
                xmldict = xmltodict.parse(fd.read())

            filename = xmldict['annotation']['filename']
            obj = xmldict['annotation']['object']
            IMGS.append(filename)

            if isinstance(obj, list):
                isMultiObj = True
                OBJECTS[filename] = obj
            else:
                isMultiObj = False
                OBJECTS[filename]= [obj]
        return OBJECTS, IMGS

    def convert_to_dataframe(self, objects, imgs):
        # function to convert dictionary of objects into pandas dataframe
        bndbox_df = pd.DataFrame(columns=['name','bndbox','image']) # creates a new empty dataframe

        for ii, imgfile in enumerate(imgs):
            df = pd.DataFrame.from_dict(objects[imgfile])
            df = df.filter(['name','bndbox'])
            df['image'] = imgfile
            bndbox_df = bndbox_df.append(df,ignore_index = True)
        return bndbox_df


if __name__ == '__main__':
    xml_dir = 'VOC2012/Annotations'
#    xml_files = load_files_from_folder(xml_dir)
    data = preprocessing(xml_dir)
    objects, imgs = data.parse_objects()
    bndbox_df = data.convert_to_dataframe(objects, imgs)

    # save items to pickle file
    # filename = open('./PASCALVOC2012.pkl', 'wb')
    # pickle.dump(objects, filename)
    # pickle.dump(imgs, filename)
    # filename.close()
