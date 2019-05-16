#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:07:02 2019

@author: jdeguzman
"""
import pickle

if __name__ == '__main__':
    # change this path to point to the pickle file
    file1= open('./PASCALVOC2012.pkl', 'rb')

    # objects is a dictionary containing the objects
    # imgs is a list with the names of the image files
    objects = pickle.load(file1)
    imgs = pickle.load(file1)
    file1.close()

    # load the PASCALVOC2012 DataFrame which contains the
    # bounding boxes of the objects
    file2 = open('./PASCAL_DF.pkl', 'rb')
    bndbox_df = pickle.load(file2)
    file2.close()
