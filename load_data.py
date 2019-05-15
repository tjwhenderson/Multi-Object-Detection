#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:07:02 2019

@author: jdeguzman
"""
import pickle

if __name__ == '__main__':
    # change this path to point to the pickle file
    filename = open('./PASCALVOC2012.pkl', 'rb')

    # objects is a dictionary containing the objects
    # imgs is a list with the names of the image files
    objects = pickle.load(filename)
    imgs = pickle.load(filename)
    filename.close()
