#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" CIS6261TML -- Project Option 1 -- utils.py

# This file contains utility functions
"""

import sys
import os

import time

import matplotlib.pyplot as plt

import numpy as np
import sklearn

# we'll use tensorflow and keras for neural networks
import tensorflow as tf
import tensorflow.keras as keras


"""
## Plots a set of images (all m x m)
## input is  a square number of images, i.e., np.array with shape (z*z, dim_x, dim_y) for some integer z > 1
"""
def plot_images(im, dim_x=32, dim_y=32, one_row=False, out_fp='out.png', save=False, show=True, cmap='gray', fig_size=(14,14), titles=None, titles_fontsize=12):
    fig = plt.figure(figsize=fig_size)
    if im.shape[-1] != 3:
        im = im.reshape((-1, dim_x, dim_y))

    num = im.shape[0]
    assert num <= 3 or np.sqrt(num)**2 == num or one_row, 'Number of images is too large or not a perfect square!'
    
    if titles is not None:
        assert num == len(titles)
    
    if num <= 3:
        for i in range(0, num):
            plt.subplot(1, num, 1 + i)
            plt.axis('off')
            if type(cmap) == list:
                assert len(cmap) == num
                plt.imshow((im[i]*255).astype(np.uint8), cmap=cmap[i]) # plot raw pixel data
            else:
                plt.imshow((im[i]*255).astype(np.uint8), cmap=cmap) # plot raw pixel data
            if titles is not None:
                plt.title(titles[i], fontsize=titles_fontsize)
    else:
        sq = int(np.sqrt(num))
        for i in range(0, num):
            if one_row:
                plt.subplot(1, num, 1 + i)
            else:
                plt.subplot(sq, sq, 1 + i)
            plt.axis('off')
            if type(cmap) == list:
                assert len(cmap) == num
                plt.imshow(im[i], cmap=cmap[i]) # plot raw pixel data
            else:
                plt.imshow(im[i], cmap=cmap) # plot raw pixel data
            if titles is not None:
                plt.title(titles[i], fontsize=titles_fontsize)

    if save:
        plt.savefig(out_fp)

    if show:
        plt.show()
    else:
        plt.close()
   
   
"""
## Loads CIFAR10 data used for training the target model. 
    - (train_x, train_y) was used for training.
    - (val_x, val_y) and (test_x, test_y) are disjoint and were not used during training.
    - labels are the class labels
"""
def load_data(fp='./data.npz'):
    with np.load(fp) as data:
        train_x = data['train_x']
        train_y = data['train_y']
        
        test_x = data['test_x']
        test_y = data['test_y']
        
        val_x = data['val_x']
        val_y = data['val_y']
        
        labels = data['labels']

    assert np.amax(train_x) <= 1 and np.amax(test_x) <= 1 and np.amax(val_x) <= 1
    assert np.amax(train_x) >= 0 and np.amax(test_x) >= 0 and np.amax(val_x) >= 0

    assert labels.shape[0] == 10 and labels.shape[0] == train_y.shape[1]

    print('Loaded dataset --- train_x shape: {}, train_y shape: {}, labels: {}'.format(train_x.shape, train_y.shape, labels))
    
    return train_x, train_y, test_x, test_y, val_x, val_y, labels
    
    
import hashlib

def memv_filehash(fp):
    hv = hashlib.sha256()
    buf = bytearray(512 * 1024)
    memv = memoryview(buf)
    with open(fp, 'rb', buffering=0) as f:
        for n in iter(lambda : f.readinto(memv), 0):
            hv.update(memv[:n])
    return hv.hexdigest()
    
"""
## Load model from file
"""        
def load_model(fp):
    hv = memv_filehash(fp)
    fg = hv[-21:-1].upper()

    model = keras.models.load_model(fp)
    model.trainable = False

    print('Loaded model from file ({}) -- [{}].'.format(fp, fg))
    return model, hv
    

    
"""
## Computes predicted label and confidence for examples 'x' using pred_fn() to access the model's predictions.
"""    
def pred_label_and_conf(pred_fn, x, num_classes=10):
    preds = pred_fn(x).reshape(-1, num_classes)
    pred_label = np.argmax(preds, axis=-1)
    pred_conf = preds[pred_label]
    
    return pred_label, pred_conf
    
