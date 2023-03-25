#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" CIS6261TML -- Project Option 1 -- part1.py

# This file contains the part1 code
"""

import sys
import os

import time

import numpy as np

import sklearn
from sklearn.metrics import confusion_matrix

# we'll use tensorflow and keras for neural networks
import tensorflow as tf
import tensorflow.keras as keras

import utils # we need this

    
"""
## Plots an adversarial perturbation, i.e., original input orig_x, adversarial example adv_x, and the difference (perturbation)
"""
def plot_adversarial_example(pred_fn, orig_x, adv_x, labels, fname='adv_exp.png', show=True, save=True):
    perturb = adv_x - orig_x
    
    # compute confidence
    in_label, in_conf = utils.pred_label_and_conf(pred_fn, orig_x)
    
    # compute confidence
    adv_label, adv_conf = utils.pred_label_and_conf(pred_fn, adv_x)
    
    titles = ['{} (conf: {:.2f})'.format(labels[in_label], in_conf), 'Perturbation',
              '{} (conf: {:.2f})'.format(labels[adv_label], adv_conf)]
    
    images = np.r_[orig_x, perturb, adv_x]
    
    # plot images
    utils.plot_images(images, fig_size=(8,3), titles=titles, titles_fontsize=12,  out_fp=fname, save=save, show=show)  


######### Prediction Fns #########

"""
## Basic prediction function
"""
def basic_predict(model, x):
    return model(x)


#### TODO: implement your defense(s) as a new prediction function
#### Put your code here
def randomized_smoothing_function(model, x):
    num_samples=1
    noise_type='gaussian'
    sigma=0.01
    y_pred_avg = np.zeros((x.shape[0], 10))
    for i in range(0, num_samples):
        #GAUSSIAN NOISE

        if(noise_type=='gaussian'):
            gaussian_noise=np.random.normal(0,sigma)
            x_noisy=x+gaussian_noise

        #LAPLACE NOISE
        elif(noise_type=='laplace'):
            laplace_noise=np.random.laplace(loc=0.0, scale=sigma)
            x_noisy=x+laplace_noise

        x_noisy_clipped = tf.clip_by_value(x_noisy, 0, 1.0)

        y_pred = model.predict(x_noisy_clipped, verbose=0)
        assert y_pred.shape == y_pred_avg.shape
        #TRIED TO DISTORT LABEL PREDICTION IN EACH SAMPLE ITERATION BY 5%
        # y_pred=tf.clip_by_value(y_pred*0.05, 0, 10.0)
        y_pred=y_pred*0.05
        y_pred_avg += y_pred
    
    y_pred_avg /= num_samples
    return y_pred_avg



######### Membership Inference Attacks (MIAs) #########

"""
## A very simple threshold-based MIA
"""
def simple_conf_threshold_mia(predict_fn, x, thresh=0.9999):   
    pred_y = predict_fn(x)
    pred_y_conf = np.max(pred_y, axis=-1)
    return (pred_y_conf > thresh).astype(int)
    
    
#### TODO [optional] implement new MIA attacks.
#### Put your code here
  
  
######### Adversarial Examples #########

  
#### TODO [optional] implement new adversarial examples attacks.
#### Put your code here  
#### Note: you can have your code save the data to file so it can be loaded and evaluated in Main() (see below).
    
   
######### Main() #########
   
if __name__ == "__main__":


    # Let's check our software versions
    print('### Python version: ' + __import__('sys').version)
    print('### NumPy version: ' + np.__version__)
    print('### Scikit-learn version: ' + sklearn.__version__)
    print('### Tensorflow version: ' + tf.__version__)
    print('### TF Keras version: ' + keras.__version__)
    print('------------')


    # global parameters to control behavior of the pre-processing, ML, analysis, etc.
    seed = 42

    # deterministic seed for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # keep track of time
    st = time.time()

    #### load the data
    print('\n------------ Loading Data & Model ----------')
    
    train_x, train_y, test_x, test_y, val_x, val_y, labels = utils.load_data()
    num_classes = len(labels)
    assert num_classes == 10 # cifar10
    
    ### load the target model (the one we want to protect)
    target_model_fp = './target-model.h5'

    model, _ = utils.load_model(target_model_fp)
    ## model.summary() ## you can uncomment this to check the model architecture (ResNet)
    
    st_after_model = time.time()
        
    ### let's evaluate the raw model on the train and test data
    train_loss, train_acc = model.evaluate(train_x, train_y, verbose=0)
    test_loss, test_acc = model.evaluate(test_x, test_y, verbose=0)
    print('[Raw Model] Train accuracy: {:.2f}% --- Test accuracy: {:.2f}%'.format(100*train_acc, 100*test_acc))
    
    
    ### let's wrap the model prediction function so it could be replaced to implement a defense
    predict_fn = lambda x: randomized_smoothing_function(model, x)

    ### now let's evaluate the model with this prediction function
    pred_y = predict_fn(train_x)
    # print(type(pred_y))
    train_acc = np.mean(np.argmax(train_y, axis=-1) == np.argmax(pred_y, axis=-1))
    
    pred_y = predict_fn(test_x)
    test_acc = np.mean(np.argmax(test_y, axis=-1) == np.argmax(pred_y, axis=-1))
    print('[Model] Train accuracy: {:.2f}% --- Test accuracy: {:.2f}%'.format(100*train_acc, 100*test_acc))
        
    
    ### evaluating the privacy of the model wrt membership inference
    mia_eval_size = 2000
    mia_eval_data_x = np.r_[train_x[0:mia_eval_size], test_x[0:mia_eval_size]]
    mia_eval_data_in_out = np.r_[np.ones((mia_eval_size,1)), np.zeros((mia_eval_size,1))]
    assert mia_eval_data_x.shape[0] == mia_eval_data_in_out.shape[0]
    
    # so we can add new attack functions as needed
    print('\n------------ Privacy Attacks ----------')
    mia_attack_fns = []
    mia_attack_fns.append(('Simple MIA Attack', simple_conf_threshold_mia))
    
    for i, tup in enumerate(mia_attack_fns):
        attack_str, attack_fn = tup
        
        in_out_preds = attack_fn(predict_fn, mia_eval_data_x).reshape(-1,1)
        assert in_out_preds.shape == mia_eval_data_in_out.shape, 'Invalid attack output format'
        
        cm = confusion_matrix(mia_eval_data_in_out, in_out_preds, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        
        attack_acc = np.trace(cm) / np.sum(np.sum(cm))
        attack_adv = tp / (tp + fn) - fp / (fp + tn)
        attack_precision = tp / (tp + fp)
        attack_recall = tp / (tp + fn)
        attack_f1 = tp / (tp + 0.5*(fp + fn))
        print('{} --- Attack accuracy: {:.2f}%; advantage: {:.3f}; precision: {:.3f}; recall: {:.3f}; f1: {:.3f}'.format(attack_str, attack_acc*100, attack_adv, attack_precision, attack_recall, attack_f1))
    
    
    
    ### evaluating the robustness of the model wrt adversarial examples
    print('\n------------ Adversarial Examples ----------')
    advexp_fps = []
    advexp_fps.append(('Adversarial examples attack0', 'advexp0.npz'))
    advexp_fps.append(('Adversarial examples attack1', 'advexp1.npz'))
    
    for i, tup in enumerate(advexp_fps):
        attack_str, attack_fp = tup
        
        data = np.load(attack_fp)
        adv_x = data['adv_x']
        benign_x = data['benign_x']
        benign_y = data['benign_y']
        
        benign_pred_y = predict_fn(benign_x)
        #print(benign_y[0:10], benign_pred_y[0:10])
        benign_acc = np.mean(benign_y == np.argmax(benign_pred_y, axis=-1))
        
        adv_pred_y = predict_fn(adv_x)
        #print(benign_y[0:10], adv_pred_y[0:10])
        adv_acc = np.mean(benign_y == np.argmax(adv_pred_y, axis=-1))
        
        print('{} --- Benign accuracy: {:.2f}%; adversarial accuracy: {:.2f}%'.format(attack_str, 100*benign_acc, 100*adv_acc))
        
    print('------------\n')

    et = time.time()
    
    print('Elapsed time -- total: {:.1f} seconds (data & model loading: {:.1f} seconds)'.format(et - st, st_after_model - st))

    sys.exit(0)
