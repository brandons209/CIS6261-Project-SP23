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

import utils  # we need this


######### Prediction Fns #########


## Basic prediction function
def basic_predict(model, x):
    return model(x)


#### TODO: implement your defense(s) as a new prediction function
#### Put your code here
def randomized_smoothing_predict(model, x, mean: float = 0.0, sigma: float = 1.0, noise_type="Gaussian"):
    if noise_type.lower() == "gaussian":
        x_noisy = x + tf.random.normal(x.shape, mean=mean, stddev=sigma)
    elif noise_type.lower() == "laplace":
        x_noisy = x + np.random.laplace(loc=mean, scale=sigma, size=x.shape)

    if np.max(x) > 1:
        x_noisy_clipped = tf.clip_by_value(x_noisy, 0, 255.0)  # clip
    else:
        x_noisy_clipped = tf.clip_by_value(x_noisy, 0, 1.0)  # clip

    return model(x_noisy_clipped)


######### Membership Inference Attacks (MIAs) #########


## A very simple threshold-based MIA
def simple_conf_threshold_mia(predict_fn, x, thresh=0.9999):
    pred_y = predict_fn(x)
    pred_y_conf = np.max(pred_y, axis=-1)
    return (pred_y_conf > thresh).astype(int)


#### TODO [optional] implement new MIA attacks.
#### Put your code here
def mia_attack(predict_fn, x):
    pass


######### Main() #########

if __name__ == "__main__":
    model_path = "./target-model.h5"

    # Let's check our software versions
    print("### Python version: " + __import__("sys").version)
    print("### NumPy version: " + np.__version__)
    print("### Scikit-learn version: " + sklearn.__version__)
    print("### Tensorflow version: " + tf.__version__)
    print("### TF Keras version: " + keras.__version__)
    print("------------")

    # global parameters to control behavior of the pre-processing, ML, analysis, etc.
    seed = 42

    # deterministic seed for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # keep track of time
    st = time.time()

    #### load the data
    print("\n------------ Loading Data & Model ----------")

    train_x, train_y, test_x, test_y, val_x, val_y, labels = utils.load_data()
    num_classes = len(labels)
    assert num_classes == 10  # cifar10

    ### load the target model (the one we want to protect)

    model, _ = utils.load_model(model_path)
    print(f"Loaded model: {model.name}")
    # model.summary()  ## you can uncomment this to check the model architecture (ResNet)

    st_after_model = time.time()

    ### let's evaluate the raw model on the train and test data
    train_loss, train_acc = model.evaluate(train_x, train_y, verbose=0)
    test_loss, test_acc = model.evaluate(test_x, test_y, verbose=0)
    print("[Raw Model] Train accuracy: {:.2f}% --- Test accuracy: {:.2f}%".format(100 * train_acc, 100 * test_acc))

    ### let's wrap the model prediction function so it could be replaced to implement a defense
    predict_fns = {
        "Basic": lambda x: basic_predict(model, x),
        "Randomized Gaussian 0.01 sigma": lambda x: randomized_smoothing_predict(
            model,
            x,
            sigma=0.01,
            noise_type="Gaussian",
        ),
        "Randomized Laplace 0.01 sigma": lambda x: randomized_smoothing_predict(
            model,
            x,
            sigma=0.01,
            noise_type="Laplace",
        ),
        "Randomized Gaussian 0.05 sigma": lambda x: randomized_smoothing_predict(
            model,
            x,
            sigma=0.05,
            noise_type="Gaussian",
        ),
        "Randomized Laplace 0.05 sigma": lambda x: randomized_smoothing_predict(
            model,
            x,
            sigma=0.05,
            noise_type="Laplace",
        ),
        "Randomized Gaussian 0.1 sigma": lambda x: randomized_smoothing_predict(
            model,
            x,
            sigma=0.1,
            noise_type="Gaussian",
        ),
        "Randomized Laplace 0.1 sigma": lambda x: randomized_smoothing_predict(
            model,
            x,
            sigma=0.1,
            noise_type="Laplace",
        ),
        "Randomized Gaussian 0.1 sigma mean 0.5": lambda x: randomized_smoothing_predict(
            model,
            x,
            mean=0.5,
            sigma=0.1,
            noise_type="Gaussian",
        ),
        "Randomized Laplace 0.1 sigma mean 0.5": lambda x: randomized_smoothing_predict(
            model,
            x,
            mean=0.5,
            sigma=0.1,
            noise_type="Laplace",
        ),
        "Randomized Gaussian 0.25 sigma": lambda x: randomized_smoothing_predict(
            model,
            x,
            sigma=0.25,
            noise_type="Gaussian",
        ),
        "Randomized Laplace 0.25 sigma": lambda x: randomized_smoothing_predict(
            model,
            x,
            sigma=0.25,
            noise_type="Laplace",
        ),
        "Randomized Gaussian 0.5 sigma": lambda x: randomized_smoothing_predict(
            model,
            x,
            sigma=0.5,
            noise_type="Gaussian",
        ),
        "Randomized Laplace 0.5 sigma": lambda x: randomized_smoothing_predict(
            model,
            x,
            sigma=0.5,
            noise_type="Laplace",
        ),
        "Randomized Gaussian 1 sigma": lambda x: randomized_smoothing_predict(
            model,
            x,
            sigma=1,
            noise_type="Gaussian",
        ),
        "Randomized Laplace 1 sigma": lambda x: randomized_smoothing_predict(
            model,
            x,
            sigma=1,
            noise_type="Laplace",
        ),
        "Randomized Gaussian 1.5 sigma": lambda x: randomized_smoothing_predict(
            model,
            x,
            sigma=1.5,
            noise_type="Gaussian",
        ),
        "Randomized Laplace 1.5 sigma": lambda x: randomized_smoothing_predict(
            model,
            x,
            sigma=1.5,
            noise_type="Laplace",
        ),
    }

    for i, predict_fn in enumerate(predict_fns.items()):
        name, predict_fn = predict_fn
        print(f"Evaluation against prediction function '{name}'")
        ### now let's evaluate the model with this prediction function
        pred_y = predict_fn(train_x)
        train_acc = np.mean(np.argmax(train_y, axis=-1) == np.argmax(pred_y, axis=-1))

        pred_y = predict_fn(test_x)
        test_acc = np.mean(np.argmax(test_y, axis=-1) == np.argmax(pred_y, axis=-1))
        print("[Model] Train accuracy: {:.2f}% --- Test accuracy: {:.2f}%".format(100 * train_acc, 100 * test_acc))

        ### evaluating the privacy of the model wrt membership inference
        mia_eval_size = 2000
        mia_eval_data_x = np.r_[train_x[0:mia_eval_size], test_x[0:mia_eval_size]]
        mia_eval_data_in_out = np.r_[np.ones((mia_eval_size, 1)), np.zeros((mia_eval_size, 1))]
        assert mia_eval_data_x.shape[0] == mia_eval_data_in_out.shape[0]

        # so we can add new attack functions as needed
        print("\n------------ Privacy Attacks ----------")
        mia_attack_fns = []
        mia_attack_fns.append(("Simple MIA Attack", simple_conf_threshold_mia))

        for i, tup in enumerate(mia_attack_fns):
            attack_str, attack_fn = tup

            in_out_preds = attack_fn(predict_fn, mia_eval_data_x).reshape(-1, 1)
            assert in_out_preds.shape == mia_eval_data_in_out.shape, "Invalid attack output format"

            cm = confusion_matrix(mia_eval_data_in_out, in_out_preds, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            attack_acc = np.trace(cm) / np.sum(np.sum(cm))
            attack_adv = tp / (tp + fn) - fp / (fp + tn)
            attack_precision = tp / (tp + fp)
            attack_recall = tp / (tp + fn)
            attack_f1 = tp / (tp + 0.5 * (fp + fn))
            print(
                "{} --- Attack accuracy: {:.2f}%; advantage: {:.3f}; precision: {:.3f}; recall: {:.3f}; f1: {:.3f}".format(
                    attack_str, attack_acc * 100, attack_adv, attack_precision, attack_recall, attack_f1
                )
            )

        ### evaluating the robustness of the model wrt adversarial examples
        print("\n------------ Adversarial Examples ----------")
        advexp_fps = []
        advexp_fps.append(("Adversarial examples attack0", os.path.join("attacks", "advexp0.npz")))
        advexp_fps.append(("Adversarial examples attack1", os.path.join("attacks", "advexp1.npz")))
        advexp_fps.append(("Adversarial gradient attack2", os.path.join("attacks", "adv2_gradient_attack.npz")))
        advexp_fps.append(("Adversarial fgsm     attack3", os.path.join("attacks", "adv3_fgsm.npz")))
        advexp_fps.append(
            ("Adversarial mifgsm   attack4", os.path.join("attacks", "adv3_mifgsm_alpha_0.1_decay_0.1.npz"))
        )

        for i, tup in enumerate(advexp_fps):
            attack_str, attack_fp = tup

            data = np.load(attack_fp, allow_pickle=True)
            adv_x = data["adv_x"]
            benign_x = data["benign_x"]
            benign_y = data["benign_y"]

            benign_pred_y = predict_fn(benign_x)
            # print(benign_y[0:10], benign_pred_y[0:10])
            benign_acc = np.mean(benign_y == np.argmax(benign_pred_y, axis=-1))

            adv_pred_y = predict_fn(adv_x)
            # print(benign_y[0:10], adv_pred_y[0:10])
            adv_acc = np.mean(benign_y == np.argmax(adv_pred_y, axis=-1))

            print(
                "{} --- Benign accuracy: {:.2f}%; adversarial accuracy: {:.2f}%".format(
                    attack_str, 100 * benign_acc, 100 * adv_acc
                )
            )

        print("------------\n")

        et = time.time()

        print(
            "Elapsed time -- total: {:.1f} seconds (data & model loading: {:.1f} seconds)".format(
                et - st, st_after_model - st
            )
        )

    sys.exit(0)
