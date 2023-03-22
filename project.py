#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" CIS6261TML -- Project Option 1 -- part1.py

# This file contains the part1 code
"""

import sys
import os

import time
import numpy as np
import pandas as pd
import sklearn
import cv2
from sklearn.metrics import confusion_matrix
from scipy.ndimage import median_filter, convolve
from glob import glob

# we'll use tensorflow and keras for neural networks
import tensorflow as tf
import tensorflow.keras as keras

import utils  # we need this

######### Prediction Fns #########


## Basic prediction function
def basic_predict(model, x):
    return model(x)


def randomized_smoothing_predict(
    model, x, mean: float = 0.0, sigma: float = 1.0, noise_type="Gaussian", raw: bool = False
):
    if noise_type.lower() == "gaussian":
        x_noisy = x + tf.random.normal(x.shape, mean=mean, stddev=sigma)
    elif noise_type.lower() == "laplace":
        x_noisy = x + np.random.laplace(loc=mean, scale=sigma, size=x.shape)

    if np.max(x) > 1:
        x_noisy_clipped = tf.clip_by_value(x_noisy, 0, 255.0)  # clip
    else:
        x_noisy_clipped = tf.clip_by_value(x_noisy, 0, 1.0)  # clip

    if raw:
        return x_noisy_clipped
    return model(x_noisy_clipped)


def local_medium_smoothing_predict(model, x, kernel_size: tuple = (2, 2, 2), mode: str = "reflect", raw: bool = False):
    filtered_image = median_filter(x, size=(1, *kernel_size), mode=mode)
    if raw:
        return filtered_image
    return model(filtered_image)


def color_bit_depth_reduction_predict(model, x, bit_depth: int = 8, raw: bool = False):
    bit_reduction = 2**bit_depth - 1
    x = (x * bit_reduction).astype(int)
    x = x.astype(float) / bit_reduction
    if raw:
        return x
    return model(x)


def smoothing_convolution_predict(model, x, filter_type: str = "smooth", raw: bool = False):
    if filter_type == "smooth":
        filter = np.array(
            [
                [1, 1, 1],
                [1, 5, 1],
                [1, 1, 1],
            ]
        )
    elif filter_type == "sharpen":
        filter = np.array(
            [
                [-2, -2, -2],
                [-2, 32, -2],
                [-2, -2, -2],
            ]
        )
    elif filter_type == "detail":
        filter = np.array(
            [
                [-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1],
            ]
        )
    elif filter_type == "blur":
        filter = np.array(
            [
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
            ]
        )
    else:
        filter = np.array(
            [
                [1, 1, 1],
                [1, 5, 1],
                [1, 1, 1],
            ]
        )

    # filters above were pulled from pillow library, need to divide them for use in [0, 1] range for our images
    filter = filter.astype(float) / 255

    # convolve requires filter to have same number of dimensions as input, so it needs to be 3d
    filter = np.array([filter, filter, filter])

    data = [convolve(i, filter) for i in x]

    x = np.array(data)

    if raw:
        return x
    return model(x)


def mean_denoising_predict(model, x, strength: float = 3, raw: bool = False):
    data = [
        cv2.fastNlMeansDenoisingColored(
            cv2.cvtColor((i * 255).astype(np.float32), cv2.COLOR_RGB2BGR).astype(np.uint8),
            None,
            h=strength,
            templateWindowSize=7,
            searchWindowSize=21,
        )
        for i in x
    ]

    # need to convert back to RGB and scale to 0, 1
    x = np.array([cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in data])

    x = x.astype(float) / 255

    if raw:
        return x
    return model(x)


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

    # storing data
    history = {}

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
    # train_loss, train_acc = model.evaluate(train_x, train_y, verbose=0)
    # test_loss, test_acc = model.evaluate(test_x, test_y, verbose=0)
    # print("[Raw Model] Train accuracy: {:.2f}% --- Test accuracy: {:.2f}%".format(100 * train_acc, 100 * test_acc))

    ### let's wrap the model prediction function so it could be replaced to implement a defense
    predict_fns = {
        "Basic": lambda x: basic_predict(model, x),
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
        "Randomized Gaussian 1.0 sigma": lambda x: randomized_smoothing_predict(
            model,
            x,
            sigma=1.0,
            noise_type="Gaussian",
        ),
        "Randomized Laplace 1.0 sigma": lambda x: randomized_smoothing_predict(
            model,
            x,
            sigma=1.0,
            noise_type="Laplace",
        ),
        "Local Median Smoothing Filter": lambda x: local_medium_smoothing_predict(model, x),
        "Color Bit Reduction 4bit": lambda x: color_bit_depth_reduction_predict(model, x, bit_depth=4),
        "Color Bit Reduction 2bit": lambda x: color_bit_depth_reduction_predict(model, x, bit_depth=2),
        "Non-local Mean denoising strength 0.8": lambda x: mean_denoising_predict(model, x, strength=0.8),
        "Non-local Mean denoising strength 1.5": lambda x: mean_denoising_predict(model, x, strength=1.5),
        "Non-local Mean denoising strength 3": lambda x: mean_denoising_predict(model, x, strength=3),
        "Non-local Mean denoising strength 7": lambda x: mean_denoising_predict(model, x, strength=7),
        "Non-local Mean denoising strength 10": lambda x: mean_denoising_predict(model, x, strength=10),
        # "Smoothing Convolution Filter": lambda x: smoothing_convolution_predict(model, x, filter_type="smooth"),
        # "Sharpen Convolution Filter": lambda x: smoothing_convolution_predict(model, x, filter_type="sharpen"),
        # "Detail Convolution Filter": lambda x: smoothing_convolution_predict(model, x, filter_type="detail"),
        # "Blur Convolution Filter": lambda x: smoothing_convolution_predict(model, x, filter_type="blur"),
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

        history[name] = {"Train Accuracy": 100 * train_acc, "Test Accuracy": 100 * test_acc}

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

            history[name].update(
                {
                    f"{attack_str} Acc": attack_acc * 100,
                    f"{attack_str} Adv": attack_adv,
                    f"{attack_str} Prec": attack_precision,
                    f"{attack_str} Recall": attack_recall,
                    f"{attack_str} F1": attack_f1,
                }
            )

        ### evaluating the robustness of the model wrt adversarial examples
        print("\n------------ Adversarial Examples ----------")
        advexp_fps = []
        advexp_fps.append(("Adversarial examples attack0", os.path.join("attacks", "advexp0.npz")))
        advexp_fps.append(("Adversarial examples attack1", os.path.join("attacks", "advexp1.npz")))
        # our created attacks
        for attack in sorted(glob(os.path.join("attacks", "*.npz"))):
            if "advexp0" in attack or "advexp1" in attack:
                continue

            advexp_fps.append((f"Adversarial attack {os.path.basename(attack)}", attack))

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

            history[name].update(
                {f"{attack_str} Benign Acc": 100 * benign_acc, f"{attack_str} Adver Acc": 100 * adv_acc}
            )

        print("------------\n")

        et = time.time()

        print(
            "Elapsed time -- total: {:.1f} seconds (data & model loading: {:.1f} seconds)".format(
                et - st, st_after_model - st
            )
        )

    # print consolidated results
    headers = [
        "Train Accuracy",
        "Test Accuracy",
    ]
    for n in mia_attack_fns:
        headers += [
            f"{n[0]} Acc",
            f"{n[0]} Adv",
            f"{n[0]} Prec",
            f"{n[0]} Recall",
            f"{n[0]} F1",
        ]

    for n in advexp_fps:
        headers += [n[0] + " Benign Acc", n[0] + " Adver Acc"]

    data = pd.DataFrame(list(history.values()), columns=headers, index=list(history.keys()))
    data.index.name = "Predicton Function"
    print(data)

    data.to_csv("results.csv")

    sys.exit(0)
