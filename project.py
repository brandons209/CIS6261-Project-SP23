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
from scipy import stats
from glob import glob
from skimage.util import random_noise
from keras.applications.convnext import LayerScale

# we'll use tensorflow and keras for neural networks
import tensorflow as tf
import tensorflow.keras as keras

import utils  # we need this

######### Prediction Fns #########


## Basic prediction function
def basic_predict(model, x, part: str = "part1", batch_size: int = 32):
    return (
        model.predict(x, verbose=0, batch_size=batch_size)
        if part != "part2"
        else model.predict(x * 255, verbose=0, batch_size=batch_size)
    )


def randomized_smoothing_predict(
    model,
    x,
    mean: float = 0.0,
    sigma: float = 1.0,
    noise_type="Gaussian",
    raw: bool = False,
    part: str = "part1",
    batch_size: int = 32,
):
    with tf.device("/CPU:0"):
        if noise_type.lower() == "gaussian":
            x_noisy = x + tf.random.normal(x.shape, mean=mean, stddev=sigma)
        elif noise_type.lower() == "laplace":
            x_noisy = x + np.random.laplace(loc=mean, scale=sigma, size=x.shape)
        elif noise_type.lower() == "poisson":
            x_noisy = x + np.random.poisson(lam=sigma, size=x.shape)

        x_noisy_clipped = tf.clip_by_value(x_noisy, 0, 1.0)  # clip

        if raw:
            return x_noisy_clipped

        x_noisy_clipped = x_noisy_clipped if part != "part2" else x_noisy_clipped * 255

    return model.predict(x_noisy_clipped, verbose=0, batch_size=batch_size)


def distort_output_predict(model, x, y: np.array = None, amount=0.05, part: str = "part1", batch_size: int = 32):
    if y is not None:
        return y * (1 + amount)

    with tf.device("/CPU:0"):
        x = x if part != "part2" else x * 255

    return model.predict(x, verbose=0, batch_size=batch_size) * (1 + amount)


def salt_and_pepper_noise_predict(
    model, x, amount: float = 0.05, raw: bool = False, part: str = "part1", batch_size: int = 32
):
    with tf.device("/CPU:0"):
        x_noisy = random_noise(x, mode="s&p", amount=amount, clip=True)
        if raw:
            return x_noisy

        x_noisy = x_noisy if part != "part2" else x_noisy * 255

    return model.predict(x_noisy, verbose=0, batch_size=batch_size)


def speckle_noise_predict(model, x, amount: float = 0.05, raw: bool = False, part: str = "part1", batch_size: int = 32):
    with tf.device("/CPU:0"):
        x_noisy = random_noise(x, mode="speckle", mean=0, var=amount)

        if raw:
            return x_noisy
        x_noisy = x_noisy if part != "part2" else x_noisy * 255

    return model.predict(x_noisy, verbose=0, batch_size=batch_size)


def local_medium_smoothing_predict(
    model,
    x,
    kernel_size: tuple = (2, 2, 2),
    mode: str = "reflect",
    raw: bool = False,
    part: str = "part1",
    batch_size: int = 32,
):
    with tf.device("/CPU:0"):
        filtered_image = median_filter(x, size=(1, *kernel_size), mode=mode)
        if raw:
            return filtered_image

        filtered_image = filtered_image if part != "part2" else filtered_image * 255

    return model.predict(filtered_image, verbose=0, batch_size=batch_size)


def color_bit_depth_reduction_predict(
    model, x, bit_depth: int = 8, raw: bool = False, part: str = "part1", batch_size: int = 32
):
    bit_reduction = 2**bit_depth - 1

    with tf.device("/CPU:0"):
        x = (x * bit_reduction).astype(int)
        x = x.astype(float) / bit_reduction
        if raw:
            return x
        x = x if part != "part2" else x * 255

    return model.predict(x, verbose=0, batch_size=batch_size)


def smoothing_convolution_predict(
    model, x, filter_type: str = "smooth", raw: bool = False, part: str = "part1", batch_size: int = 32
):
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

    with tf.device("/CPU:0"):
        data = [convolve(i, filter) for i in x]

        x = np.array(data)

        if raw:
            return x

    return (
        model.predict(x, verbose=0, batch_size=batch_size)
        if part != "part2"
        else model.predict(x * 255, verbose=0, batch_size=batch_size)
    )


def mean_denoising_predict(model, x, strength: float = 3, raw: bool = False, part: str = "part1", batch_size: int = 32):
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
    return (
        model.predict(x, verbose=0, batch_size=batch_size)
        if part != "part2"
        else model.predict(x * 255, verbose=0, batch_size=batch_size)
    )


def defense_distillation_autoencoder(
    model, x, autoencoder_path: str = "ae_defense_best.h5", part: str = "part1", batch_size: int = 32
):
    ae, _ = utils.load_model(autoencoder_path)

    distilled_x = ae.predict(x, verbose=0, batch_size=batch_size)

    return (
        model.predict(distilled_x, verbose=0, batch_size=batch_size)
        if part != "part2"
        else model.predict(distilled_x * 255, verbose=0, batch_size=batch_size)
    )


######### Membership Inference Attacks (MIAs) #########


## A very simple threshold-based MIA
def simple_conf_threshold_mia(predict_fn, x, thresh=0.9999):
    pred_y = predict_fn(x)
    pred_y_conf = np.max(pred_y, axis=-1)
    return (pred_y_conf > thresh).astype(int)


#### NEW MIA attacks.
def compute_loss(y_true, y_pred):
    loss = keras.backend.categorical_crossentropy(
        tf.convert_to_tensor(y_true),
        tf.convert_to_tensor(y_pred),
        from_logits=False,
    )
    return keras.backend.eval(loss)


def do_loss_attack(
    x_targets, y_targets, predict_fn, loss_fn, mean_train_loss, std_train_loss, mean_test_loss, std_test_loss
):
    pv = predict_fn(x_targets)
    loss_vec = loss_fn(y_targets, pv)

    in_or_out_pred = np.zeros((x_targets.shape[0],))

    gauss_train = stats.norm(mean_train_loss, std_train_loss).pdf(loss_vec)
    gauss_test = stats.norm(mean_test_loss, std_test_loss).pdf(loss_vec)
    in_or_out_pred = np.where(gauss_train > gauss_test, 1, 0)

    return in_or_out_pred


def do_loss_attack2(x_targets, y_targets, predict_fn, loss_fn, mean_train_loss, std_train_loss, threshold=0.6):
    pv = predict_fn(x_targets)
    loss_vec = loss_fn(y_targets, pv)

    in_or_out_pred = np.zeros((x_targets.shape[0],))

    gauss = stats.norm(mean_train_loss, std_train_loss).cdf(loss_vec)
    in_or_out_pred = np.where(gauss < threshold, 1, 0)

    return in_or_out_pred


"""
## Membership inference attack based on Shokri et al. (2017)
"""


def shokri_mia(predict_fn, shadow_train_x, shadow_train_y):
    # generate the dataset for training the attack model
    attack_train_x = predict_fn(shadow_train_x)
    attack_train_y = shadow_train_y

    # define the attack model
    attack_model = keras.Sequential(
        [
            keras.layers.Input(shape=(attack_train_x.shape[1],)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    attack_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # train the attack model
    attack_model.fit(attack_train_x, attack_train_y, epochs=20, batch_size=64, verbose=0)

    # evaluate the attack model
    in_out_preds = (attack_model.predict(predict_fn(shadow_train_x)) > 0.5).astype(int)
    return in_out_preds


######### Main() #########

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    model_path = "./part2_model_best.h5"
    part = "part2"
    batch_size = 4

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

    train_x, train_y, test_x, test_y, _, _, labels = utils.load_data()
    if part == "part2":
        train_x, train_y, test_x, test_y = utils.keras_load_data()
        train_x = train_x.astype(float) / 255
        test_x = test_x.astype(float) / 255

    num_classes = len(labels)
    assert num_classes == 10  # cifar10

    ### load the target model (the one we want to protect)

    model, _ = utils.load_model(model_path, custom_objects={"LayerScale": LayerScale})
    print(f"Loaded model: {model.name}")
    # model.summary()  ## you can uncomment this to check the model architecture (ResNet)

    st_after_model = time.time()

    ### let's evaluate the raw model on the train and test data
    # train_loss, train_acc = model.evaluate(train_x, train_y, verbose=0)
    # test_loss, test_acc = model.evaluate(test_x, test_y, verbose=0)
    # print("[Raw Model] Train accuracy: {:.2f}% --- Test accuracy: {:.2f}%".format(100 * train_acc, 100 * test_acc))

    ### let's wrap the model prediction function so it could be replaced to implement a defense
    predict_fns = {
        # "Basic": lambda x: basic_predict(model, x, part=part, batch_size=batch_size),
        # "Randomized Gaussian 0.05 sigma": lambda x: randomized_smoothing_predict(
        #    model, x, sigma=0.05, noise_type="Gaussian", part=part, batch_size=batch_size
        # ),
        # "Randomized Laplace 0.05 sigma": lambda x: randomized_smoothing_predict(
        #    model, x, sigma=0.05, noise_type="Laplace", part=part, batch_size=batch_size
        # ),
        # "Randomized Gaussian 0.1 sigma": lambda x: randomized_smoothing_predict(
        #    model, x, sigma=0.1, noise_type="Gaussian", part=part, batch_size=batch_size
        # ),
        # "Randomized Laplace 0.1 sigma": lambda x: randomized_smoothing_predict(
        #    model, x, sigma=0.1, noise_type="Laplace", part=part, batch_size=batch_size
        # ),
        # "Randomized Gaussian 0.2 sigma": lambda x: randomized_smoothing_predict(
        #    model, x, sigma=0.2, noise_type="Gaussian", part=part, batch_size=batch_size
        # ),
        # "Randomized Laplace 0.2 sigma": lambda x: randomized_smoothing_predict(
        #    model, x, sigma=0.2, noise_type="Laplace", part=part, batch_size=batch_size
        # ),
        # "Label distortion 0.05 sigma": lambda x: distort_output_predict(model, x, part=part, batch_size=batch_size),
        # "Label distortion 0.1 sigma": lambda x: distort_output_predict(
        #    model, x, amount=0.1, part=part, batch_size=batch_size
        # ),
        # "Label distortion 0.2 sigma": lambda x: distort_output_predict(
        #    model, x, amount=0.2, part=part, batch_size=batch_size
        # ),
        # "Salt and Pepper Noise 0.01": lambda x: salt_and_pepper_noise_predict(
        #    model, x, amount=0.01, part=part, batch_size=batch_size
        # ),
        # "Salt and Pepper Noise 0.02": lambda x: salt_and_pepper_noise_predict(
        #    model, x, amount=0.02, part=part, batch_size=batch_size
        # ),
        # "Salt and Pepper Noise 0.05": lambda x: salt_and_pepper_noise_predict(
        #    model, x, amount=0.05, part=part, batch_size=batch_size
        # ),
        # "Salt and Pepper Noise 0.07": lambda x: salt_and_pepper_noise_predict(
        #    model, x, amount=0.07, part=part, batch_size=batch_size
        # ),
        # "Salt and Pepper Noise 0.09": lambda x: salt_and_pepper_noise_predict(model, x, amount=0.09, part=part, batch_size=batch_size),
        # "Speckle Noise 0.01": lambda x: speckle_noise_predict(model, x, amount=0.01, part=part, batch_size=batch_size),
        # "Speckle Noise 0.02": lambda x: speckle_noise_predict(model, x, amount=0.02, part=part, batch_size=batch_size),
        # "Speckle Noise 0.05": lambda x: speckle_noise_predict(model, x, amount=0.05, part=part, batch_size=batch_size),
        # "Speckle Noise 0.07": lambda x: speckle_noise_predict(model, x, amount=0.07, part=part, batch_size=batch_size),
        # "Speckle Noise 0.09": lambda x: speckle_noise_predict(model, x, amount=0.09, part=part, batch_size=batch_size),
        # "Poisson_Noise 0.01 sigma": lambda x: randomized_smoothing_predict(
        #    model, x, sigma=0.01, noise_type="poisson", part=part, batch_size=batch_size
        # ),
        # "Poisson_Noise 0.02 sigma": lambda x: randomized_smoothing_predict(
        #    model, x, sigma=0.02, noise_type="poisson", part=part, batch_size=batch_size
        # ),
        # "Poisson_Noise 0.03 sigma": lambda x: randomized_smoothing_predict(
        #    model, x, sigma=0.03, noise_type="poisson", part=part, batch_size=batch_size
        # ),
        # "Poisson_Noise 0.04 sigma": lambda x: randomized_smoothing_predict(
        #    model, x, sigma=0.04, noise_type="poisson", part=part, batch_size=batch_size
        # ),
        # "Local Median Smoothing Filter 2x2": lambda x: local_medium_smoothing_predict(
        #    model, x, part=part, batch_size=batch_size
        # ),
        # "Local Median Smoothing Filter 3x3": lambda x: local_medium_smoothing_predict(
        #    model, x, kernel_size=(3, 3, 3), part=part, batch_size=batch_size
        # ),
        # "Color Bit Reduction 4bit": lambda x: color_bit_depth_reduction_predict(
        #    model, x, bit_depth=4, part=part, batch_size=batch_size
        # ),
        # "Color Bit Reduction 2bit": lambda x: color_bit_depth_reduction_predict(
        #    model, x, bit_depth=2, part=part, batch_size=batch_size
        # ),
        "Non-local Mean denoising strength 0.8": lambda x: mean_denoising_predict(
            model, x, strength=0.8, part=part, batch_size=batch_size
        ),
        "Non-local Mean denoising strength 1.5": lambda x: mean_denoising_predict(
            model, x, strength=1.5, part=part, batch_size=batch_size
        ),
        "Non-local Mean denoising strength 3": lambda x: mean_denoising_predict(
            model, x, strength=3, part=part, batch_size=batch_size
        ),
        "Non-local Mean denoising strength 7": lambda x: mean_denoising_predict(
            model, x, strength=7, part=part, batch_size=batch_size
        ),
        "Non-local Mean denoising strength 10": lambda x: mean_denoising_predict(
            model, x, strength=10, part=part, batch_size=batch_size
        ),
        "Non-local Mean denoising strength 12": lambda x: mean_denoising_predict(
            model, x, strength=12, part=part, batch_size=batch_size
        ),
        "Non-local Mean denoising strength 15": lambda x: mean_denoising_predict(
            model, x, strength=15, part=part, batch_size=batch_size
        ),
        # "Smoothing Convolution Filter": lambda x: smoothing_convolution_predict(model, x, filter_type="smooth", batch_size=batch_size),
        # "Sharpen Convolution Filter": lambda x: smoothing_convolution_predict(model, x, filter_type="sharpen", batch_size=batch_size),
        # "Detail Convolution Filter": lambda x: smoothing_convolution_predict(model, x, filter_type="detail", batch_size=batch_size),
        # "Blur Convolution Filter": lambda x: smoothing_convolution_predict(model, x, filter_type="blur", batch_size=batch_size),
        "Defense Autoencoder": lambda x: defense_distillation_autoencoder(model, x, part=part, batch_size=batch_size),
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
        mia_eval_data_y = np.r_[train_y[0:mia_eval_size], test_y[0:mia_eval_size]]
        mia_eval_data_in_out = np.r_[np.ones((mia_eval_size, 1)), np.zeros((mia_eval_size, 1))]
        assert mia_eval_data_x.shape[0] == mia_eval_data_in_out.shape[0]

        # New loss_fn
        loss_fn = compute_loss
        loss_train_vec = loss_fn(train_y[0:mia_eval_size], predict_fn(train_x[0:mia_eval_size]))
        loss_test_vec = loss_fn(test_y[0:mia_eval_size], predict_fn(test_x[0:mia_eval_size]))

        mean_train_loss = np.mean(loss_train_vec)
        std_train_loss = np.std(loss_train_vec)
        mean_test_loss = np.mean(loss_test_vec)
        std_test_loss = np.std(loss_test_vec)

        # so we can add new attack functions as needed
        print("\n------------ Privacy Attacks ----------")
        mia_attack_fns = []
        mia_attack_fns.append(("Simple MIA Attack", simple_conf_threshold_mia))
        mia_attack_fns.append(("Loss attack", do_loss_attack))
        mia_attack_fns.append(("Loss attack2", do_loss_attack2))
        # mia_attack_fns.append(("Shokri et al.", shokri_mia))

        for i, tup in enumerate(mia_attack_fns):
            attack_str, attack_fn = tup
            if attack_fn == simple_conf_threshold_mia:
                in_out_preds = simple_conf_threshold_mia(predict_fn, mia_eval_data_x).reshape(-1, 1)
            elif attack_fn == do_loss_attack:
                in_out_preds = do_loss_attack(
                    mia_eval_data_x,
                    mia_eval_data_y,
                    predict_fn,
                    loss_fn,
                    mean_train_loss,
                    std_train_loss,
                    mean_test_loss,
                    std_test_loss,
                ).reshape(-1, 1)
            elif attack_fn == do_loss_attack2:
                in_out_preds = do_loss_attack2(
                    mia_eval_data_x, mia_eval_data_y, predict_fn, loss_fn, mean_train_loss, std_train_loss
                ).reshape(-1, 1)
            elif attack_fn == shokri_mia:
                in_out_preds = shokri_mia(predict_fn, mia_eval_data_x, mia_eval_data_in_out).reshape(-1, 1)

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
        for attack in sorted(glob(os.path.join("attacks", f"part*test*.npz"))):
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
        data.to_csv(f"{part}_model_results.csv")

    # print consolidated results
    print(data)

    sys.exit(0)
