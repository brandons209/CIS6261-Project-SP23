import os
import numpy as np
from glob import glob

# we'll use tensorflow and keras for neural networks
import tensorflow as tf
import tensorflow.keras as keras

import utils  # we need this
import project  # use this to import other prediction functions


## Plots an adversarial perturbation, i.e., original input orig_x, adversarial example adv_x, and the difference (perturbation)
def plot_adversarial_example(pred_fn, orig_x, adv_x, labels, fname="adv_exp.png", show=True, save=True):
    perturb = adv_x - orig_x

    # compute confidence
    in_label, in_conf = utils.pred_label_and_conf(pred_fn, orig_x)

    # compute confidence
    adv_label, adv_conf = utils.pred_label_and_conf(pred_fn, adv_x)

    titles = [
        "{} (conf: {:.2f})".format(labels[in_label], in_conf),
        "Perturbation",
        "{} (conf: {:.2f})".format(labels[adv_label], adv_conf),
    ]

    images = np.r_[orig_x, perturb, adv_x]

    # plot images
    utils.plot_images(images, fig_size=(8, 3), titles=titles, titles_fontsize=12, out_fp=fname, save=save, show=show)


if __name__ == "__main__":
    model_path = "./target-model.h5"
    attacks_path = "attacks"
    save_path = "examples"
    num_images_plot = 10

    model, _ = utils.load_model(model_path)
    predict_fn = lambda x: project.basic_predict(model, x)
    labels = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    for attack in sorted(glob(os.path.join(attacks_path, "*.npz"))):
        print(f"--> Processing attack {attack}")
        data = np.load(attack, allow_pickle=True)
        adv_x = data["adv_x"]
        benign_x = data["benign_x"]

        idx = np.random.choice(np.arange(len(adv_x)), size=num_images_plot)

        for i in idx:
            plot_adversarial_example(
                predict_fn,
                np.expand_dims(benign_x[i], axis=0),
                np.expand_dims(adv_x[i], axis=0),
                labels,
                fname=os.path.join(save_path, f"{os.path.basename(attack)[:-4]}_example{i}.png"),
                show=False,
            )
