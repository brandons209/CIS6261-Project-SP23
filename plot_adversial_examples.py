import os
import numpy as np

from glob import glob
from keras.applications.convnext import LayerScale

import utils  # we need this
import project  # use this to import other prediction functions
import matplotlib

matplotlib.use("Agg")


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
    model_path = "./part2_model_best.h5"  # "./target-model.h5"
    part = "part2"
    attacks_path = "attacks"
    save_path = "examples"
    num_images_plot = 10

    model, _ = utils.load_model(model_path, custom_objects={"LayerScale": LayerScale})
    predict_fn = lambda x: project.basic_predict(model, x, part=part)
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

    example_benign = None

    for attack in sorted(glob(os.path.join(attacks_path, "part*test*.npz"))):
        print(f"--> Processing attack {attack}")
        data = np.load(attack, allow_pickle=True)
        adv_x = data["adv_x"]
        benign_x = data["benign_x"]

        example_benign = benign_x.copy()

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

    # also plot examples of defense functions on benign images
    predict_fns = {
        "Randomized Gaussian 0.05 sigma": lambda x: project.randomized_smoothing_predict(
            model, x, sigma=0.05, noise_type="Gaussian", part=part
        ),
        "Randomized Laplace 0.05 sigma": lambda x: project.randomized_smoothing_predict(
            model, x, sigma=0.05, noise_type="Laplace", part=part
        ),
        "Randomized Gaussian 0.1 sigma": lambda x: project.randomized_smoothing_predict(
            model, x, sigma=0.1, noise_type="Gaussian", part=part
        ),
        "Randomized Laplace 0.1 sigma": lambda x: project.randomized_smoothing_predict(
            model, x, sigma=0.1, noise_type="Laplace", part=part
        ),
        "Randomized Gaussian 0.2 sigma": lambda x: project.randomized_smoothing_predict(
            model, x, sigma=0.2, noise_type="Gaussian", part=part
        ),
        "Randomized Laplace 0.2 sigma": lambda x: project.randomized_smoothing_predict(
            model, x, sigma=0.2, noise_type="Laplace", part=part
        ),
        "Label distortion 0.05 sigma": lambda x: project.distort_output_predict(model, x, part=part),
        "Label distortion 0.1 sigma": lambda x: project.distort_output_predict(model, x, amount=0.1, part=part),
        "Label distortion 0.2 sigma": lambda x: project.distort_output_predict(model, x, amount=0.2, part=part),
        "Salt and Pepper Noise 0.01": lambda x: project.salt_and_pepper_noise_predict(model, x, amount=0.01, part=part),
        "Salt and Pepper Noise 0.02": lambda x: project.salt_and_pepper_noise_predict(model, x, amount=0.02, part=part),
        "Salt and Pepper Noise 0.05": lambda x: project.salt_and_pepper_noise_predict(model, x, amount=0.05, part=part),
        "Salt and Pepper Noise 0.07": lambda x: project.salt_and_pepper_noise_predict(model, x, amount=0.07, part=part),
        "Salt and Pepper Noise 0.09": lambda x: project.salt_and_pepper_noise_predict(model, x, amount=0.09, part=part),
        "Speckle Noise 0.01": lambda x: project.speckle_noise_predict(model, x, amount=0.01, part=part),
        "Speckle Noise 0.02": lambda x: project.speckle_noise_predict(model, x, amount=0.02, part=part),
        "Speckle Noise 0.05": lambda x: project.speckle_noise_predict(model, x, amount=0.05, part=part),
        "Speckle Noise 0.07": lambda x: project.speckle_noise_predict(model, x, amount=0.07, part=part),
        "Speckle Noise 0.09": lambda x: project.speckle_noise_predict(model, x, amount=0.09, part=part),
        "Poisson_Noise 0.01 sigma": lambda x: project.randomized_smoothing_predict(
            model, x, sigma=0.01, noise_type="poisson", part=part
        ),
        "Poisson_Noise 0.02 sigma": lambda x: project.randomized_smoothing_predict(
            model, x, sigma=0.02, noise_type="poisson", part=part
        ),
        "Poisson_Noise 0.03 sigma": lambda x: project.randomized_smoothing_predict(
            model, x, sigma=0.03, noise_type="poisson", part=part
        ),
        "Poisson_Noise 0.04 sigma": lambda x: project.randomized_smoothing_predict(
            model, x, sigma=0.04, noise_type="poisson", part=part
        ),
        "Local Median Smoothing Filter 2x2": lambda x: project.local_medium_smoothing_predict(model, x, part=part),
        "Local Median Smoothing Filter 3x3": lambda x: project.local_medium_smoothing_predict(
            model, x, kernel_size=(3, 3, 3), part=part
        ),
        "Color Bit Reduction 4bit": lambda x: project.color_bit_depth_reduction_predict(
            model, x, bit_depth=4, part=part
        ),
        "Color Bit Reduction 2bit": lambda x: project.color_bit_depth_reduction_predict(
            model, x, bit_depth=2, part=part
        ),
        "Non-local Mean denoising strength 0.8": lambda x: project.mean_denoising_predict(
            model, x, strength=0.8, part=part
        ),
        "Non-local Mean denoising strength 1.5": lambda x: project.mean_denoising_predict(
            model, x, strength=1.5, part=part
        ),
        "Non-local Mean denoising strength 3": lambda x: project.mean_denoising_predict(
            model, x, strength=3, part=part
        ),
        "Non-local Mean denoising strength 7": lambda x: project.mean_denoising_predict(
            model, x, strength=7, part=part
        ),
        "Non-local Mean denoising strength 10": lambda x: project.mean_denoising_predict(
            model, x, strength=10, part=part
        ),
        "Non-local Mean denoising strength 12": lambda x: project.mean_denoising_predict(
            model, x, strength=12, part=part
        ),
        "Non-local Mean denoising strength 15": lambda x: project.mean_denoising_predict(
            model, x, strength=15, part=part
        ),
        "Smoothing Convolution Filter": lambda x: project.moothing_convolution_predict(
            model, x, filter_type="smooth", part=part
        ),
        "Sharpen Convolution Filter": lambda x: project.smoothing_convolution_predict(
            model, x, filter_type="sharpen", part=part
        ),
        "Detail Convolution Filter": lambda x: project.smoothing_convolution_predict(
            model, x, filter_type="detail", part=part
        ),
        "Blur Convolution Filter": lambda x: project.smoothing_convolution_predict(
            model, x, filter_type="blur", part=part
        ),
        "Defense Autoencoder": lambda x: project.defense_distillation_autoencoder(model, x, part=part),
    }

    for name, defense in predict_fns.items():
        print(f"--> Processing defense {name}")

        adv_x = defense(example_benign)
        benign_x = example_benign

        idx = np.random.choice(np.arange(len(adv_x)), size=num_images_plot)

        for i in idx:
            plot_adversarial_example(
                predict_fn,  # basic prediction of model w/o defense
                np.expand_dims(benign_x[i], axis=0),
                np.expand_dims(adv_x[i], axis=0),
                labels,
                fname=os.path.join(save_path, f"{name}_example{i}.png"),
                show=False,
            )
