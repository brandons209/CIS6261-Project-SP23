import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import utils

from tqdm import tqdm


######### Adversarial Examples #########
def gradient_of_loss_wrt_input(model, x, y):
    x = tf.convert_to_tensor(x, dtype=tf.float32)  # convert to tensor
    y = tf.convert_to_tensor(y, dtype=tf.float32)  # convert to tensor

    loss_func = tf.keras.losses.CategoricalCrossentropy()
    with tf.GradientTape() as g:
        g.watch(x)
        loss = loss_func(y, model(x))

    return g.gradient(loss, x)


def targeted_gradient_noise(
    model,
    x_input,
    y_input,
    eps: float = 0.1,
    max_iter: int = 5,
    alpha: int = 0.05,
    conf: float = 0.8,
):
    x_in = tf.convert_to_tensor(x_input, dtype=tf.float32)
    y_flat = np.argmax(y_input, axis=-1)
    completed = np.array([False for _ in range(len(x_input))])
    for i in tqdm(range(0, max_iter)):
        idx = np.argwhere(~completed).flatten()

        x_adv = tf.gather(x_in, idx)
        y_input_nc = tf.gather(y_input, idx)

        grad_vec = gradient_of_loss_wrt_input(model, x_adv, y_input_nc)

        # create perturbation
        r = tf.random.uniform(grad_vec.shape)
        perturb = alpha * r * tf.sign(grad_vec)
        perturb = tf.clip_by_value(perturb, -eps, eps)

        # add perturbation
        x_adv = x_adv + perturb
        x_adv = tf.clip_by_value(x_adv, 0, 1.0)

        # update input array with perturbed images
        x_in = x_in.numpy()
        x_in[idx] = x_adv
        x_in = tf.identity(x_in)

        # set the most likely incorrect label as target
        y_pred = model(x_adv).numpy()
        y_pred[y_flat[idx]] = 0
        target_class_number = np.argmax(y_pred, axis=-1)

        # check if predicted label is the target
        adv_label, adv_conf = utils.pred_label_and_conf_model(model, x_adv)

        # check if done
        completed[idx] = (adv_label == target_class_number) & (adv_conf >= conf)
        if completed.all():
            break

    return x_in.numpy(), y_flat


def untargeted_fgsm(
    model,
    x,
    y,
    eps: int = 0.1,
    max_iter: int = 5,
    conf: float = 0.7,
    alpha: float = 0.1,
    method: str = "fgsm",
    decay: float = 1.0,
):
    def do_untargeted_fgsm(in_x, in_y, completed):
        # find non-completed samples to update
        idx = np.argwhere(~completed).flatten()

        in_x_nc = tf.gather(in_x, idx)
        in_y_nc = tf.gather(in_y, idx)

        grad_vec = gradient_of_loss_wrt_input(model, in_x_nc, in_y_nc)

        adv_x = in_x_nc + alpha * tf.sign(grad_vec)
        adv_x = tf.clip_by_value(adv_x, 0, 1.0)

        # update in_x with the changed samples only
        in_x = in_x.numpy()
        in_x[idx] = adv_x

        return tf.identity(in_x)  ## the adversarial example

    def mi_fgsm(in_x, in_y, prev_grad, completed):
        idx = np.argwhere(~completed).flatten()
        full_grad_vec = gradient_of_loss_wrt_input(model, in_x, in_y)

        in_x_nc = tf.gather(in_x, idx)
        if not isinstance(prev_grad, int):
            prev_grad = tf.gather(prev_grad, idx)

        grad_vec = tf.gather(full_grad_vec, idx)

        grad_vec = decay * prev_grad + grad_vec / tf.reduce_mean(tf.abs(grad_vec), [1, 2, 3], keepdims=True)

        adv_x = in_x_nc + alpha / max_iter * tf.sign(grad_vec)
        adv_x = tf.clip_by_value(adv_x, 0, 1.0)

        # update in_x with the changed samples only
        in_x = in_x.numpy()
        in_x[idx] = adv_x

        return tf.identity(in_x), full_grad_vec  ## the adversarial example

    adv_x = tf.convert_to_tensor(x, dtype=tf.float32)
    y_onehot = tf.keras.utils.to_categorical(y, 10)

    minv = np.maximum(x.astype(float) - eps, 0.0)
    maxv = np.minimum(x.astype(float) + eps, 1.0)

    prev_grad = 0
    completed = np.array([False for _ in range(len(x))])
    for i in tqdm(range(0, max_iter)):
        # do one step of FGSM
        if method.lower() == "fgsm":
            adv_x = do_untargeted_fgsm(adv_x, y_onehot, completed)
        elif method.lower() == "mifgsm":
            adv_x, prev_grad = mi_fgsm(adv_x, y_onehot, prev_grad, completed)

        # clip to ensure we stay within an epsilon radius of the input
        adv_x = tf.clip_by_value(adv_x, clip_value_min=minv, clip_value_max=maxv)

        # check if predicted label is the target
        adv_label, adv_conf = utils.pred_label_and_conf_model(model, adv_x)

        # check if done
        completed = (adv_label != y) & (adv_conf >= conf)
        if completed.all():
            break

    return adv_x.numpy()


def craft_adversarial_fgsmk(
    model,
    x_aux,
    y_aux,
    num_adv_samples,
    eps,
    alpha: float = 0.01,
    method: str = "fgsm",
    decay: float = 1.0,
):
    idx = np.random.choice(np.arange(len(x)), size=num_adv_samples)

    x_input = x_aux[idx]
    y_input = y_aux[idx]

    # keep track of the benign examples
    x_benign_samples = x_input.copy()

    correct_labels = np.argmax(y_input, axis=-1)

    if method == "fgsm":
        x_adv_samples = untargeted_fgsm(
            model,
            x_input,
            correct_labels,
            max_iter=100,
            eps=eps,
            alpha=alpha,
            method=method,
            decay=decay,
        )
    elif method == "mifgsm":
        x_adv_samples = untargeted_fgsm(
            model,
            x_input,
            correct_labels,
            max_iter=100,
            eps=eps,
            alpha=alpha,
            method=method,
            decay=decay,
        )

    return x_benign_samples, correct_labels, x_adv_samples


if __name__ == "__main__":
    num_samples = 1000
    alpha_values = [0.001, 0.005, 0.01, 0.025, 0.05, 0.075]
    decay_values = [
        0.001,
        0.01,
        0.1,
        0.5,
        1.0,
        1.5,
        2.0,
    ]
    part = "part1"
    model_path = "./target-model.h5"

    model, _ = utils.load_model(model_path)

    if part == "part1":
        train_x, train_y, test_x, test_y, val_x, val_y, labels = utils.load_data()
        x = np.concatenate([train_x, test_x, val_x])
        y = np.concatenate([train_y, test_y, val_y])
    elif part == "part2":
        (train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()
        x = np.concatenate([train_x, test_x]).astype(float) / 255
        y = np.concatenate([train_y, test_y])

    print(x.shape)
    print(y.shape)
    print(np.max(x))

    print(f"Generating {num_samples} adversial examples per attack for {part}.")

    print("--> Starting targeted gradient attack...")

    for a in alpha_values:
        idx = np.random.choice(np.arange(len(x)), size=num_samples)
        # don't recreate if it already exists
        if os.path.isfile(os.path.join("attacks", f"adv2_gradient_attack_alpha_{a}.npz")):
            continue

        x_benign = x[idx]
        x_adv, correct_labels = targeted_gradient_noise(
            model,
            x[idx],
            y[idx],
            max_iter=100,
            alpha=a,
            eps=0.1,
            conf=0.7,
        )

        if part == "part2":
            # if using part2, input needs to be unnormalized
            x_benign *= 255
            x_adv *= 255

        np.savez(
            os.path.join("attacks", f"adv2_gradient_attack_alpha_{a}.npz"),
            benign_x=x_benign,
            benign_y=correct_labels,
            adv_x=x_adv,
        )

        print(f"\t--> Finished targeted gradient attack. Saved to attacks/adv2_gradient_attack_alpha_{a}.npz")

    print("\n--> Starting untargeted FGSM attack...")
    for a in alpha_values:
        if not os.path.isfile(os.path.join("attacks", f"adv3_fgsm_alpha_{a}.npz")):
            print(f"\t--> Performing FGSM with alpha value {a}")
            x_benign, correct_labels, x_adv = craft_adversarial_fgsmk(
                model,
                x,
                y,
                num_samples,
                eps=0.1,
                alpha=a,
            )

            if part == "part2":
                # if using part2, input needs to be unnormalized
                x_benign *= 255
                x_adv *= 255

            np.savez(
                os.path.join("attacks", f"adv3_fgsm_alpha_{a}.npz"),
                benign_x=x_benign,
                benign_y=correct_labels,
                adv_x=x_adv,
            )

            print(f"\t-->Finished untargeted FGSM attack. Saved to attacks/adv3_fgsm_alpha_{a}.npz")

        print(f"\t--> Starting MI-FGSM with alpha value {a}")
        for d in decay_values:
            if os.path.isfile(os.path.join("attacks", f"adv3_mifgsm_alpha_{a}_decay_{d}.npz")):
                continue
            print(f"\t\t--> Performing MI FGSM with alpha value {a} and decay value {d}")
            x_benign, correct_labels, x_adv = craft_adversarial_fgsmk(
                model,
                x,
                y,
                num_samples,
                eps=0.1,
                alpha=a,
                method="mifgsm",
                decay=d,
            )

            if part == "part2":
                # if using part2, input needs to be unnormalized
                x_benign *= 255
                x_adv *= 255

            np.savez(
                os.path.join("attacks", f"adv3_mifgsm_alpha_{a}_decay_{d}.npz"),
                benign_x=x_benign,
                benign_y=correct_labels,
                adv_x=x_adv,
            )

            print(f"\t\t-->Finished MI-FGSM attack. Saved to attacks/adv3_mifgsm_alpha_{a}_decay_{d}.npz")

        print("\t-->Finished MI-FGSM attack.")