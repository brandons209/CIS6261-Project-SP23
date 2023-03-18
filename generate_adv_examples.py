import sys
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import utils


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
    max_iter: int = 20,
    alpha: int = 0.05,
    conf: float = 0.8,
):
    x_in = tf.convert_to_tensor(x_input, dtype=tf.float32)
    x_adv = x_in  # initial adversarial example
    y_flat = np.argmax(y_input, axis=-1)

    for i in range(0, max_iter):
        grad_vec = gradient_of_loss_wrt_input(model, x_adv, y_input)

        # create perturbation
        r = tf.random.uniform(grad_vec.shape)
        perturb = alpha * r * tf.sign(grad_vec)

        perturb = tf.clip_by_value(perturb, -eps, eps)

        # add perturbation
        x_adv = x_adv + perturb

        x_adv = tf.clip_by_value(x_adv, 0, 1.0)

        # set the most likely incorrect label as target
        y_pred = model(x_adv)[0].numpy()
        y_pred[y_flat] = 0
        target_class_number = np.argmax(y_pred, axis=-1)

        # check if we should stop the attack early
        y_pred_v = model.predict(x_adv, verbose=0)[0]
        y_pred = np.argmax(y_pred_v, axis=-1)

        if y_pred == target_class_number and y_pred_v[y_pred] >= conf:
            break

    return x_adv.numpy().astype(int), y_flat


def untargeted_fgsm(
    model,
    x,
    y,
    eps: int = 0.1,
    max_iter: int = 100,
    conf: float = 0.8,
    alpha: float = 1.0,
    method: str = "fgsm",
    decay: float = 1.0,
):
    def do_untargeted_fgsm(in_x, in_y):
        grad_vec = gradient_of_loss_wrt_input(model, in_x, in_y)

        adv_x = in_x + alpha * tf.sign(grad_vec)
        adv_x = tf.clip_by_value(adv_x, 0, 1.0)

        return adv_x  ## the adversarial example

    def mi_fgsm(in_x, in_y, prev_grad):
        grad_vec = gradient_of_loss_wrt_input(model, in_x, in_y)

        grad_vec = decay * prev_grad + grad_vec / tf.reduce_mean(tf.abs(grad_vec), [1, 2, 3], keepdims=True)

        adv_x = in_x + alpha / max_iter * tf.sign(grad_vec)
        adv_x = tf.clip_by_value(adv_x, 0, 1.0)

        return adv_x, grad_vec  ## the adversarial example

    adv_x = tf.convert_to_tensor(np.expand_dims(x, axis=0), dtype=tf.float32)
    y_onehot = tf.keras.utils.to_categorical(np.expand_dims(y, axis=0), 10)

    minv = np.maximum(x.astype(float) - eps, 0.0)
    maxv = np.minimum(x.astype(float) + eps, 1.0)

    i = 0
    prev_grad = 0
    while True:
        # do one step of FGSM
        if method.lower() == "fgsm":
            adv_x = do_untargeted_fgsm(adv_x, y_onehot)
        elif method.lower() == "mifgsm":
            adv_x, prev_grad = mi_fgsm(adv_x, y_onehot, prev_grad)

        # clip to ensure we stay within an epsilon radius of the input
        adv_x = tf.clip_by_value(adv_x, clip_value_min=minv, clip_value_max=maxv)

        # check if predicted label is the target
        adv_label, adv_conf = utils.pred_label_and_conf_model(model, adv_x)

        i += 1

        # check if done
        if i >= max_iter or (adv_label != y and adv_conf >= conf):
            break

    return adv_x.numpy().astype(int)


def craft_adversarial_fgsmk(
    model,
    x_aux,
    y_aux,
    num_adv_samples,
    eps,
    alpha: float = 1.0,
    method: str = "fgsm",
    decay: float = 1.0,
):
    x_adv_samples = np.zeros((num_adv_samples, x_aux.shape[1], x_aux.shape[2], x_aux.shape[3]))
    x_benign_samples = np.zeros((num_adv_samples, x_aux.shape[1], x_aux.shape[2], x_aux.shape[3]))
    correct_labels = np.zeros((num_adv_samples,))

    # sys.stdout.write("Crafting {} adversarial examples (untargeted FGSMk -- eps: {})".format(num_adv_samples, eps))
    for i in range(0, num_adv_samples):
        ridx = np.random.randint(low=0, high=x_aux.shape[0])

        x_input = x_aux[ridx]
        y_input = y_aux[ridx]

        # keep track of the benign examples
        x_benign_samples[i] = x_input

        correct_labels[i] = np.argmax(y_input)

        if method == "fgsm":
            x_adv = untargeted_fgsm(
                model,
                x_input,
                correct_labels[i],
                max_iter=100,
                eps=eps,
                alpha=alpha,
                method=method,
                decay=decay,
            )
        elif method == "mifgsm":
            x_adv = untargeted_fgsm(
                model,
                x_input,
                correct_labels[i],
                max_iter=10,
                eps=eps,
                alpha=alpha,
                method=method,
                decay=decay,
            )

        x_adv_samples[i] = x_adv

        if i % 100 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
    print("Done.")

    return x_benign_samples, correct_labels, x_adv_samples


if __name__ == "__main__":
    num_samples = 100
    alpha_values = [0.01, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    eps_values = [0.01, 0.05, 0.06, 0.08, 0.1, 0.2]
    decay_values = [
        0.01,
        0.05,
        0.08,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
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

        x_benign = x[idx]
        x_adv, correct_labels = targeted_gradient_noise(
            model,
            x[idx],
            y[idx],
            alpha=a,
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

        print(f"\t--> Finished targeted gradient attack. Saved to attacks/adv2_gradient_attack_alpha_{a}.npz\n")

    print("--> Starting untargeted FGSM attack...")
    for a in alpha_values:
        print(f"\t--> Performing FGSM with alpha value {a}")
        x_benign, correct_labels, x_adv = craft_adversarial_fgsmk(model, x, y, num_samples, eps=0.1, alpha=a)

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
