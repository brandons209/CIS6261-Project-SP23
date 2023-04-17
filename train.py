import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# we'll use tensorflow and keras for neural networks
import tensorflow as tf
import tensorflow.keras as keras
from keras import layers
from glob import glob

import utils  # we need this
import os
import json


# from https://keras.io/examples/generative/vae/
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# from https://keras.io/examples/generative/vae/
class VAE(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # encoder
        encoder_inputs = keras.Input(shape=(32, 32, 3))
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)
        z_mean = layers.Dense(16, name="z_mean")(x)
        z_log_var = layers.Dense(16, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()

        # decoder
        latent_inputs = keras.Input(shape=(16,))
        x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((7, 7, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()

        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def get_model(part: str, model_path: str = None):
    if part == "part1":
        target_model_fp = "./target-model.h5"
        model, _ = utils.load_model(target_model_fp)
        return model
    elif part == "part2":
        if model_path:
            model, _ = utils.load_model(model_path)
            return model

        # setup model for CIFAR-10 classification
        model = keras.applications.ConvNeXtSmall(include_top=False, input_shape=(32, 32, 3), pooling="avg")
        # ConvNeXt models expect their inputs to be float or uint8 tensors of pixels with values in the [0-255] range.

        # add in top layers for classification
        fc = layers.Dense(512, activation="relu")(model.output)
        fc = layers.Dense(256, activation="relu")(fc)
        fc = layers.Dense(128, activation="relu")(fc)
        fc = layers.Dense(10, activation="softmax")(fc)

        model = keras.Model(inputs=model.input, outputs=fc, name="part2_model")

        # compile model for training/testing
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        return model
    elif part == "ae":
        input = layers.Input(shape=(32, 32, 3))
        # Encoder
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
        x = layers.MaxPooling2D((2, 2), padding="same")(x)
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2), padding="same")(x)

        # Decoder
        x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
        x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
        x = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

        # Autoencoder
        model = keras.Model(input, x, name="ae_defense")
        model.compile(optimizer="adam", loss="binary_crossentropy")

        return model
    elif part == "vae":
        vae = VAE()
        vae.compile(optimizer=keras.optimizers.Adam())
        return vae
    else:
        return None


def train_model(model, train_data: tuple, validation_data: tuple, test_data: tuple):
    batch_size = 32
    epochs = 100
    checkpointer = keras.callbacks.ModelCheckpoint(
        f"{model.name}_best.h5", verbose=1, save_best_only=True, monitor="val_loss"
    )
    earlystop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, verbose=1)

    history = model.fit(
        train_data[0],
        train_data[1],
        batch_size=batch_size,
        callbacks=[checkpointer, earlystop],
        validation_data=validation_data,
        epochs=epochs,
        workers=4,
        use_multiprocessing=True,
        verbose=1,
    )

    print("Testing data results:")
    results = model.evaluate(x=test_data[0], y=test_data[1])

    print(results)

    return history


if __name__ == "__main__":
    part = "ae_finetune"
    model = get_model(part)

    if part == "part1":
        train_x, train_y, test_x, test_y, val_x, val_y, labels = utils.load_data()
    elif part == "part2":
        (train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()
        # split testing data into validation and testing
        test_x, val_x, test_y, val_y = train_test_split(test_x, test_y, test_size=0.5)
    elif part == "ae" or part == "vae":
        (train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()
        # convert to values in [0, 1]
        train_x = train_x.astype(float) / 255
        test_x = test_x.astype(float) / 255

        train_y = train_x
        test_y = test_x

        # split testing data into validation and testing
        test_x, val_x, test_y, val_y = train_test_split(test_x, test_y, test_size=0.5)

    elif part == "ae_finetune" or "vae_finetune":
        # fine tune on attacked images
        attacks = sorted(glob(os.path.join("attacks", "*.npz")))

        x, y = [], []
        for attack in attacks:
            data = np.load(attack, allow_pickle=True)
            x.append(data["adv_x"])
            y.append(data["benign_x"])

        x = np.vstack(x)
        y = np.vstack(y)

        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
        test_x, val_x, test_y, val_y = train_test_split(test_x, test_y, test_size=0.5)

    print(model.summary())
    train_model(model, (train_x, train_y), (val_x, val_y), (test_x, test_y))
