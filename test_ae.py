import tensorflow.keras as keras
import numpy as np
import utils  # we need this


model_path = "ae_defense_best.h5"
model, _ = utils.load_model(model_path)

(train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()

### for autoencoder only
train_x = train_x.astype(float) / 255
test_x = test_x.astype(float) / 255

train_y = train_x
test_y = test_x
###

print(model.evaluate(x=test_x, y=test_y))


# show examples of autoencoder recreation
input_images = test_x[:8]
plot_images = []
titles = []
for i, im in enumerate(input_images):
    im = np.expand_dims(im, axis=0)
    out = model(im)
    plot_images += [im, out]
    titles += [f"Input Image {i + 1}", f"Recreated Image {i + 1}"]

plot_images = np.vstack(plot_images)

utils.plot_images(plot_images, titles=titles)
