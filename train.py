import numpy as np
import tensorflow as tf
import os
import h5py
import tensorflow_datasets as tfds
from deep_learning_tools.network import Unet


# from tensorflow example, modified
def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


# Get image and gt from hdf5
def patchReader(path):
    path = tfds.as_numpy(path).decode("utf-8")

    with h5py.File(path, "r") as f:
        image = np.asarray(f["input"]).astype("float32")
        gt = np.asarray(f["output"]).astype("float32")
    return image, gt


# disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

bs = 2


dataset_path = './datasets/TMASegmentation020622/'  # path to directory
paths = np.array([dataset_path + x for x in os.listdir(dataset_path)]).astype("U400")  # make list of elements in


ds_all = tf.data.Dataset.from_tensor_slices(paths)  # list of paths to tensor in data.Dataset format

ds_all = ds_all.map(lambda x: tf.py_function(patchReader, [x], [tf.float32, tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)

ds_all = ds_all.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

#ds_train = ds_train.cache()

# create test set
N = 68
ds_train = ds_all.take(50)
ds_test = ds_all.skip(50)

ds_train = ds_train.shuffle(buffer_size=4)
ds_train = ds_train.batch(bs)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
ds_train = ds_train.repeat(-1)

ds_test = ds_test.shuffle(buffer_size=4)
ds_test = ds_test.batch(bs)
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.repeat(-1)

network = Unet(input_shape=(512, 512, 3), nb_classes=2)  # binary = 2
model = network.create()

#print(model.summary())

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=network.get_dice_loss(),
    # metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    ds_train,
    steps_per_epoch=50 // bs,
    epochs=6,
    validation_data=ds_test,
    validation_steps=18 // bs,
)


