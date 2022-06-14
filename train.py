import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import h5py
import tensorflow_datasets as tfds
from deep_learning_tools.network import Unet
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from datetime import datetime, date
from tensorflow.keras.models import load_model



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

curr_date = "".join(date.today().strftime("%d/%m").split("/")) + date.today().strftime("%Y")[2:]
curr_time = "".join(str(datetime.now()).split(" ")[1].split(".")[0].split(":"))

# disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

bs = 4
N_train = 50
lr = 1e-4

name = curr_date + "_" + curr_time + "_" + "unet"

# paths
dataset_path = './datasets/TMASegmentation020622/'  # path to directory
history_path = './output/history/'  # path to directory
model_path = './output/models/'  # path to directory
save_ds_path = './output/datasets/dataset_' + name + '/' #inni her først en med name, så ds_train og test inni der

paths = np.array([dataset_path + x for x in os.listdir(dataset_path)]).astype("U400")  # make list of elements in


ds_all = tf.data.Dataset.from_tensor_slices(paths)  # list of paths to tensor in data.Dataset format
ds_all = ds_all.map(lambda x: tf.py_function(patchReader, [x], [tf.float32, tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)
ds_all = ds_all.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

#ds_train = ds_train.cache()

# create test set
N = 68
ds_train = ds_all.take(50)
ds_test = ds_all.skip(50)

os.makedirs(save_ds_path, exist_ok=True)  # check if exist, then create, otherwise not

tf.data.experimental.save(
   ds_train, save_ds_path + 'ds_train', compression=None, shard_func=None
)

tf.data.experimental.save(
    ds_test, save_ds_path + 'ds_test', compression=None, shard_func=None
)

ds_train = ds_train.shuffle(buffer_size=4)
ds_train = ds_train.batch(bs)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
ds_train = ds_train.repeat(-1)

ds_test = ds_test.shuffle(buffer_size=4)
ds_test = ds_test.batch(bs)
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.repeat(-1)

convs = [8, 16, 32, 64, 64, 128, 128, 256]  # 128, 128, 64, 64, 32, 16, 8
convs = convs + convs[:-1][::-1]

network = Unet(input_shape=(512, 512, 3), nb_classes=2)  # binary = 2
network.set_convolutions(convs)
model = network.create()

#print(model.summary())

history = CSVLogger(
    history_path + "history_" + name + ".csv",
    append=True
)

save_best = ModelCheckpoint(
            model_path + "model_" + name,
            monitor="val_loss",
            verbose=2,  #
            save_best_only=True,
            save_weights_only=False,
            mode="min",  # use "auto" with "f1_score", "auto" with "val_loss" (or "min")
            save_freq="epoch"
        )

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr),
    loss=network.get_dice_loss(),
    # metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

epochs = 100

history = model.fit(
    ds_train,
    steps_per_epoch=N_train // bs,
    epochs=epochs,
    validation_data=ds_test,
    validation_steps=N_train // bs,
    callbacks=[save_best, history],
    verbose=1,
)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, epochs + 1)
plt.plot(epochs, loss, '-', label = 'Training loss')
plt.plot(epochs, val_loss, '-', label = 'Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# load best model from disk
del model
model = load_model(model_path + "model_" + name, compile=False)

print(model.summary())

# Predict on new data, not the case here:
# loop over all elements in test set, run model prediction, save pred -> after loop, compute metrics (Dice)
# from tensorflow tutorial:
