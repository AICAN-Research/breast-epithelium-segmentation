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
from augment import random_brightness, random_rot90, random_fliplr, random_flipud, \
    random_hue, random_saturation, random_shift
from utils import normalize_img, patchReader
from argparse import ArgumentParser
import sys


parser = ArgumentParser()
parser.add_argument('--batch_size', metavar='--bs', type=int, nargs='?', default=8,
                    help="set which batch size to use for training.")
parser.add_argument('--learning_rate', metavar='--lr', type=float, nargs='?', default=0.001,
                    help="set which learning rate to use for training.")
parser.add_argument('--epochs', metavar='--ep', type=int, nargs='?', default=500,
                    help="number of epochs to train.")
parser.add_argument('--patience', metavar='--pa', type=int, nargs='?', default=10,
                    help="number of epochs to wait (patience) for early stopping.")
ret = parser.parse_known_args(sys.argv[1:])[0]

print(ret)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # due to this: https://github.com/tensorflow/tensorflow/issues/35029

curr_date = "".join(date.today().strftime("%d/%m").split("/")) + date.today().strftime("%Y")[2:]
curr_time = "".join(str(datetime.now()).split(" ")[1].split(".")[0].split(":"))

# disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# bs = 16
lr = 1e-4
img_size = 512
nb_classes = 2
epochs = 100

name = curr_date + "_" + curr_time + "_" + "unet_bs_" + str(ret.batch_size) # + "_eps_" + str(ret.epochs)

# paths
dataset_path = './datasets/TMASegmentation070922_level_0_psize_512/'  # path to directory
history_path = './output/history/'  # path to directory
model_path = './output/models/'  # path to directory
save_ds_path = './output/datasets/dataset_' + name + '/' #inni her først en med name, så ds_train og test inni der

patches = os.listdir(dataset_path)
paths = np.array([dataset_path + x for x in patches]).astype("U400")  # make list of elements in


ds_all = tf.data.Dataset.from_tensor_slices(paths)  # list of paths to tensor in data.Dataset format
ds_all = ds_all.map(lambda x: tf.py_function(patchReader, [x], [tf.float32, tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)

#ds_train = ds_train.cache()

# create test set
N = len(patches)
N_train = int(np.round(N * 0.8))
N_test = N - N_train
ds_train = ds_all.take(N_train)
ds_test = ds_all.skip(N_train)



"""
# hsv augmentation next
#ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)  # remove this when not testing
image, mask  = next(iter(ds_train))
image = np.asarray(image)
image = image.astype("uint8")

f, axes = plt.subplots(1, 2)  # Figure of the two corresponding TMAs
axes[0].imshow(image)
axes[1].imshow(mask[:,:,1], cmap="gray")
plt.show()

#image_aug = random_saturation(image, saturation=0.5)
#image_aug = np.asarray(image_aug)
mask = np.asarray(mask)
#image, mask = random_shift(image, mask)  # how does this work, or does it? It is a keras layer? does it just work during training?
image, mask = random_rot90(image, mask)
image = np.asarray(image)
mask = np.asarray(mask)
f, axes = plt.subplots(1, 3)  # Figure of the two corresponding TMAs
axes[0].imshow(image.astype("uint8"))
axes[1].imshow(mask[:, :, 0].astype("uint8"), cmap="gray")
axes[2].imshow(mask[:, :, 1].astype("uint8"), cmap="gray")
plt.show()

exit()
"""

os.makedirs(save_ds_path, exist_ok=True)  # check if exist, then create, otherwise not

tf.data.experimental.save(
   ds_train, save_ds_path + 'ds_train', compression=None, shard_func=None
)

tf.data.experimental.save(
    ds_test, save_ds_path + 'ds_test', compression=None, shard_func=None
)

ds_train = ds_train.shuffle(buffer_size=4)
ds_train = ds_train.batch(ret.batch_size)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
ds_train = ds_train.repeat(-1)

ds_test = ds_test.shuffle(buffer_size=4)
ds_test = ds_test.batch(ret.batch_size)
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.repeat(-1)

# only augment train data
# shift last
ds_train = ds_train.map(lambda x, y: random_fliplr(x, y))
ds_train = ds_train.map(lambda x, y: random_flipud(x, y))
ds_train = ds_train.map(lambda x, y: (random_brightness(x, brightness=25), y))
ds_train = ds_train.map(lambda x, y: (random_hue(x, max_delta=0.1), y))  # look at best value for max_delta
ds_train = ds_train.map(lambda x, y: (random_saturation(x, saturation=0.5), y))  # look at best value for saturation
ds_train = ds_train.map(lambda x, y: random_shift(x, y, translate=50))
# shift last

# normalize intensities
ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

convs = [8, 16, 32, 64, 64, 128, 128, 256]  # 128, 128, 64, 64, 32, 16, 8
convs = convs + convs[:-1][::-1]

network = Unet(input_shape=(img_size, img_size, 3), nb_classes=nb_classes)  # binary = 2
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

history = model.fit(
    ds_train,
    steps_per_epoch=N_train // ret.batch_size,
    epochs=epochs,
    validation_data=ds_test,
    validation_steps=N_test // ret.batch_size,
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
