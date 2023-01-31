import tensorflow as tf
import os
from deep_learning_tools.network import Unet
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from datetime import datetime, date
from augment import random_brightness, random_fliplr, random_flipud, \
    random_hue, random_saturation, random_shift, random_blur
from source.utils import normalize_img, patchReader, get_random_path_from_random_class
from argparse import ArgumentParser
import sys

parser = ArgumentParser()
parser.add_argument('--batch_size', metavar='--bs', type=int, nargs='?', default=16,
                    help="set which batch size to use for training.")
parser.add_argument('--learning_rate', metavar='--lr', type=float, nargs='?', default=0.0005,
                    help="set which learning rate to use for training.")
parser.add_argument('--epochs', metavar='--ep', type=int, nargs='?', default=500,
                    help="number of epochs to train.")
parser.add_argument('--patience', metavar='--pa', type=int, nargs='?', default=10,
                    help="number of epochs to wait (patience) for early stopping.")
parser.add_argument('--proc', metavar='--pr', type=int, nargs='?', default=4,
                    help="number of workers to use with tf.data.")
ret = parser.parse_known_args(sys.argv[1:])[0]

print(ret)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # due to this: https://github.com/tensorflow/tensorflow/issues/35029

curr_date = "".join(date.today().strftime("%d/%m").split("/")) + date.today().strftime("%Y")[2:]
curr_time = "".join(str(datetime.now()).split(" ")[1].split(".")[0].split(":"))

# disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

img_size = 512
nb_classes = 4

name = curr_date + "_" + curr_time + "_" + "unet_bs_" + str(ret.batch_size)  # + "_eps_" + str(ret.epochs)


# paths
dataset_path = './datasets/310123_130125_level_2_psize_512_ds_4/'  # path to directory
train_path = dataset_path + 'ds_train'
val_path = dataset_path + 'ds_val'
#test_path = dataset_path + 'ds_test'
history_path = './output/history/'  # path to directory
model_path = './output/models/'  # path to directory
save_ds_path = './output/datasets/dataset_' + name + '/'  # inni her først en med name, så ds_train og test inni der

N_train_tot = 1000  # @TODO: Change this number
N_val_tot = 200

# Cross-validation for division into train, val, test:
# The numbers corresponds to wsi-numbers created in create data
"""
k = 5  # number of folds in cross-validation
nbr_files = len(os.listdir(dataset_path))  # number of slides in total dataset (24 including zero, 0-23)
nbr_val = int(np.floor(nbr_files / k))
nbr_test = int(np.floor(nbr_files / k))
print("nbr_val", nbr_val)
print("nbr_test", nbr_test)

order = np.arange(nbr_files)
np.random.shuffle(order)
"""
# --------------------

train_paths = []
for directory in os.listdir(train_path):
    dir_path = train_path + "/" + directory + "/"
    dir_paths = []
    for file_ in os.listdir(dir_path):
        file_path = dir_path + file_
        dir_paths.append(file_path)
    train_paths.append(dir_paths)  # nested list of three lists containing paths for each folder/class
    
val_paths = []
for directory in os.listdir(val_path):
    dir_path = train_path + "/" + directory + "/"
    dir_paths = []
    for file_ in os.listdir(dir_path):
        file_path = dir_path + file_
        dir_paths.append(file_path)
    val_paths.append(dir_paths)  # nested list of three lists containing paths for each folder/class
    

# combine all train/val paths
ds_train = tf.data.Dataset.from_generator(
    get_random_path_from_random_class,
    output_shapes=tf.TensorShape([]),
    output_types=tf.string,
    args=train_paths
)

ds_val = tf.data.Dataset.from_generator(
    get_random_path_from_random_class,
    output_shapes=tf.TensorShape([]),
    output_types=tf.string,
    args=val_paths
)

# load patch from randomly selected patch
ds_train = ds_train.map(lambda x: tf.py_function(patchReader, [x], [tf.float32, tf.float32]), num_parallel_calls=ret.proc)
ds_val = ds_val.map(lambda x: tf.py_function(patchReader, [x], [tf.float32, tf.float32]), num_parallel_calls=ret.proc)


#ds_train = ds_train.shuffle(buffer_size=N_train)  # is this correct, do I need to "reshuffle_each_iteration"?
ds_train = ds_train.batch(ret.batch_size)
ds_train = ds_train.prefetch(1)
#ds_train = ds_train.repeat(-1)  # repeat indefinitely (?)

#ds_val = ds_val.shuffle(buffer_size=N_val)  # is this correct, do I need to "reshuffle_each_iteration"?
ds_val = ds_val.batch(ret.batch_size)
ds_val = ds_val.prefetch(1)
#ds_val = ds_val.repeat(-1)  # repeat indefinitely (?)  # TODO: Remove! no longer needed as we have infinite generator


# normalize intensities
ds_train = ds_train.map(normalize_img)  # , num_parallel_calls=tf.data.AUTOTUNE)
ds_val = ds_val.map(normalize_img)  # , num_parallel_calls=tf.data.AUTOTUNE)

# --------------------
# TODO: Put all above in a function and call them for both train/val to generate generators

# only augment train data
# shift last
ds_train = ds_train.map(lambda x, y: random_fliplr(x, y), num_parallel_calls=4)
ds_train = ds_train.map(lambda x, y: random_flipud(x, y), num_parallel_calls=4)
ds_train = ds_train.map(lambda x, y: (random_brightness(x, brightness=0.2), y), num_parallel_calls=4)  # ADDITIVE
ds_train = ds_train.map(lambda x, y: (random_hue(x, max_delta=0.05), y), num_parallel_calls=4)  # ADDITIVE
ds_train = ds_train.map(lambda x, y: (random_saturation(x, saturation=0.5), y),
                            num_parallel_calls=4)  # @TODO: MULTIPLICATIVE?
ds_train = ds_train.map(lambda x, y: (random_blur(x), y), num_parallel_calls=4)
ds_train = ds_train.map(lambda x, y: random_shift(x, y, translate=50), num_parallel_calls=4)
# shift last

convs = [8, 16, 32, 64, 64, 128, 128, 256]  # 128, 128, 64, 64, 32, 16, 8
convs = convs + convs[:-1][::-1]

network = Unet(input_shape=(img_size, img_size, 3), nb_classes=nb_classes)  # binary = 2
network.set_convolutions(convs)
model = network.create()

# print(model.summary())
# @TODO: Plot loss for each class (invasive, benign and inSitu seperately)

history = CSVLogger(
    history_path + "history_" + name + ".csv",
    append=True
)

early = EarlyStopping(
    monitor="val_loss",
    min_delta=0,  # 0: any improvement is considered an improvement
    patience=50,  # if not improved for 50 epochs, stops
    verbose=1,
     mode="min",  # set "min" for catching the lowest val_loss
    restore_best_weights=False,
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
    optimizer=tf.keras.optimizers.Adam(ret.learning_rate),
    loss=network.get_dice_loss(),
    # metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    run_eagerly=False,
)

history = model.fit(
    ds_train,
    steps_per_epoch=N_train_tot // ret.batch_size,
    epochs=ret.epochs,
    validation_data=ds_val,
    validation_steps=N_val_tot // ret.batch_size,
    callbacks=[save_best, history, early],
    verbose=1,
)
