import tensorflow as tf
import os
from deep_learning_tools.network import Unet
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, TensorBoard
from datetime import datetime, date
from source.augment import random_brightness, random_fliplr, random_flipud, \
    random_hue, random_saturation, random_shift, random_blur
from source.utils import normalize_img, patchReader, get_random_path_from_random_class, \
     create_multiscale_input, get_random_path
from source.losses import get_dice_loss, class_dice_loss
from argparse import ArgumentParser
import sys
from AttentionUNet import AttentionUnet
from gradient_accumulator import GradientAccumulateModel
from tensorflow.keras import mixed_precision


def main(ret):
    curr_date = "".join(date.today().strftime("%d/%m").split("/")) + date.today().strftime("%Y")[2:]
    curr_time = "".join(str(datetime.now()).split(" ")[1].split(".")[0].split(":"))

    img_size = 1024

    # network stuff
    encoder_convs = [16, 32, 32, 64, 64, 128, 128, 256, 256]
    nb_downsamples = len(encoder_convs) - 1
    N_train_batches = 120  # @TODO: Change this number
    N_val_batches = 30

    lr_temp = str(ret.learning_rate)
    br_temp = str(ret.brightness)

    name = curr_date + "_" + curr_time + "_" + ret.network + "_bs_" + str(ret.batch_size) + "_as_" + \
        str(ret.accum_steps) + "_lr_" + lr_temp.split(".")[0] + lr_temp.split(".")[1] + "_d_" + \
        "_bl_" + str(ret.blur) + "_br_" + br_temp.split(".")[0] + \
        br_temp.split(".")[1] + "_h_" + "_s_" + str(ret.saturation) + "_st_" + str(ret.shift) + "_mp_" + \
        ret(mixed_precision) + "_ntb_" + str(N_train_batches) + "_nvb_" + str(N_val_batches)

    # paths
    dataset_path = '/mnt/EncryptedSSD1/maren/datasets/200423_125554_level_2_psize_1024_ds_4/'
    dataset_path_wsi = '/mnt/EncryptedSSD1/maren/datasets/210423_122737_wsi_level_2_psize_1024_ds_4/'
    train_path = dataset_path + 'ds_train'
    train_path_wsi = dataset_path_wsi + 'ds_train'
    val_path = dataset_path + 'ds_val'
    val_path_wsi = dataset_path_wsi + 'ds_val'
    # test_path = dataset_path + 'ds_test'
    history_path = './output/history/'  # path to directory
    model_path = './output/models/'  # path to directory

    # use this when only looking at all epithelium as one class
    if ret.nbr_classes == 2:
        class_names = ["epithelium"]
        train_paths = []
        for file_ in os.listdir(train_path + "/"):
            file_path = train_path + "/" + file_
            train_paths.append(file_path)  # nested list of three lists containing paths for each folder/class

        val_paths = []
        for file_ in os.listdir(val_path + "/"):
            file_path = val_path + "/" + file_
            val_paths.append(file_path)  # nested list of three lists containing paths for each folder/class

        # combine all train/val paths
        ds_train = tf.data.Dataset.from_generator(
            get_random_path,  # get_random_path_from_random_class, for inv, ins, ben
            output_shapes=tf.TensorShape([]),
            output_types=tf.string,
            args=[train_paths]  # val_paths for inv, ins, ben
        )

        ds_val = tf.data.Dataset.from_generator(
            get_random_path,  # get_random_path_from_random_class, for inv, ins, ben
            output_shapes=tf.TensorShape([]),
            output_types=tf.string,
            args=[val_paths]  # val_paths for inv, ins, ben @TODO: why brackets, does not make sense(?)
        )

    # use this with invasive, benign, insitu
    if ret.nbr_classes == 4:
        class_names = ["invasive", "benign", "insitu"]
        train_paths = []
        for directory in os.listdir(train_path):
            dir_path = train_path + "/" + directory + "/"
            dir_paths = []
            for file_ in os.listdir(dir_path):
                file_path = dir_path + file_
                dir_paths.append(file_path)
            train_paths.append(dir_paths)  # nested list of three lists containing paths for each folder/class
        for i, directory in enumerate(os.listdir(train_path_wsi)):
            dir_path = train_path_wsi + "/" + directory + "/"
            for file_ in os.listdir(dir_path):
                file_path = dir_path + file_
                train_paths[i].append(file_path)  # nested list of three lists containing paths for each folder/class

        val_paths = []
        for directory in os.listdir(val_path):
            dir_path = val_path + "/" + directory + "/"
            dir_paths = []
            for file_ in os.listdir(dir_path):
                file_path = dir_path + file_
                dir_paths.append(file_path)
            val_paths.append(dir_paths)  # nested list of three lists containing paths for each folder/class
        for i, directory in enumerate(os.listdir(val_path_wsi)):
            dir_path = val_path_wsi + "/" + directory + "/"
            for file_ in os.listdir(dir_path):
                file_path = dir_path + file_
                val_paths[i].append(file_path)  # nested list of three lists containing paths for each folder/class

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
    ds_train = ds_train.map(lambda x: tf.py_function(patchReader, [x], [tf.float32, tf.float32]),
                            num_parallel_calls=ret.proc, deterministic=False)
    ds_val = ds_val.map(lambda x: tf.py_function(patchReader, [x], [tf.float32, tf.float32]),
                        num_parallel_calls=ret.proc, deterministic=False)

    # @TODO: Check if good idea to do deterministic=False here as well (as in lines above)
    # normalize intensities
    ds_train = ds_train.map(normalize_img)  # , num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.map(normalize_img)  # , num_parallel_calls=tf.data.AUTOTUNE)

    # batch data before aug -> faster, can't do with agunet
    # ds_train = ds_train.batch(ret.batch_size)
    # ds_val = ds_val.batch(ret.batch_size)

    # only augment train data
    # shift last
    ds_train = ds_train.map(lambda x, y: random_fliplr(x, y), num_parallel_calls=1)
    ds_train = ds_train.map(lambda x, y: random_flipud(x, y), num_parallel_calls=1)
    if ret.brightness:
        ds_train = ds_train.map(lambda x, y: (random_brightness(x, brightness=ret.brightness), y), num_parallel_calls=1)  # ADDITIVE
    if ret.hue:
        ds_train = ds_train.map(lambda x, y: (random_hue(x, max_delta=ret.hue), y), num_parallel_calls=1)  # ADDITIVE
    if ret.saturation:
        ds_train = ds_train.map(lambda x, y: (random_saturation(x, saturation=ret.saturation), y),
                            num_parallel_calls=1)  # @TODO: MULTIPLICATIVE?
    if ret.blur:
        ds_train = ds_train.map(lambda x, y: (random_blur(x), y), num_parallel_calls=1)
    if ret.shift:
        ds_train = ds_train.map(lambda x, y: random_shift(x, y, translate=50), num_parallel_calls=1)

    # create multiscale input
    # tf.py_function(patchReader, [x], [tf.float32, tf.float32])
    if ret.network == "agunet":
        ds_train = ds_train.map(lambda x, y: (x, create_multiscale_input(y, nb_downsamples)), num_parallel_calls=1)
        ds_val = ds_val.map(lambda x, y: (x, create_multiscale_input(y, nb_downsamples)), num_parallel_calls=1)

    # batch data before aug -> faster
    ds_train = ds_train.batch(ret.batch_size)
    ds_val = ds_val.batch(ret.batch_size)

    # prefetch augmented batches -> GPU does not need to wait -> batch always ready
    ds_train = ds_train.prefetch(1)
    ds_val = ds_val.prefetch(1)

    if ret.network == "unet":
        convs = encoder_convs + encoder_convs[:-1][::-1]
        network = Unet(input_shape=(img_size, img_size, 3), nb_classes=ret.nbr_classes)  # binary = 2
        network.set_convolutions(convs)
        model = network.create()
    elif ret.network == "agunet":
        agunet = AttentionUnet(input_shape=(1024, 1024, 3), nb_classes=ret.nbr_classes,
                               encoder_spatial_dropout=ret.dropout, decoder_spatial_dropout=ret.dropout,
                               accum_steps=ret.accum_steps, deep_supervision=True, input_pyramid=True, grad_accum=False,
                               encoder_use_bn=True, decoder_use_bn=True)
        agunet.set_convolutions(encoder_convs)
        model = agunet.create()

    else:
        raise ValueError("Unsupported architecture chosen. Please, choose either 'unet' or 'agunet'.")

    if ret.accum_steps > 1:
        model = GradientAccumulateModel(
            accum_steps=ret.accum_steps, mixed_precision=ret.mixed_precision, inputs=model.input, outputs=model.outputs
        )

    print(model.summary())

    history = CSVLogger(
        history_path + "history_" + name + ".csv",
        append=True
    )

    # tensorboard history logger
    tb_logger = TensorBoard(log_dir="output/logs/" + name + "/", histogram_freq=0, update_freq="epoch")

    early = EarlyStopping(
        monitor="val_conv2d_72_loss",  # "val_loss"
        min_delta=0,  # 0: any improvement is considered an improvement
        patience=ret.patience,  # if not improved for ret.patience epochs, stops
        verbose=1,
        mode="min",  # set "min" for catching the lowest val_loss
        restore_best_weights=False,
    )

    save_best = ModelCheckpoint(
        model_path + "model_" + name,
        monitor="val_conv2d_72_loss",  # "val_loss"
        verbose=2,  #
        save_best_only=True,
        save_weights_only=False,
        mode="min",  # use "auto" with "f1_score", "auto" with "val_loss" (or "min")
        save_freq="epoch"
    )

    opt = tf.keras.optimizers.Adam(ret.learning_rate)  # , epsilon=1e-4)
    if ret.mixed_precision:
        opt = mixed_precision.LossScaleOptimizer(opt)

    model.compile(
        optimizer=opt,
        loss=get_dice_loss(nb_classes=ret.nbr_classes, use_background=False, dims=2),
        # loss_weights=None if architecture == "unet" else loss_weights,
        metrics=[
            *[class_dice_loss(class_val=i + 1, metric_name=x) for i, x in enumerate(class_names)]
        ],
        run_eagerly=False,
    )

    model.fit(
        ds_train,
        steps_per_epoch=N_train_batches,
        epochs=ret.epochs,
        validation_data=ds_val,
        validation_steps=N_val_batches,
        callbacks=[save_best, history, early, tb_logger],
        verbose=1,
    )


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--batch_size', metavar='--bs', type=int, nargs='?', default=16,
                        help="set which batch size to use for training.")
    parser.add_argument('--accum_steps', metavar='--as', type=int, nargs='?', default=2,
                        help="set how many gradient accumulations to perform.")
    parser.add_argument('--mixed_precision', metavar='--mp', type=int, nargs='?', default=1,
                        help="whether to perform mixed precision (float16). Default=1 (True).")
    parser.add_argument('--learning_rate', metavar='--lr', type=float, nargs='?', default=0.0001,
                        help="set which learning rate to use for training.")
    parser.add_argument('--epochs', metavar='--ep', type=int, nargs='?', default=500,
                        help="number of epochs to train.")
    parser.add_argument('--patience', metavar='--pa', type=int, nargs='?', default=200,
                        help="number of epochs to wait (patience) for early stopping.")
    parser.add_argument('--proc', metavar='--pr', type=int, nargs='?', default=4,
                        help="number of workers to use with tf.data.")
    parser.add_argument('--gpu', metavar='--g', type=str, nargs='?', default="0",
                        help="which gpu to use.")
    parser.add_argument('--network', metavar='--nw', type=str, nargs='?', default="agunet",
                        help="agunet or unet.")
    parser.add_argument('--nbr_classes', metavar='--nbr_c', type=int, nargs='?', default=4,
                        help="four classes for multiclass, two for single class epithelium segmentation.")
    parser.add_argument('--dropout', metavar='--d', type=float, nargs='?', default=None,
                        help="spatial dropout in encoder and decoder.")
    parser.add_argument('--blur', metavar='--bl', type=int, nargs='?', default=0,
                        help="blur aug added to train set.")
    parser.add_argument('--brightness', metavar='--br', type=float, nargs='?', default=0,
                        help="brightness aug added to train set.")
    parser.add_argument('--hue', metavar='--h', type=float, nargs='?', default=0,
                        help="hue aug added to train set.")
    parser.add_argument('--saturation', metavar='--s', type=float, nargs='?', default=0,
                        help="saturation aug added to train set.")
    parser.add_argument('--shift', metavar='--st', type=float, nargs='?', default=0,
                        help="shift aug added to train set.")
    ret = parser.parse_known_args(sys.argv[1:])[0]

    print(ret)

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # due to this: https://github.com/tensorflow/tensorflow/issues/35029

    # choose which GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = ret.gpu

    if ret.mixed_precision:
        mixed_precision.set_global_policy('mixed_float16')

    main(ret)

    print("Finished!")
