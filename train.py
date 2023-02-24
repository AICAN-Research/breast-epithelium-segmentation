import tensorflow as tf
import os
from deep_learning_tools.network import Unet
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, TensorBoard
from datetime import datetime, date
from source.augment import random_brightness, random_fliplr, random_flipud, \
    random_hue, random_saturation, random_shift, random_blur, random_contrast
from source.utils import normalize_img, patchReader, get_random_path_from_random_class, class_dice_loss, \
    class_categorical_focal_dice_loss, categorical_focal_dice_loss, create_multiscale_input
from source.losses import get_dice_loss, categorical_focal_tversky_loss, categorical_focal_tversky_loss_2
from argparse import ArgumentParser
import sys
from AttentionUNet import AttentionUnet


def main(ret):
    curr_date = "".join(date.today().strftime("%d/%m").split("/")) + date.today().strftime("%Y")[2:]
    curr_time = "".join(str(datetime.now()).split(" ")[1].split(".")[0].split(":"))

    img_size = 1024  #@TODO: None not allowed with convolutions, why?
    nb_classes = 4
    class_names = ["invasive", "benign", "insitu"]

    # network stuff
    encoder_convs = [16, 32, 32, 64, 64, 128, 128, 256, 256]
    nb_downsamples = len(encoder_convs) - 1
    architecture = "agunet"
    # @TODO: Calculate which output layer name (top prediction) you get from deep supervision AGU-Net

    name = curr_date + "_" + curr_time + "_" + architecture + "_bs_" + str(ret.batch_size)  # + "_eps_" + str(ret.epochs)

    # paths
    dataset_path = './datasets/220223_140143_level_2_psize_1024_ds_4/'  # path to directory
    train_path = dataset_path + 'ds_train'
    val_path = dataset_path + 'ds_val'
    # test_path = dataset_path + 'ds_test'
    history_path = './output/history/'  # path to directory
    model_path = './output/models/'  # path to directory
    save_ds_path = './output/datasets/dataset_' + name + '/'  # inni her først en med name, så ds_train og test inni der

    N_train_batches = 120  # @TODO: Change this number
    N_val_batches = 30

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
        dir_path = val_path + "/" + directory + "/"
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
    ds_train = ds_train.map(lambda x: tf.py_function(patchReader, [x], [tf.float32, tf.float32]),
                            num_parallel_calls=ret.proc, deterministic=False)
    ds_val = ds_val.map(lambda x: tf.py_function(patchReader, [x], [tf.float32, tf.float32]),
                        num_parallel_calls=ret.proc, deterministic=False)

    # @TODO: Check if good idea to do deterministic=False here as well (as in lines above)
    # normalize intensities
    ds_train = ds_train.map(normalize_img)  # , num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.map(normalize_img)  # , num_parallel_calls=tf.data.AUTOTUNE)

    # batch data before aug -> faster
    #ds_train = ds_train.batch(ret.batch_size)
    #ds_val = ds_val.batch(ret.batch_size)

    # --------------------
    # TODO: Put all above in a function and call them for both train/val to generate generators

    # only augment train data

    # shift last
    ds_train = ds_train.map(lambda x, y: random_fliplr(x, y), num_parallel_calls=1)
    ds_train = ds_train.map(lambda x, y: random_flipud(x, y), num_parallel_calls=1)
    ds_train = ds_train.map(lambda x, y: (random_brightness(x, brightness=0.2), y), num_parallel_calls=1)  # ADDITIVE
    ds_train = ds_train.map(lambda x, y: (random_hue(x, max_delta=0.05), y), num_parallel_calls=1)  # ADDITIVE
    ds_train = ds_train.map(lambda x, y: (random_saturation(x, saturation=0.5), y),
                            num_parallel_calls=1)  # @TODO: MULTIPLICATIVE?
    ds_train = ds_train.map(lambda x, y: (random_blur(x), y), num_parallel_calls=1)
    ds_train = ds_train.map(lambda x, y: (random_contrast(x, low=0.6, up=0.8), y), num_parallel_calls=1)
    ds_train = ds_train.map(lambda x, y: random_shift(x, y, translate=50),
                            num_parallel_calls=1)  # @TODO: DO I need to do shift, is it really necessary?
    # shift last

    # create multiscale input
    # tf.py_function(patchReader, [x], [tf.float32, tf.float32])
    ds_train = ds_train.map(lambda x, y: (x, create_multiscale_input(y, nb_downsamples)), num_parallel_calls=1)
    ds_val = ds_val.map(lambda x, y: (x, create_multiscale_input(y, nb_downsamples)), num_parallel_calls=1)

    # batch data before aug -> faster
    ds_train = ds_train.batch(ret.batch_size)
    ds_val = ds_val.batch(ret.batch_size)

    # prefetch augmented batches -> GPU does not need to wait -> batch always ready
    ds_train = ds_train.prefetch(1)
    ds_val = ds_val.prefetch(1)

    if architecture == "unet":
        # define network architecture
        #convs = [8, 16, 32, 64, 64, 128, 128, 256]  # 128, 128, 64, 64, 32, 16, 8
        convs = encoder_convs + encoder_convs[:-1][::-1]
        network = Unet(input_shape=(img_size, img_size, 3), nb_classes=nb_classes)  # binary = 2
        network.set_convolutions(convs)
        model = network.create()
    elif architecture == "agunet":
        # Test new Attention UNet
        agunet = AttentionUnet(input_shape=(1024, 1024, 3), nb_classes=nb_classes, deep_supervision=True, input_pyramid=True)
        agunet.decoder_dropout = 0.1
        agunet.set_convolutions(encoder_convs)
        model = agunet.create()
    else:
        raise ValueError("Unsupported architecture chosen. Please, choose either 'unet' or 'agunet'.")

    print(model.summary())
    # @TODO: Plot loss for each class (invasive, benign and inSitu seperately)
    # metrics to monitor training and validation loss for each class (invasive, benign and inSitu)

    history = CSVLogger(
        history_path + "history_" + name + ".csv",
        append=True
    )

    # tensorboard history logger
    tb_logger = TensorBoard(log_dir="output/logs/" + name + "/", histogram_freq=0, update_freq="epoch")

    early = EarlyStopping(
        monitor="val_conv2d_72_loss",  # "val_loss"
        min_delta=0,  # 0: any improvement is considered an improvement
        patience=ret.patience,  # if not improved for 50 epochs, stops
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

    model.compile(
        optimizer=tf.keras.optimizers.Adam(ret.learning_rate),
        loss=get_dice_loss(nb_classes=nb_classes, use_background=False, dims=2),  # network.get_dice_loss(),
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
    parser.add_argument('--batch_size', metavar='--bs', type=int, nargs='?', default=8,
                        help="set which batch size to use for training.")
    parser.add_argument('--learning_rate', metavar='--lr', type=float, nargs='?', default=0.0001,
                        help="set which learning rate to use for training.")
    parser.add_argument('--epochs', metavar='--ep', type=int, nargs='?', default=500,
                        help="number of epochs to train.")
    parser.add_argument('--patience', metavar='--pa', type=int, nargs='?', default=50,
                        help="number of epochs to wait (patience) for early stopping.")
    parser.add_argument('--proc', metavar='--pr', type=int, nargs='?', default=4,
                        help="number of workers to use with tf.data.")
    ret = parser.parse_known_args(sys.argv[1:])[0]

    print(ret)

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # due to this: https://github.com/tensorflow/tensorflow/issues/35029

    # choose which GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    main(ret)

    print("Finished!")
