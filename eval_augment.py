import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from utils import normalize_img, patchReader
from augment import random_rot90, random_fliplr, random_flipud, random_hue, random_saturation, random_brightness


# evaluate augmentation
def eval_augmentation():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    plot_flag = True

    ds_name = '081222_135917_level_2_psize_512_ds_4/'  # change manually to determine desired dataset

    ds_train_path = './datasets/' + ds_name + 'ds_train/invasive/'

    # load generated train dataset
    patches = os.listdir(ds_train_path)
    paths = np.array([ds_train_path + x for x in patches]).astype("U400")
    ds_train = tf.data.Dataset.from_tensor_slices(paths)
    ds_train = ds_train.map(lambda x: tf.py_function(patchReader, [x], [tf.float32, tf.float32]), num_parallel_calls=4)

    ds_train = ds_train.take(10)  # testing with 10
    ds_train = ds_train.map(normalize_img)

    for image, mask in tqdm(ds_train):
        image_rot, mask_rot = random_rot90(image, mask)
        image_fup, mask_fup = random_flipud(image, mask)
        image_flr, mask_flr = random_fliplr(image, mask)
        image_rs = random_saturation(image, saturation=0.5)
        image_rh = random_hue(image, max_delta=0.1)
        image_rb = random_brightness(image, brightness=0.2)

        if plot_flag:
            f, axes = plt.subplots(3, 3)  # Figure of the two corresponding TMAs
            axes[0, 0].imshow(image)
            axes[0, 0].set_title("image")

            axes[0, 1].imshow(image_rot)
            axes[0, 1].set_title("image rot")

            axes[0, 2].imshow(image_fup)
            axes[0, 2].set_title("image fup")

            axes[1, 0].imshow(image_flr)
            axes[1, 0].set_title("image flr")

            axes[1, 1].imshow(image_rb)
            axes[1, 1].set_title("image rb")

            axes[1, 2].imshow(image_rh)
            axes[1, 2].set_title("image rh")

            axes[2, 0].imshow(image_rs)
            axes[2, 0].set_title("image rs")

            plt.show()
            exit()


if __name__ == "__main__":
    eval_augmentation()
    