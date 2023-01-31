import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from source.utils import normalize_img, patchReader
from source.augment import random_rot90, random_fliplr, \
    random_flipud, random_hue, random_saturation, random_brightness, random_blur


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

    ds_train = ds_train.take(100)  # testing with 10
    ds_train = ds_train.map(normalize_img)

    count = 0
    image_alls = []
    for image, mask in tqdm(ds_train):

        image_all, mask_rot = random_rot90(image, mask)
        image_all, mask_fup = random_flipud(image_all, mask)
        image_all, mask_flr = random_fliplr(image_all, mask)
        image_all = random_saturation(image_all, saturation=0.5)
        image_all = random_hue(image_all, max_delta=0.05)
        image_all = random_brightness(image_all, brightness=0.2)
        image_all = random_blur(image_all)
        image_alls.append(image_all)

        if plot_flag and count > 8:
            f, axes = plt.subplots(3, 3)  # Figure of the two corresponding TMAs
            axes[0, 0].imshow(image_alls[0])
            axes[0, 0].set_title("all1")

            axes[0, 1].imshow(image_alls[1])
            axes[0, 1].set_title("all2")

            axes[0, 2].imshow(image_alls[2])
            axes[0, 2].set_title("all3")

            axes[1, 0].imshow(image_alls[3])
            axes[1, 0].set_title("all4")

            axes[1, 1].imshow(image_alls[4])
            axes[1, 1].set_title("all5")

            axes[1, 2].imshow(image_alls[5])
            axes[1, 2].set_title("all6")

            axes[2, 0].imshow(image_alls[6])
            axes[2, 0].set_title("all7")

            axes[2, 1].imshow(image_alls[7])
            axes[2, 1].set_title("all8")

            axes[2, 2].imshow(image_alls[8])
            axes[2, 2].set_title("all9")

            plt.show()
        count += 1
        if count == 10:
            count = 0
            image_alls.clear()


if __name__ == "__main__":
    eval_augmentation()
