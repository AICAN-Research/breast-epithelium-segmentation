import numpy as np
import logging as log
import tensorflow as tf
import tensorflow_datasets as tfds
import h5py


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
        # TODO: cpu, gpu issue. Slow?
        #temp_gt = np.zeros(gt.shape)  # only do if all epithlium as one class
        #temp_gt = ((gt[:, :, 1] == 1) | (gt[:, :, 2] == 1) | (gt[:, :, 3] == 1))  # only do if all epithelium as one class
        #gt = temp_gt  # only do if all epithelium as one class
    return image, gt


def get_random_path_from_random_class(x1, x2, x3):
    nested_class_folder = [x1, x2, x3]

    # make infinite generator
    while True:
        # get random class
        random_class_int = np.random.randint(0, len(nested_class_folder), 1)[0]
        chosen_class = nested_class_folder[random_class_int]

        # get random patch from selected class
        random_patch = np.random.choice(chosen_class, 1)[0].decode('utf-8')

        # convert to tensor
        random_patch = tf.convert_to_tensor(random_patch, dtype=tf.string)

        yield random_patch


# generator to use when all epithelium as one class
def get_random_path(x1):

    # make infinite generator
    while True:
        # get random patch
        random_patch = np.random.choice(x1, 1)[0].decode('utf-8')

        # convert to tensor
        random_patch = tf.convert_to_tensor(random_patch, dtype=tf.string)

        yield random_patch


# by André Pedersen:
def create_multiscale_input(gt, nb_downsamples):
    hierarchical_gt = [gt, ]
    for i in range(1, nb_downsamples):
        tmp = tf.identity(gt)
        limit = int(pow(2, i))
        new_gt = tmp[0::limit, 0::limit]
        hierarchical_gt.append(new_gt)
    return tuple(hierarchical_gt)


def define_logger(verbose=1):
    """
    method which sets the verbose handler
    """
    if verbose == 0:
        level = None
    elif verbose == 1:
        level = log.INFO
    elif verbose == 2:
        level = log.DEBUG
    elif verbose == 3:
        level = log.WARNING
    else:
        raise ValueError("Unknown verbose was set. 0 to disable verbose, 1 for INFO, 2 for DEBUG, 3 for WARNING.")

    log.basicConfig(
        format="%(levelname)s %(filename)s %(lineno)s %(message)s",
        level=level
        )
