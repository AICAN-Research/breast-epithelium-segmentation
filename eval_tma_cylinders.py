"""
Script for evaluating models trained on patche son 10x on whole TMA cylinders created at 20x
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import os
from tqdm import tqdm
from source.utils import normalize_img, patchReader
from stats import BCa_interval_macro_metric
import h5py
import fast
import cv2


# No smoothing when evaluating, to make differenciable during training
def class_dice_(y_true, y_pred, class_val):
    output1 = y_pred[:, :, :, class_val]
    gt1 = y_true[:, :, :, class_val]

    intersection1 = tf.reduce_sum(output1 * gt1)
    union1 = tf.reduce_sum(output1 * output1) + tf.reduce_sum(gt1 * gt1)  #@TODO: why do we need output*output in reduce sum?
    if union1 == 0:
        dice = 0.
        dice_u = True
    else:
        dice = (2. * intersection1) / union1
        dice_u = False

    return dice, dice_u


def class_dice_class_present(y_true, y_pred, class_val):
    count = False

    output1 = y_pred[:, :, :, class_val]
    gt1 = y_true[:, :, :, class_val]

    intersection1 = tf.reduce_sum(output1 * gt1)
    union1 = tf.reduce_sum(output1 * output1) + tf.reduce_sum(gt1 * gt1)  #@TODO: why do we need output*output in reduce sum?
    if union1 == 0:
        dice = 0.
        dice_u = True
    else:
        dice = (2. * intersection1) / union1
        dice_u = False

    if tf.reduce_sum(gt1):
        count = True

    return dice, count, dice_u


#  @TODO: look at what they do in scikit-learn for edge cases. They set precision/recall to zero if denominator is zero
#  @TODO: by default, but one can set to 1
def precision(y_true, y_pred, object_):
    """
    based on https://github.com/andreped/H2G-Net/blob/main/src/utils/metrics.py
    and network.get_dice_loss()
    :param y_true: true values
    :param y_pred: predicted values
    :param nb_classes: number of classes
    :param use_background: True or False
    :param dims:
    :return: precision: tp / (tp + fp)
    """
    precision_ = 0

    output1 = y_pred[:, :, :, object_]
    target1 = y_true[:, :, :, object_]

    true_positives = tf.reduce_sum(target1 * output1)
    predicted_positives = tf.reduce_sum(output1)
    if predicted_positives == 0:
        precision_ = 0
    else:
        precision_ += true_positives / predicted_positives

    return precision_


def precision_class_present(y_true, y_pred, object_):
    """
    Only calculate precision when there are positives in y_true
    :param y_true: true values
    :param y_pred: predicted values
    :param nb_classes: number of classes
    :param use_background: True or False
    :param dims:
    :return: precision: tp / (tp + fp), True/False depending on whether there are positives in image
    """
    precision_ = 0
    count = False

    output1 = y_pred[:, :, :, object_]
    target1 = y_true[:, :, :, object_]

    true_positives = tf.reduce_sum(target1 * output1)
    predicted_positives = tf.reduce_sum(output1)
    if predicted_positives == 0:
        precision_ = 0
    else:
        precision_ += true_positives / predicted_positives
    if tf.reduce_sum(target1):
        count = True

    return precision_, count


def recall(y_true, y_pred, object_):
    """
    based on https://github.com/andreped/H2G-Net/blob/main/src/utils/metrics.py
    and network.get_dice_loss()
    :param y_true: true values
    :param y_pred: predicted values
    :param nb_classes: number of classes
    :param use_background: True or False
    :param dims:
    :return: recall: tp / (tp + fn)
    """
    recall_ = 0

    output1 = y_pred[:, :, :, object_]
    target1 = y_true[:, :, :, object_]

    true_positives = tf.reduce_sum(target1 * output1)  # TODO: consider reduce_sum vs K.sum, is there a difference in speed
    possible_positives = tf.reduce_sum(target1)
    if possible_positives == 0:
        recall_ = 0
    else:
        recall_ += true_positives / possible_positives

    return recall_


def recall_class_present(y_true, y_pred, object_):
    """
    Only calculate recall when there are positives in y_true
    :param y_true: true values
    :param y_pred: predicted values
    :param nb_classes: number of classes
    :param use_background: True or False
    :param dims:
    :return: recall: tp / (tp + fn)
    """
    recall_ = 0
    count = False

    output1 = y_pred[:, :, :, object_]
    target1 = y_true[:, :, :, object_]

    true_positives = tf.reduce_sum(target1 * output1)  # TODO: consider reduce_sum vs K.sum, is there a difference in speed
    possible_positives = tf.reduce_sum(target1)
    if possible_positives == 0:
        recall_ = 0
    else:
        recall_ += true_positives / possible_positives
    if tf.reduce_sum(target1):
        count = True

    return recall_, count


# @TODO: add Jaccard similarity coefficient score


def eval_on_dataset():

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    plot_flag = False
    bs = 1  # @TODO: should bs match bs in train?

    path = './datasets_tma_cores/150523_180206_level_1_ds_4/ds_val/'
    model_name = ''

    with h5py.File(path, "r") as f:
        image = np.asarray(f["input"]).astype("float32")
        gt = np.asarray(f["output"]).astype("float32")

    # gt = np.argmax(gt, axis=-1)  # @TODO: this I shouldn't keep, right? removes one-hot

    print(image.shape, gt.shape)
    image = image.astype("uint8")

    plt.rcParams.update({'font.size': 28})
    f, axes = plt.subplots(2, 2, figsize=(30, 30))
    axes[0, 0].imshow(image)
    axes[0, 1].imshow(gt[:, :, 1])
    axes[1, 0].imshow(gt[:, :, 2])
    axes[1, 1].imshow(gt[:, :, 3])
    plt.show()

    # patch generator that makes patches of size (2048, 2048) then resize to (1024, 1024)
    data = [image, gt]
    data_fast = [fast.Image.createFromArray(curr) for curr in data]
    generators = [fast.PatchGenerator.create(2048, 2048).connect(0, curr) for curr in data_fast]
    streamers = [fast.DataStream(curr) for curr in generators]

    for patch_idx, (patch_he, patch_gt) in enumerate(zip(*streamers)):
        try:
            patch_he = np.asarray(patch_he)
            patch_gt = np.asarray(patch_gt)
        except RuntimeError as e:
            print(e)
            continue

        # resize patch to (1024, 1024)
        patch_he = cv2.resize(patch_he, (1024, 1024, 3), interpolation=cv2.INTER_NEAREST)
        patch_gt = cv2.resize((patch_gt, (1024, 1024, gt.shape[2])), interpolation=cv2.INTER_NEAREST)

        print(patch_he.shape)
        print(patch_gt.shape)

    exit()

if __name__ == "__main__":
    eval_on_dataset()