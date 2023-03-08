"""
Metrics file for multiclass epithelium segmentation. Recall, precision, dice
"""
from tensorflow.python.keras import backend as K
import tensorflow as tf


def check_units(y_true, y_pred):
    """
    from https://github.com/andreped/H2G-Net/blob/main/src/utils/metrics.py
    :param y_true:
    :param y_pred:
    :return:
    """
    if y_pred.shape[1] != 1:
        y_pred = y_pred[:, 1:2]
        y_true = y_true[:, 1:2]
        return y_true, y_pred


def precision(y_true, y_pred, nb_classes, use_background=False, dims=2):
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
    for object_ in range(0 if use_background else 1, nb_classes):
        if dims == 2:
            output1 = y_pred[:, :, :, object_]
            target1 = y_true[:, :, :, object_]
        else:
            output1 = y_pred[:, :, :, :, object_]
            target1 = y_true[:, :, :, :, object_]
        true_positives = tf.reduce_sum(target1 * output1)
        predicted_positives = tf.reduce_sum(output1)
        precision_ += (true_positives / (predicted_positives + K.epsilon()))
        # TODO: consider problem when there are no true positives (or no positives at all) in one of the classes
        # TODO: it will influence the results a lot. Happens often for at least one class.
    if use_background:
        precision_ /= nb_classes
    else:
        precision_ /= (nb_classes - 1)
    # @TODO: maybe clip at end instead
    return precision_


def recall(y_true, y_pred, nb_classes, use_background=False, dims=2):
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
    for object_ in range(0 if use_background else 1, nb_classes):
        if dims == 2:
            output1 = y_pred[:, :, :, object_]
            target1 = y_true[:, :, :, object_]
        else:
            output1 = y_pred[:, :, :, :, object_]
            target1 = y_true[:, :, :, :, object_]
        true_positives = K.sum(K.round(K.clip(target1 * output1, 0, 1)))  # TODO: consider reduce_sum instead, is there a difference in speed
        possible_positives = K.sum(K.round(K.clip(target1, 0, 1)))
        recall_ += (true_positives / (possible_positives + K.epsilon()))

    if use_background:
        recall_ /= nb_classes
    else:
        recall_ /= (nb_classes - 1)
    # @TODO: maybe clip at end instead

    return recall_

