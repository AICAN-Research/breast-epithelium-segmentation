"""
Metrics file for multiclass epithelium segmentation. Recall, precision, dice
"""
from tensorflow.python.keras import backend as K


def check_units(y_true, y_pred):
    """
    based on https://github.com/andreped/H2G-Net/blob/main/src/utils/metrics.py
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
    :param nb_classes:
    :param use_background:
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
        target1, output1 = check_units(target1, output1)
        true_positives = K.sum(K.round(K.clip(target1 * output1, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(output1, 0, 1)))
        precision_ += true_positives / (predicted_positives + K.epsilon())

        if use_background:
            precision_ /= nb_classes
        else:
            precision_ /= (nb_classes - 1)
        # @TODO: maybe clip at end instead
    return precision_


def recall(y_true, y_pred):
    """
    from https://github.com/andreped/H2G-Net/blob/main/src/utils/metrics.py
    and network.get_dice_loss()
    :param y_true: true values
    :param y_pred: predicted values
    :return: recall: tp / (tp + fn)
    """
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_ = true_positives / (possible_positives + K.epsilon())
    return recall_

