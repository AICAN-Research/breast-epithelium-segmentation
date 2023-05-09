#code from: https://github.com/mlyg/unified-focal-loss/blob/main/loss_functions.py

from tensorflow.keras import backend as K
import tensorflow as tf


# By Erik Smistad (from network.get_dice_loss()
def get_dice_loss(nb_classes=2, dims=2, use_background=False):
    def dice_loss(target, output, epsilon=1e-10):
        # @TODO: could I change the smoothing?
        smooth = 1.
        dice = 0
        for object in range(0 if use_background else 1, nb_classes):
            if dims == 2:
                output1 = output[:, :, :, object]
                target1 = target[:, :, :, object]
            else:
                output1 = output[:, :, :, :, object]
                target1 = target[:, :, :, :, object]
            intersection1 = tf.reduce_sum(output1 * target1)
            union1 = tf.reduce_sum(output1 * output1) + tf.reduce_sum(target1 * target1)
            dice += (2. * intersection1 + smooth) / (union1 + smooth)
        if use_background:
            dice /= nb_classes
        else:
            dice /= (nb_classes - 1)
        return tf.clip_by_value(1. - dice, 0., 1. - epsilon)
    return dice_loss


def class_dice_loss(class_val, metric_name):
    def dice_loss(y_true, y_pred):
        smooth = 1.
        output1 = y_pred[:, :, :, class_val]
        gt1 = y_true[:, :, :, class_val]

        intersection1 = tf.reduce_sum(output1 * gt1)
        union1 = tf.reduce_sum(output1 * output1) + tf.reduce_sum(gt1 * gt1)
        dice = (2. * intersection1 + smooth) / (union1 + smooth)

        # self.dice_values.assign_add((1 - dice) / 10)
        return 1 - dice

    # set name of metric to be shown in tensorflow progress bar
    dice_loss.__name__ = metric_name

    return dice_loss


# Helper function to enable loss function to be flexibly used for
# both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn
def identify_axis(shape):
    # Three-dimensional
    if len(shape) == 5:
        return [1, 2, 3]
    # Two-dimensional
    elif len(shape) == 4:
        return [1, 2]
    # Exception - Unknown
    else:
        raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')


def categorical_focal_tversky_loss(delta=0.7, gamma=0.75, nb_classes=4):
    """
    :param delta: float, optional
       controls weight given to false positive and false negatives, by default 0.7
    :param gamma: float, optional
       focal parameter controls degree of down-weighting of easy examples, by default 0.75
    :param nb_classes:
    :return:
    """
    # based on: https://github.com/mlyg/unified-focal-loss/issues/13
    def focal_tversky(y_true, y_pred):
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        axis = [1, 2]  #identify_axis(y_true.get_shape())
        smooth = 1.

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1 - y_pred), axis=axis)
        fp = K.sum((1 - y_true) * y_pred, axis=axis)
        class_ = (tp + smooth) / (tp + delta * fn + (1 - delta) * fp + smooth)

        # calculate losses separately for each class
        loss = 0
        for class_val in range(nb_classes):
            loss += (1 - class_[:, class_val]) * K.pow(1 - class_[:, class_val], -gamma)

        # Average class scores
        loss /= nb_classes
        return K.clip(loss, epsilon, 1. - epsilon)

    return focal_tversky


# based on code from https://github.com/mlyg/unified-focal-loss/blob/main/loss_functions.py and
# equation (12) in https://arxiv.org/pdf/2102.04525.pdf
def categorical_focal_tversky_loss_2(nb_classes=4, dims=2, use_background=False):
    """
    A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation
    Link: https://arxiv.org/abs/1810.07842
     :param gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    :param delta:
    :param smooth:
    :param nb_classes:
    :return:
    """
    def loss_function(y_true, y_pred, delta=0.7, gamma=0.75, smooth=0.000001):
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        #y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        loss = 0
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        for object in range(0 if use_background else 1, nb_classes):
            output1 = y_pred[:, :, :, object]
            target1 = y_true[:, :, :, object]

            tp = tf.reduce_sum(target1 * output1)
            fn = tf.reduce_sum(target1 * (1 - output1))
            fp = tf.reduce_sum((1 - target1) * output1)
            tversky_class = (tp + smooth) / (tp + delta * fn + (1 - delta) * fp + smooth)

            # calculate losses separately for each class, enhancing both classes
            loss += (K.pow(1 - tversky_class, (1/gamma)))

        # Average class scores
        loss = loss/(nb_classes - 1)
        return K.clip(loss, epsilon, 1. - epsilon)

    return loss_function