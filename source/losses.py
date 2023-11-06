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
