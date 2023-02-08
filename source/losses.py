#code from: https://github.com/mlyg/unified-focal-loss/blob/main/loss_functions.py

from tensorflow.keras import backend as K


# Helper function to enable loss function to be flexibly used for
# both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn
def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5:
        return [1, 2, 3]
    # Two dimensional
    elif len(shape) == 4:
        return [1, 2]
    # Exception - Unknown
    else:
        raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')


def categorical_focal_tversky_loss(delta=0.7, gamma=0.75, nb_classes=4):
    """This is the implementation for multiclass segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
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
        dice_class = (tp + smooth) / (tp + delta * fn + (1 - delta) * fp + smooth)  # output: (batch, classes)

        # calculate losses separately for each class, enhancing both classes
        loss = 0
        for class_val in range(nb_classes):
            loss += (1 - dice_class[:, class_val]) * K.pow(1 - dice_class[:, class_val], -gamma)

        # Average class scores
        loss /= nb_classes
        return K.clip(loss, epsilon, 1. - epsilon)
    return focal_tversky

# based on code from https://github.com/mlyg/unified-focal-loss/blob/main/loss_functions.py and
# equation (12) in https://arxiv.org/pdf/2102.04525.pdf
def categorical_focal_tversky_loss_2(delta=0.7, gamma=0.75, smooth=0.000001, nb_classes=4):
    """A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation
    Link: https://arxiv.org/abs/1810.07842
    Parameters
    ----------
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """

    def loss_function(y_true, y_pred):
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        axis = [1, 2]

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1 - y_pred), axis=axis)
        fp = K.sum((1 - y_true) * y_pred, axis=axis)
        tversky_class = (tp + smooth) / (tp + delta * fn + (1 - delta) * fp + smooth)

        # calculate losses separately for each class, enhancing both classes
        loss = 0
        for class_val in range(1, nb_classes):
            loss += (K.pow(1 - tversky_class[:, class_val], (1/gamma)))

        # Average class scores
        loss = loss/(nb_classes - 1)
        return K.clip(loss, epsilon, 1. - epsilon)

        return focal_tversky_loss

    return loss_function