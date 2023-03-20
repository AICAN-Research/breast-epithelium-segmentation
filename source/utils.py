import cv2
import numpy as np
import logging as log
import tensorflow as tf
import tensorflow_datasets as tfds
import h5py
from copy import deepcopy


# focal dice loss to focus on "difficult classes"
# implemented by André Pedersen:
def categorical_focal_dice_loss(nb_classes, use_background=False, gamma=2):
    def focal_dice_loss(target, output, epsilon=1e-10):
        smooth = 1

        # calculate Dice for each class separately and accumulate
        dice = 0
        for class_val in range(0 if use_background else 1, nb_classes):
            # get prediction and target for current class only
            output1 = output[
                ..., class_val]  # @TODO: doing "..." is probably slower than doing [:, :, :, class_val], but works N-D
            target1 = target[..., class_val]

            # calculate Dice for current class
            intersection1 = tf.reduce_sum(output1 * target1)
            union1 = tf.reduce_sum(output1 * output1) + tf.reduce_sum(target1 * target1)
            dice += (2. * intersection1 + smooth) / (union1 + smooth)

        # divide accumulated dice by class count -> macro-average Dice
        dice /= (nb_classes - 1)

        # perform negative of Dice and focal transform to get Focal-Dice loss
        focal_dice = tf.math.pow((1. - dice), gamma)
        return tf.clip_by_value(1. - focal_dice, 0., 1. - epsilon)

    return focal_dice_loss


def class_categorical_focal_dice_loss(class_val, metric_name, gamma=2):
    def focal_dice_loss(target, output, epsilon=1e-10):
        smooth = 1

        # calculate Dice for each class separately and accumulate
        dice = 0
        # get prediction and target for current class only
        output1 = output[:, :, :, class_val]
        target1 = target[:, :, :, class_val]

        # calculate Dice for current class
        intersection1 = tf.reduce_sum(output1 * target1)
        union1 = tf.reduce_sum(output1 * output1) + tf.reduce_sum(target1 * target1)
        dice += (2. * intersection1 + smooth) / (union1 + smooth)

        # divide accumulated dice by class count -> macro-average Dice
        #dice /= (nb_classes - 1)

        # perform negative of Dice and focal transform to get Focal-Dice loss
        focal_dice = tf.math.pow((1. - dice), gamma)
        return tf.clip_by_value(1. - focal_dice, 0., 1. - epsilon)

        # set name of metric to be shown in tensorflow progress bar

    focal_dice_loss.__name__ = metric_name

    return focal_dice_loss

def class_dice_loss(class_val, metric_name):
    def dice_loss(y_true, y_pred):
        smooth = 1
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


def alignImagesNew(im1, im2):

    return 1





  # from: https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
def alignImages(im1, im2, max_features=1000, good_match_percent=0.05):
    # Convert images to grayscale
    #im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    #im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    im1Gray = im1[..., 0]
    im2Gray = im2[..., 2]

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    print("---")

    # Sort matches by score
    matches = list(matches)
    matches.sort(key=lambda x: x.distance, reverse=False)

    print(len(matches))
    # Remove not so good matches
    numGoodMatches = int(len(matches) * good_match_percent)
    matches = matches[:numGoodMatches]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1Gray, h, (width, height))

    return im1Reg, h
