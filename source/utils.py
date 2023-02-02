import cv2
import numpy as np
import logging as log
import tensorflow as tf
import tensorflow_datasets as tfds
import h5py


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
