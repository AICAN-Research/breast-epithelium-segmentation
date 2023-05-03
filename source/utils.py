import cv2
import numpy as np
import logging as log
import tensorflow as tf
import tensorflow_datasets as tfds
import h5py
from skimage.color import rgb2gray
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1, optical_flow_ilk
import pyelastix


# focal dice loss to focus on "difficult classes"
# implemented by André Pedersen:
def categorical_focal_dice_loss(nb_classes, use_background=False, gamma=2):
    def focal_dice_loss(target, output, epsilon=1e-10):
        smooth = 1.

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
        smooth = 1.

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


# from: https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
# and https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html
# and https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
# https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
def alignImages(im1, im2, method, max_features=70000, good_match_percent=0.10):
    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    print("im1 gray shape: ", im1_gray.shape)
    print("im2 gray shape: ", im2_gray.shape)

    if method == "orb":
        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(nfeatures=max_features) #, scaleFactor=1.8, patchSize=256, edgeThreshold=256)

        kp = orb.detect(im1_gray, None)
        kp1, des1 = orb.compute(im1_gray, kp)

        kp = orb.detect(im2_gray, None)
        kp2, des2 = orb.compute(im2_gray, kp)

        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)  # cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        #matcher = cv2. BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)  # @TODO: what is the difference between this and the one above?
        matches = matcher.match(des1, des2, None)
        # matches = matcher.radiusMatch(des1, des2, 500, None)

        # Sort matches by score
        matches = list(matches)
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * good_match_percent)
        matches = matches[:numGoodMatches]

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

    # Find homography
    h, _ = cv2.findHomography(points1, points2, cv2.RANSAC) #, ransacReprojThreshold=2)

    # Use homography
    height, width, channels = im2.shape
    im1_reg = cv2.warpPerspective(im1, h, (width, height))

    #im1_key = cv2.drawKeypoints()
    #im2_key = cv2.drawKeypoints()

    return im1_reg, h, height, width


# https://scikit-image.org/docs/stable/auto_examples/registration/plot_opticalflow.html#sphx-glr-auto-examples-registration-plot-opticalflow-py
# https://scikit-image.org/docs/stable/api/skimage.registration.html#skimage.registration.optical_flow_tvl1
def align_optical_flow(im1, im2):
    # --- Convert the images to gray level: color is not supported.
    image0 = rgb2gray(im1)
    image1 = rgb2gray(im2)

    # --- Compute the optical flow
    v, u = optical_flow_ilk(image0, image1, num_warp=10, radius=100)  # iLK faster but less robust than TVL1

    # --- Use the estimated optical flow for registration
    nr, nc = image0.shape
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
    image1_warp = warp(image1, np.array([row_coords + v, col_coords + u]), mode='constant')

    return image1_warp, row_coords, col_coords, v, u


class ImageRegistrationOpticalFlow:
    def __init__(self):
        self.v = None
        self.u = None
        self.nr = None
        self.nc = None

    def fit(self, im1, im2):
        # --- Convert the images to gray level: color is not supported.
        image0 = rgb2gray(im1)
        image1 = rgb2gray(im2)

        # --- Compute the optical flow
        self.v, self.u = optical_flow_ilk(image0, image1, num_warp=10, radius=100)  # iLK faster but less acc than TVL1

        self.nr, self.nc = image0.shape

    def transform(self, image):
        image = rgb2gray(image)

        # --- Use the estimated optical flow for registration
        row_coords, col_coords = np.meshgrid(np.arange(self.nr), np.arange(self.nc), indexing='ij')
        return warp(image, np.array([row_coords + self.v, col_coords + self.u]), mode='constant')


def align_pyelastix(im1, im2):
    # Get params and change a few values
    params = pyelastix.get_default_params()
    params.MaximumNumberOfIterations = 200
    params.FinalGridSpacingInVoxels = 10

    # Apply the registration (im1 and im2 can be 2D or 3D)
    im1_deformed, field = pyelastix.register(im1, im2, params)

    return im1_deformed
