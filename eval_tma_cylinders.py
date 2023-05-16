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


class PadderPO(fast.PythonProcessObject):
    def __init__(self, width=1024, height=1024):
        super().__init__()
        self.createInputPort(0)
        self.createOutputPort(0)

        self.height = height
        self.width = width

    def execute(self):
        # Get image and invert it with numpy
        image = self.getInputData()
        np_image = np.asarray(image)
        tmp = np.zeros((self.height, self.width, 3), dtype="uint8")
        shapes = np_image.shape
        tmp[:shapes[0], :shapes[1]] = np_image

        # Create new fast image and add as output
        new_output_image = fast.Image.createFromArray(tmp)
        new_output_image.setSpacing(image.getSpacing())
        self.addOutputData(0, new_output_image)


def eval_patch(path, model):

    with h5py.File(path, "r") as f:
        image = np.asarray(f["input"])
        gt = np.asarray(f["output"])

    # gt = np.argmax(gt, axis=-1)  # @TODO: this I shouldn't keep, right? removes one-hot

    image = image.astype("uint8")

    data_fast = fast.Image.createFromArray(image)
    generator = fast.PatchGenerator.create(2048, 2048, overlapPercent=0.3).connect(0, data_fast)
    padder = PadderPO.create(width=2048, height=2048).connect(generator)
    network = fast.NeuralNetwork.create(modelFilename=model, inferenceEngine="OpenVINO", scaleFactor=0.00392156862)\
        .connect(padder)
    converter = fast.TensorToSegmentation.create(threshold=0.5).connect(0, network, 7)
    resizer = fast.ImageResizer.create(width=2048, height=2048, useInterpolation=False, preserveAspectRatio=True)\
        .connect(converter)
    stitcher = fast.PatchStitcher.create().connect(resizer)

    for _ in fast.DataStream(stitcher):
        pass

    pred = stitcher.runAndGetOutputData()

    pred = np.asarray(pred)

    gt_shape = gt.shape
    pred = pred[:gt_shape[0], :gt_shape[1]]

    print(pred.shape)
    print(gt.shape)

    gt = np.argmax(gt, axis=-1).astype("uint8")
    pred = pred[..., 0].astype("uint8")



    fig, ax = plt.subplots(1, 3, figsize=(30, 20))
    ax[0].imshow(image)
    ax[1].imshow(pred, vmin=0, vmax=3)
    ax[2].imshow(gt, vmin=0, vmax=3)
    plt.show()

    print()



def eval_on_dataset():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    plot_flag = False
    bs = 1  # @TODO: should bs match bs in train?

    path = './datasets_tma_cores/150523_180206_level_1_ds_4/ds_val/'
    model_name = './output/converted_models/model_120523_094220_agunet_bs_8_as_1_lr_0.0005_d__bl_1_br_0.2_h__s_0.2_st_1.0_mp_0_ntb_160_nvb_40.onnx'

    cylinders_paths = os.listdir(path)
    paths_ = np.array([path + x for x in cylinders_paths]).astype("U400")

    for path in paths_:
        eval_patch(path, model_name)


    ds_val = tf.data.Dataset.from_tensor_slices(paths_)
    ds_val = ds_val.map(lambda x: tf.py_function(eval_patch, [x], [tf.float32, tf.float32]), num_parallel_calls=8)
    ds_val = ds_val.batch(bs)
    ds_val.prefetch(1)

    for image, mask in tqdm(ds_val):
        print(image.shape)
        print(mask.shape)



if __name__ == "__main__":
    eval_on_dataset()