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
    output1 = y_pred[:, :, class_val]
    gt1 = y_true[:, :, class_val]

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

    output1 = y_pred[:, :, class_val]
    gt1 = y_true[:, :, class_val]

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

    output1 = y_pred[:, :, object_]
    target1 = y_true[:, :, object_]

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

    output1 = y_pred[:, :, object_]
    target1 = y_true[:, :, object_]

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

    output1 = y_pred[:, :, object_]
    target1 = y_true[:, :, object_]

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

    output1 = y_pred[:, :, object_]
    target1 = y_true[:, :, object_]

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


def eval_patch(path, model, plot_flag):

    with h5py.File(path, "r") as f:
        image = np.asarray(f["input"])
        gt = np.asarray(f["output"])

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

    #print(pred.shape)
    #print(gt.shape)

    gt = np.argmax(gt, axis=-1).astype("uint8")
    pred = pred[..., 0].astype("uint8")

    if plot_flag:
        fig, ax = plt.subplots(1, 3, figsize=(30, 20))
        ax[0].imshow(image)
        ax[1].imshow(pred, vmin=0, vmax=3)
        ax[2].imshow(gt, vmin=0, vmax=3)
        plt.show()

    print()

    # one-hot gt and pred
    gt_back = (gt == 0).astype("float32")
    gt_inv = (gt == 1).astype("float32")
    gt_healthy = (gt == 2).astype("float32")
    gt_inSitu = (gt == 3).astype("float32")
    pred_back = (pred == 0).astype("float32")
    pred_inv = (pred == 1).astype("float32")
    pred_healthy = (pred == 2).astype("float32")
    pred_inSitu = (pred == 3).astype("float32")

    gt_one_hot = np.stack(
        [gt_back, gt_inv,
         gt_healthy, gt_inSitu], axis=-1)
    pred_one_hot = np.stack(
        [pred_back, pred_inv,
         pred_healthy, pred_inSitu], axis=-1)

    return image, gt_one_hot, pred_one_hot



def eval_on_dataset():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    plot_flag = False
    bs = 1  # @TODO: should bs match bs in train?

    path = './datasets_tma_cores/150523_180206_level_1_ds_4/ds_val/'
    model_name = './output/converted_models/model_140523_222332_agunet_bs_8_as_1_lr_0.0005_d__bl_1_br_0.2_h_0.05_s_0.2_st_1.0_mp_0_ntb_160_nvb_40.onnx'

    cylinders_paths = os.listdir(path)
    paths_ = np.array([path + x for x in cylinders_paths]).astype("U400")

    dices_1 = []  # invasive
    dices_2 = []  # benign
    dices_3 = []  # insitu
    dices_1_exist = []  # invasive
    dices_2_exist = []  # benign
    dices_3_exist = []  # insitu
    precisions_1 = 0
    precisions_2 = 0
    precisions_3 = 0
    precisions_1_exist = 0
    precisions_2_exist = 0
    precisions_3_exist = 0
    recalls_1 = 0
    recalls_2 = 0
    recalls_3 = 0
    recalls_1_exist = 0
    recalls_2_exist = 0
    recalls_3_exist = 0

    cnt = 0  # number of cylinders in total
    count_present_1_d = 0  # number of cylinders with invasive ep present
    count_present_2_d = 0  # number of cylinders with benign ep present
    count_present_3_d = 0  # number of cylinders with in situ ep present

    count_present_1_p = 0  # number of cylinders with invasive ep present
    count_present_2_p = 0  # number of cylinders with benign ep present
    count_present_3_p = 0  # number of cylinders with in situ ep present

    count_present_1_r = 0  # number of cylinders with invasive ep present
    count_present_2_r = 0  # number of cylinders with benign ep present
    count_present_3_r = 0  # number of cylinders with in situ ep present

    cnt = 0

    for path in paths_:
        image, gt, pred = eval_patch(path, model_name, plot_flag)
        fig, ax = plt.subplots(1, 3, figsize=(30, 20))
        ax[0].imshow(image)
        ax[1].imshow(pred[:, :, 1])
        ax[2].imshow(gt[:, :, 1])
        plt.show()
        continue
        class_names = ["invasive", "benign", "insitu"]
        for i, x in enumerate(class_names):
            c_dice, union_d = class_dice_(gt, pred, class_val=i + 1)
            c_precision = precision(gt, pred, object_=i + 1)
            c_recall = recall(gt, pred, object_=i + 1)
            if union_d:
                c_dice = [c_dice]
            else:
                c_dice = [c_dice.numpy()]

            c_dice_exist, count_d, union_d = class_dice_class_present(gt, pred, class_val=i + 1)
            c_precision_exist, count_p = precision_class_present(gt, pred, object_=i + 1)
            c_recall_exist, count_r = recall_class_present(gt, pred, object_=i + 1)
            if union_d:
                c_dice_exist = [c_dice_exist]
            else:
                c_dice_exist = [c_dice_exist.numpy()]

            if i == 0:
                dices_1.extend(c_dice)
                precisions_1 += c_precision
                recalls_1 += c_recall

                if count_d:
                    dices_1_exist.extend(c_dice_exist)
                    count_present_1_d += 1
                if count_p:
                    precisions_1_exist += c_precision_exist
                    count_present_1_p += 1
                if count_r:
                    recalls_1_exist += c_recall_exist
                    count_present_1_r += 1

            elif i == 1:
                dices_2.extend(c_dice)
                precisions_2 += c_precision
                recalls_2 += c_recall

                if count_d:
                    dices_2_exist.extend(c_dice_exist)
                    count_present_2_d += 1
                if count_p:
                    precisions_2_exist += c_precision_exist
                    count_present_2_p += 1
                if count_r:
                    recalls_2_exist += c_recall_exist
                    count_present_2_r += 1

            elif i == 2:
                dices_3.extend(c_dice)
                precisions_3 += c_precision
                recalls_3 += c_recall

                if count_d:
                    dices_3_exist.extend(c_dice_exist)
                    count_present_3_d += 1
                if count_p:
                    precisions_3_exist += c_precision_exist
                    count_present_3_p += 1
                if count_r:
                    recalls_3_exist += c_recall_exist
                    count_present_3_r += 1

        cnt = cnt + 1
        print("count: ", cnt)

    print("cnt: ", cnt)
    mu_1 = np.mean(dices_1)
    mu_2 = np.mean(dices_2)
    mu_3 = np.mean(dices_3)
    p_1 = precisions_1 / cnt
    p_2 = precisions_2 / cnt
    p_3 = precisions_3 / cnt
    r_1 = recalls_1 / cnt
    r_2 = recalls_2 / cnt
    r_3 = recalls_3 / cnt
    # create 95%-CI
    dice_ci_1, _ = BCa_interval_macro_metric(dices_1, func=lambda x: np.mean(x), B=10000)
    dice_ci_2, _ = BCa_interval_macro_metric(dices_2, func=lambda x: np.mean(x), B=10000)
    dice_ci_3, _ = BCa_interval_macro_metric(dices_3, func=lambda x: np.mean(x), B=10000)

    mu_1_exist = np.mean(dices_1_exist)
    mu_2_exist = np.mean(dices_2_exist)
    mu_3_exist = np.mean(dices_3_exist)
    p_1_exist = precisions_1_exist / count_present_1_p
    p_2_exist = precisions_2_exist / count_present_2_p
    p_3_exist = precisions_3_exist / count_present_3_p
    r_1_exist = recalls_1_exist / count_present_1_r
    r_2_exist = recalls_2_exist / count_present_2_r
    r_3_exist = recalls_3_exist / count_present_3_r

    print(mu_1, dice_ci_1)
    print("mean precisions invasive: ", p_1)
    print("mean recalls invasive: ", r_1)
    print()
    print(mu_2, dice_ci_2)
    print("mean precisions benign: ", p_2)
    print("mean recalls benign: ", r_2)
    print()
    print(mu_3, dice_ci_3)
    print("mean precisions inSitu: ", p_3)
    print("mean recalls inSitu: ", r_3)

    print("EXISTS: ")
    print(mu_1_exist)
    print("mean precisions invasive exist: ", p_1_exist)
    print("mean recalls invasive exist: ", r_1_exist)
    print()
    print(mu_2_exist)
    print("mean precisions benign exist: ", p_2_exist)
    print("mean recalls benign exist: ", r_2_exist)
    print()
    print(mu_3_exist)
    print("mean precisions inSitu exist: ", p_3_exist)
    print("mean recalls inSitu exist: ", r_3_exist)

    print("COUNT:")
    print(count_present_1_p, count_present_1_r, count_present_1_d)
    print(count_present_2_p, count_present_2_r, count_present_2_d)
    print(count_present_3_p, count_present_3_r, count_present_3_d)


if __name__ == "__main__":
    eval_on_dataset()