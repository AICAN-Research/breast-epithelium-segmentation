import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import os
from tqdm import tqdm
from source.utils import normalize_img, patchReader
from source.losses import class_dice_loss
from stats import BCa_interval_macro_metric
from tensorflow.python.keras import backend as K


# make dice function (pred, gt) -> DSC value (float)
def dice_metric(pred, gt):
    smooth = 1.
    intersection1 = tf.reduce_sum(pred * gt)
    union1 = tf.reduce_sum(pred * pred) + tf.reduce_sum(gt * gt)
    dice = (2. * intersection1 + smooth) / (union1 + smooth)
    return dice


def class_dice_(y_true, y_pred, class_val):
    smooth = 1.
    output1 = y_pred[:, :, :, class_val]
    gt1 = y_true[:, :, :, class_val]

    intersection1 = tf.reduce_sum(output1 * gt1)
    union1 = tf.reduce_sum(output1 * output1) + tf.reduce_sum(gt1 * gt1)
    dice = (2. * intersection1 + smooth) / (union1 + smooth)

    return dice


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
    smooth = 1.

    output1 = y_pred[:, :, :, object_]
    target1 = y_true[:, :, :, object_]

    true_positives = tf.reduce_sum(target1 * output1)
    predicted_positives = tf.reduce_sum(output1)
    if predicted_positives == 0:
        precision_ = 0
    else:
        precision_ += true_positives / predicted_positives

    return precision_


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
    smooth = 1.

    output1 = y_pred[:, :, :, object_]
    target1 = y_true[:, :, :, object_]

    true_positives = tf.reduce_sum(target1 * output1)  # TODO: consider reduce_sum vs K.sum, is there a difference in speed
    possible_positives = tf.reduce_sum(target1)
    if possible_positives == 0:
        recall_ = 0
    else:
        recall_ += true_positives / possible_positives

    return recall_
def eval_on_dataset():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    plot_flag = False
    bs = 1  # @TODO: should bs match bs in train?
    network = "agunet"

    ds_name = '200423_125554_level_2_psize_1024_ds_4'  # change manually to determine desired dataset
    #ds_name_2 = '210423_122737_wsi_level_2_psize_1024_ds_4'
    model_name = 'model_070523_211917_agunet_bs_8_as_1_lr_00001_d__bl_1_br_02_h__s_0.2_st_1.0'  # change manually to determine desired model
    model_path = './output/models/' + model_name

    ds_val_path1 = '/mnt/EncryptedSSD1/maren/datasets/' + ds_name + '/ds_val/inSitu/'
    ds_val_path2 = '/mnt/EncryptedSSD1/maren/datasets/' + ds_name + '/ds_val/benign/'
    ds_val_path3 = '/mnt/EncryptedSSD1/maren/datasets/' + ds_name + '/ds_val/invasive/'
    #ds_val_path_1 = '/mnt/EncryptedSSD1/maren/datasets/' + ds_name_2 + '/ds_val/invasive/'
    #ds_val_path_2 = '/mnt/EncryptedSSD1/maren/datasets/' + ds_name_2 + '/ds_val/benign/'
    #ds_val_path_3 = '/mnt/EncryptedSSD1/maren/datasets/' + ds_name_2 + '/ds_val/inSitu/'
    model_path = './output/models/' + model_name

    # load generated tf test dataset
    patches1 = os.listdir(ds_val_path1)
    patches2 = os.listdir(ds_val_path2)
    patches3 = os.listdir(ds_val_path3)
    #patches_1 = os.listdir(ds_val_path_1)
    #patches_2 = os.listdir(ds_val_path_2)
    #patches_3 = os.listdir(ds_val_path_3)
    paths1 = np.array([ds_val_path1 + x for x in patches1]).astype("U400")
    paths2 = np.array([ds_val_path2 + x for x in patches2]).astype("U400")
    paths3 = np.array([ds_val_path3 + x for x in patches3]).astype("U400")
    #paths_1 = np.array([ds_val_path_1 + x for x in patches_1]).astype("U400")
    #paths_2 = np.array([ds_val_path_2 + x for x in patches_2]).astype("U400")
    #paths_3 = np.array([ds_val_path_3 + x for x in patches_3]).astype("U400")

    path_append1 = np.append(paths1, paths2)
    path_append2 = np.append(path_append1, paths3)
    #path_append3 = np.append(path_append2, paths_1)
    #path_append4 = np.append(path_append3, paths_2)
    #paths = np.append(path_append4, paths_3)

    ds_val = tf.data.Dataset.from_tensor_slices(path_append2)
    ds_val = ds_val.map(lambda x: tf.py_function(patchReader, [x], [tf.float32, tf.float32]), num_parallel_calls=8)

    ds_val = ds_val.map(normalize_img)
    ds_val = ds_val.batch(bs)  # @TODO: Shouldnt skip last incomplete batch, ok when bs = 1 or should it match bs in train?
    ds_val = ds_val.prefetch(1)

    # load model
    # @TODO: get warning about custom metrics and custom_objects, should I specify the metrics I have in model.compile?
    model = load_model(model_path, compile=False)  # do not compile now, already done

    # for plotting
    titles = ["image", "heatmap", "pred", "gt"]

    dices_1 = []  # invasive
    dices_2 = []  # benign
    dices_3 = []  # insitu
    precisions_1 = 0
    precisions_2 = 0
    precisions_3 = 0
    recalls_1 = 0
    recalls_2 = 0
    recalls_3 = 0

    cnt = 0
    for image, mask in tqdm(ds_val):
        pred_mask = model.predict_on_batch(image)
        if network == "agunet":
            threshold = (pred_mask[0] >= 0.5).astype("float32")  # need pred_mask[0] with agunet to get top output
        elif network == "unet":
            threshold = (pred_mask >= 0.5).astype("float32")
        if plot_flag:
            for j in range(mask.shape[0]):
                plt.rcParams.update({'font.size': 28})
                f, axes = plt.subplots(2, 3, figsize=(30, 30))
                axes[0, 0].imshow(mask[j, ..., 1])
                axes[0, 1].imshow(mask[j, ..., 2])
                axes[0, 2].imshow(mask[j, ..., 3])
                axes[1, 0].imshow(threshold[j, ..., 1], cmap="gray")
                axes[1, 1].imshow(threshold[j, ..., 2], cmap="gray")
                axes[1, 2].imshow(threshold[j, ..., 3], cmap="gray")
                plt.show()

        class_names = ["invasive", "benign", "insitu"]
        for i, x in enumerate(class_names):
            c_dice = class_dice_(mask, threshold, class_val=i + 1)
            c_precision = precision(mask, threshold, object_=i + 1)
            c_recall = recall(mask, threshold, object_=i + 1)
            c_dice = [c_dice.numpy()]
            if i == 0:
                dices_1.extend(c_dice)
                precisions_1 += c_precision
                recalls_1 += c_recall
            elif i == 1:
                dices_2.extend(c_dice)
                precisions_2 += c_precision
                recalls_2 += c_recall
            elif i == 2:
                dices_3.extend(c_dice)
                precisions_3 += c_precision
                recalls_3 += c_recall

        cnt = cnt + 1

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


if __name__ == "__main__":
    eval_on_dataset()