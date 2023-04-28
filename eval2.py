import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import os
from tqdm import tqdm
from source.utils import normalize_img, patchReader, class_dice_loss
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
    smooth = 1.  #@TODO: should this be 1.? not 1
    output1 = y_pred[:, :, :, class_val]
    gt1 = y_true[:, :, :, class_val]

    intersection1 = tf.reduce_sum(output1 * gt1)
    union1 = tf.reduce_sum(output1 * output1) + tf.reduce_sum(gt1 * gt1)
    dice = (2. * intersection1 + smooth) / (union1 + smooth)

    # self.dice_values.assign_add((1 - dice) / 10)
    return dice


#  @TODO: look at what they do in scikit-learn for edge cases
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
    precision_ += ((true_positives + smooth) / (predicted_positives + K.epsilon() + smooth))

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

    true_positives = K.sum(K.round(K.clip(target1 * output1, 0, 1)))  # TODO: consider reduce_sum instead, is there a difference in speed
    possible_positives = K.sum(K.round(K.clip(target1, 0, 1)))
    recall_ += ((true_positives + smooth) / (possible_positives + K.epsilon() + smooth))

    return recall_
def eval_on_dataset():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    plot_flag = False
    bs = 1
    network = "agunet"

    ds_name = '200423_125554_level_2_psize_1024_ds_4'  # change manually to determine desired dataset
    ds_name_2 = '210423_122737_wsi_level_2_psize_1024_ds_4'
    model_name = 'model_260423_093216_agunet_bs_8'  # change manually to determine desired model
    # bs = 4  # change manually to match batch size in train.py

    ds_val_path1 = '/mnt/EncryptedSSD1/maren/datasets/' + ds_name + '/ds_val/invasive/'
    ds_val_path2 = '/mnt/EncryptedSSD1/maren/datasets/' + ds_name + '/ds_val/benign/'
    ds_val_path3 = '/mnt/EncryptedSSD1/maren/datasets/' + ds_name + '/ds_val/inSitu/'
    ds_val_path_1 = '/mnt/EncryptedSSD1/maren/datasets/' + ds_name_2 + '/ds_val/invasive/'
    ds_val_path_2 = '/mnt/EncryptedSSD1/maren/datasets/' + ds_name_2 + '/ds_val/benign/'
    ds_val_path_3 = '/mnt/EncryptedSSD1/maren/datasets/' + ds_name_2 + '/ds_val/inSitu/'
    model_path = './output/models/' + model_name

    # load generated tf test dataset
    patches1 = os.listdir(ds_val_path1)
    patches2 = os.listdir(ds_val_path2)
    patches3 = os.listdir(ds_val_path3)
    patches_1 = os.listdir(ds_val_path_1)
    patches_2 = os.listdir(ds_val_path_2)
    patches_3 = os.listdir(ds_val_path_3)
    paths1 = np.array([ds_val_path1 + x for x in patches1]).astype("U400")
    paths2 = np.array([ds_val_path2 + x for x in patches2]).astype("U400")
    paths3 = np.array([ds_val_path3 + x for x in patches3]).astype("U400")
    paths_1 = np.array([ds_val_path_1 + x for x in patches_1]).astype("U400")
    paths_2 = np.array([ds_val_path_2 + x for x in patches_2]).astype("U400")
    paths_3 = np.array([ds_val_path_3 + x for x in patches_3]).astype("U400")

    path_append1 = np.append(paths1, paths2)
    path_append2 = np.append(path_append1, paths3)
    path_append3 = np.append(path_append2, paths_1)
    path_append4 = np.append(path_append3, paths_2)
    paths = np.append(path_append4, paths_3)

    ds_val = tf.data.Dataset.from_tensor_slices(paths)
    ds_val = ds_val.map(lambda x: tf.py_function(patchReader, [x], [tf.float32, tf.float32]), num_parallel_calls=8)

    # ds_test = tf.data.experimental.load(ds_test_path, element_spec=None, compression=None, reader_func=None)
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
                axes[0, 0].imshow(image[j])
                axes[0, 1].imshow(mask[j, ..., 1])
                axes[0, 2].imshow(mask[j, ..., 2])
                axes[1, 0].imshow(np.array(pred_mask[0][j, ..., 2]))
                axes[1, 1].imshow(np.array(threshold[j, ..., 2]))
                axes[1, 2].imshow(np.array(pred_mask[0][j, ..., 3]))
                plt.show()

        class_names = ["invasive", "benign", "insitu"]
        for i, x in enumerate(class_names):
            c_dice = class_dice_(mask, threshold, class_val=i + 1)
            c_precision = precision(mask, threshold, object_=i + 1).numpy()
            c_recall = recall(mask, threshold, object_=i + 1).numpy()
            c_dice = [c_dice.numpy()]
            if i == 1:
                dices_1.extend(c_dice)
                precisions_1 += c_precision
                recalls_1 += c_recall
            elif i == 2:
                dices_2.extend(c_dice)
                precisions_2 += c_precision
                recalls_2 += c_recall
            else:
                dices_3.extend(c_dice)
                precisions_3 += c_precision
                recalls_3 += c_recall
        #dice2 = [dice_metric(threshold[i, ..., 2], mask[i, ..., 2]).numpy() for i in range(mask.shape[0])]  # now bs = 1, unnecessary with for i in ...
        #dice_ = dice(mask, threshold, 4, use_background=False, dims=2, epsilon=1e-10)
        #dice_ = [dice_.numpy()]
        #recall_ = recall(mask, threshold, nb_classes=4).numpy()
        #precision_ = precision(mask, threshold, nb_classes=4).numpy()

        #dices_.extend(dice_)  # this dice loss does not only include invasive, right, ch 1?
        #precisions += precision_
        #recalls += recall_

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