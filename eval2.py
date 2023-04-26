import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import os
from tqdm import tqdm
from utils import normalize_img, patchReader
from stats import BCa_interval_macro_metric
from source.metrics import precision, recall


# make dice function (pred, gt) -> DSC value (float)
def dice_metric(pred, gt):
    smooth = 1.
    intersection1 = tf.reduce_sum(pred * gt)
    union1 = tf.reduce_sum(pred * pred) + tf.reduce_sum(gt * gt)
    dice = (2. * intersection1 + smooth) / (union1 + smooth)
    return dice


def eval_on_dataset():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    plot_flag = False
    bs = 1

    ds_name = '200423_125554_level_2_psize_1024_ds_4'  # change manually to determine desired dataset
    ds_name_2 = '210423_122737_wsi_level_2_psize_1024_ds_4'
    model_name = 'model_210423_144951_agunet_bs_8'  # change manually to determine desired model
    # bs = 4  # change manually to match batch size in train.py

    ds_val_path = '/mnt/EncryptedSSD1/maren/datasets/' + ds_name + '/ds_val/benign/'
    ds_val_path_2 = '/mnt/EncryptedSSD1/maren/datasets/' + ds_name_2 + '/ds_val/benign/'
    model_path = './output/models/' + model_name

    # load generated tf test dataset
    patches = os.listdir(ds_val_path)
    patches_2 = os.listdir(ds_val_path_2)
    paths = np.array([ds_val_path + x for x in patches]).astype("U400")
    paths_2 = np.array([ds_val_path_2 + x for x in patches_2]).astype("U400")
    paths = np.append(paths, paths_2)  # add paths_2 to paths
    ds_val = tf.data.Dataset.from_tensor_slices(paths)
    ds_val = ds_val.map(lambda x: tf.py_function(patchReader, [x], [tf.float32, tf.float32]), num_parallel_calls=8)

    # ds_test = tf.data.experimental.load(ds_test_path, element_spec=None, compression=None, reader_func=None)
    ds_val = ds_val.map(normalize_img)
    ds_val = ds_val.batch(bs)  # @TODO: Shouldnt skip last incomplete batch, ok when bs = 1 or should it match bs in train?
    ds_val = ds_val.prefetch(1)

    # load model
    model = load_model(model_path, compile=False)  # do not compile now, already done

    # for plotting
    titles = ["image", "heatmap", "pred", "gt"]

    dice_losses = []
    precisions = 0
    recalls = 0

    cnt = 0
    for image, mask in tqdm(ds_val):
        pred_mask = model.predict_on_batch(image)
        threshold = (pred_mask[0] >= 0.5).astype("float32")
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

        dice = [dice_metric(threshold[i, ..., 1], mask[i, ..., 1]).numpy() for i in range(mask.shape[0])]
        recall_ = recall(mask, threshold, nb_classes=4).numpy()
        precision_ = precision(mask, threshold, nb_classes=4).numpy()

        dice_losses.extend(dice)  # this dice loss does not only include invasive, right, ch 1?
        precisions += precision_
        recalls += recall_

        cnt = cnt + 1

    mu_ = np.mean(dice_losses)
    p_ = precisions / cnt
    r_ = recalls / cnt
    # create 95%-CI
    dice_ci, _ = BCa_interval_macro_metric(dice_losses, func=lambda x: np.mean(x), B=10000)

    print(mu_, dice_ci)
    print("mean precisions: ", p_)
    print("mean recalls: ", r_)


if __name__ == "__main__":
    eval_on_dataset()