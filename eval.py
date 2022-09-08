import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import os
from tqdm import tqdm
from utils import normalize_img, patchReader
from stats import BCa_interval_macro_metric

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
    bs = 32

    ds_name = 'dataset_070922_154010_unet'  # change manually to determine desired dataset
    model_name = 'model_070922_154010_unet'  # change manually to determine desired model
    # bs = 4  # change manually to match batch size in train.py

    ds_test_path = './output/datasets/' + ds_name + '/ds_train'
    model_path = './output/models/' + model_name

    # load generated tf test dataset
    ds_test = tf.data.experimental.load(ds_test_path, element_spec=None, compression=None, reader_func=None)
    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(bs)
    ds_test = ds_test.prefetch(1)

    # load model
    model = load_model(model_path, compile=False)  # do not compile now, already done

    # for plotting
    titles = ["image", "heatmap", "pred", "gt"]

    dice_losses = []
    dis = []
    cnt = 0
    for image, mask in tqdm(ds_test):
        #print(image.shape)
        #print(mask.shape)
        # if image.shape[0] == 1:

        pred_mask = model.predict_on_batch(image)
        threshold = (pred_mask >= 0.5).astype("float32")
        #pred_masks.append(pred_mask)
        if plot_flag:
            f, axes = plt.subplots(1, 4)  # Figure of the two corresponding TMAs
            print("image shape", image.shape)
            print("mask shape", mask.shape)
            print("pred mask shape", pred_mask.shape)
            axes[0].imshow(image)
            axes[1].imshow(np.array(pred_mask[0, ..., 1]), cmap="gray")
            axes[2].imshow(np.array(threshold[0, ..., 1]),cmap="gray")
            axes[3].imshow(np.array(mask[..., 1]), cmap="gray")
            [axes[i].set_title(title_) for i, title_ in enumerate(titles)]
            plt.show()

        dice = [dice_metric(threshold[i, ..., 1], mask[i, ..., 1]).numpy() for i in range(mask.shape[0])]
        dice_losses.extend(dice)
        #print(dice)
        #print('-----------')
        #exit()
        cnt = cnt + 1
        #if cnt == 10:
        #    break
    mu_ = np.mean(dice_losses)
    # print(mu_)
    # print(np.std(dice_losses))

    # create 95%-CI
    dice_ci, _ = BCa_interval_macro_metric(dice_losses, func=lambda x: np.mean(x), B=10000)

    print(mu_, dice_ci)


if __name__ == "__main__":
    eval_on_dataset()
