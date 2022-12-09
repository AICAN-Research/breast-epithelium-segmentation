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
    plot_flag = True
    bs = 16

    ds_name = 'dataset_251122_103606_unet_bs_16'  # change manually to determine desired dataset
    model_name = 'model_251122_103606_unet_bs_16'  # change manually to determine desired model
    # bs = 4  # change manually to match batch size in train.py

    ds_test_path = './output/datasets/' + ds_name + '/ds_test'
    model_path = './output/models/' + model_name

    # load generated tf test dataset
    ds_test = tf.data.experimental.load(ds_test_path, element_spec=None, compression=None, reader_func=None)
    ds_test = ds_test.map(normalize_img)
    ds_test = ds_test.batch(bs)
    ds_test = ds_test.prefetch(1)

    # load model
    model = load_model(model_path, compile=False)  # do not compile now, already done

    # for plotting
    titles = ["image", "heatmap", "pred", "gt"]

    dice_losses = []
    cnt = 0
    for image, mask in tqdm(ds_test):

        pred_mask = model.predict_on_batch(image)
        threshold = (pred_mask >= 0.5).astype("float32")
        print("pred mask shape", pred_mask.shape)

        if plot_flag:
            for j in range(mask.shape[0]):
                #plt.rcParams.update({'font.size': 28})
                f, axes = plt.subplots(4, 3, figsize=(20, 20))  # Figure of the two corresponding TMAs
                axes[0, 0].imshow(image[j])
                axes[0, 0].set_title("image")

                axes[1, 0].imshow(np.array(pred_mask[j, ..., 1]), cmap="gray")
                axes[1, 0].set_title("heatmap invasive")  # invasive
                axes[1, 1].imshow(np.array(threshold[j, ..., 1]),cmap="gray")
                axes[1, 1].set_title("pred invasive")
                axes[1, 2].imshow(np.array(mask[j,..., 1]), cmap="gray")
                axes[1, 2].set_title("gt invasive")

                axes[2, 0].imshow(np.array(pred_mask[j, ..., 2]), cmap="gray")
                axes[2, 0].set_title("heatmap healthy")  # healthy
                axes[2, 1].imshow(np.array(threshold[j, ..., 2]),cmap="gray")
                axes[2, 1].set_title("pred healthy")
                axes[2, 2].imshow(np.array(mask[j,..., 2]), cmap="gray")
                axes[2, 2].set_title("gt healthy")

                axes[3, 0].imshow(np.array(pred_mask[j, ..., 3]), cmap="gray")
                axes[3, 0].set_title("heatmap in situ")  # in situ lesions
                axes[3, 1].imshow(np.array(threshold[j, ..., 3]),cmap="gray")
                axes[3, 1].set_title("pred in situ")
                axes[3, 2].imshow(np.array(mask[j,..., 3]), cmap="gray")
                axes[3, 2].set_title("gt in situ")

                #[axes[i, j].set_title(title_) for i, title_ in enumerate(titles)]
                plt.show()

        dice = [dice_metric(threshold[i, ..., 1], mask[i, ..., 1]).numpy() for i in range(mask.shape[0])]
        dice_losses.extend(dice)  # this dice loss does not only include invasive, right, ch 1?
        cnt = cnt + 1
    mu_ = np.mean(dice_losses)

    # create 95%-CI
    dice_ci, _ = BCa_interval_macro_metric(dice_losses, func=lambda x: np.mean(x), B=10000)

    print(mu_, dice_ci)


if __name__ == "__main__":
    eval_on_dataset()
