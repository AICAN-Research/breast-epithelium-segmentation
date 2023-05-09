import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import os
from tqdm import tqdm
from source.utils import normalize_img, patchReader
from stats import BCa_interval_macro_metric
from deep_learning_tools import network
from source.utils import class_dice_loss
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras import backend as K


# make dice function (pred, gt) -> DSC value (float)
def dice_metric(gt, pred):
    smooth = 1.  # @TODO: Should this be zero for evaluation
    axes = [0, 1, 2]
    intersection1 = tf.reduce_sum(pred * gt, axis=axes)
    union1 = tf.reduce_sum(pred * pred, axis=axes) + tf.reduce_sum(gt * gt, axis=axes)
    dice = (2. * intersection1 + smooth) / (union1 + smooth)
    return dice


def precision(y_true, y_pred):
    axes = [0, 1, 2]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=axes)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=axes)
    pr = (true_positives + K.epsilon()) / (possible_positives + K.epsilon())
    return pr


def recall(y_true, y_pred):
    axes = [0, 1, 2]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=axes)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=axes)
    rec = (true_positives + K.epsilon()) / (predicted_positives + K.epsilon())
    return rec


def f1(y_true, y_pred):
    pr = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * ((pr * rec) / (pr + rec + K.epsilon()))


def eval_on_dataset():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    plot_flag = True
    bs = 64

    ds_name = '310123_130125_level_2_psize_512_ds_4'  # change manually to determine desired dataset
    model_name = 'model_060223_122342_unet_bs_32'  # change manually to determine desired model
    # bs = 4  # change manually to match batch size in train.py

    ds_test_path = './datasets/' + ds_name + '/ds_val/'  # for testing. When predicting on training set, it is almost perfect...
    model_path = './output/models/' + model_name

    # load model
    model = load_model(model_path, compile=False)  # do not compile now, already done

    # for plotting
    titles = ["image", "heatmap", "pred", "gt"]

    #metrics = {"benign": [], "inSitu": [], "invasive": []}
    metrics = []

    cnt = 0
    for class_ in tqdm(os.listdir(ds_test_path), "Folder"):
        print("Class:", class_)

        class_path = ds_test_path + class_ + "/"

        # load generated tf test dataset
        patches = os.listdir(class_path)
        paths = np.array([class_path + x for x in patches]).astype("U400")
        ds_test = tf.data.Dataset.from_tensor_slices(paths)
        ds_test = ds_test.map(lambda x: tf.py_function(patchReader, [x], [tf.float32, tf.float32]), num_parallel_calls=8)

        # ds_test = tf.data.experimental.load(ds_test_path, element_spec=None, compression=None, reader_func=None)
        ds_test = ds_test.map(normalize_img)
        ds_test = ds_test.batch(bs, drop_remainder=True)  # @TODO: Shouldnt skip last incomplete batch
        ds_test = ds_test.prefetch(1)

        # loop over all patches, calculate metrics
        for image, mask in tqdm(ds_test, "Patch"):

            pred_mask = model.predict_on_batch(image)
            final_prediction = np.argmax(pred_mask, axis=-1, keepdims=True).astype(np.float32)

            #mask = np.argmax(mask, axis=-1).astype(np.float32)
            #values = precision_recall_fscore_support(mask.flatten(), final_prediction.flatten(),
            #                                         labels=[1, 2, 3], average="macro",
            #                                         zero_division=1)[:3]

            pr = precision(mask, final_prediction)  #@TODO: does not work for multiclass
            rec = recall(mask, final_prediction) #@TODO: does not work for multiclass
            dsc = dice_metric(mask, final_prediction) #@TODO: does not work for multiclass

            values = [pr, rec, dsc]
            values = np.array(values)

            metrics.append(values)

    metrics = np.array(metrics)

    print("Precision/Recall/DSC:")
    print("mean:", np.mean(metrics, axis=0))
    print("std:", np.std(metrics, axis=0))

    # print("Overall metric: (PR/REC/DSC)")





    exit()
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
        dice_losses.extend(dice)  # @:TODO this dice loss does not only include invasive, right, ch 1?
        cnt = cnt + 1
    mu_ = np.mean(dice_losses)

    # create 95%-CI
    dice_ci, _ = BCa_interval_macro_metric(dice_losses, func=lambda x: np.mean(x), B=10000)

    print(mu_, dice_ci)


if __name__ == "__main__":
    eval_on_dataset()
