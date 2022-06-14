import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ds_name = 'dataset_130622_143222_unet' # change manually to determine desired dataset
model_name = 'model_130622_143222_unet' # change manually to determine desired model
#bs = 4  # change manually to match batch size in train.py

ds_test_path = './output/datasets/' + ds_name + '/ds_test'
model_path = './output/models/' + model_name

ds_test = tf.data.experimental.load(
    ds_test_path, element_spec=None, compression=None, reader_func=None
)
model = load_model(model_path, compile=False)  # do not compile now, already done

titles = ["image", "heatmap", "pred", "gt"]

# make dice function (pred, gt) -> DSC value (float)
def dice_loss(pred, gt, epsilon=1e-10):
    smooth = 1.
    dice = 0
    intersection1 = tf.reduce_sum(pred * gt)
    union1 = tf.reduce_sum(pred * pred) + tf.reduce_sum(gt * gt)
    dice += (2. * intersection1 + smooth) / (union1 + smooth)

    dice2 = 1 - dice
    return tf.clip_by_value(1. - dice, 0., 1. - epsilon), dice2

def pred():
    dice_losses = []
    dis = []
    cnt = 0
    for image, mask in ds_test:
        #print(image.shape)
        #print(mask.shape)
        pred_mask = model.predict(np.expand_dims(image, axis = 0))
        threshold = (pred_mask >= 0.5).astype("float32")
        #pred_masks.append(pred_mask)
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

        loss,di = dice_loss(threshold, mask)
        dice_losses.append(loss)
        dis.append(di)
        print(loss)
        print(di)
        print('-----------')
        #exit()
        cnt = cnt + 1
        if cnt == 5:
            exit()
    mean = np.mean(dice_losses)
    print(mean)
    m_di = np.mean(dis)
    print(m_di)
pred()