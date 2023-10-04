"""
Script to evaluate segmentations qualitatively. Plotting ground truth and segmentation on cylinder level.
"""
import os
import numpy as np
import h5py
import multiprocessing as mp
import matplotlib.pyplot as plt
import tensorflow as tf

def class_dice_(y_true, y_pred, class_val):
    output1 = y_pred[..., class_val]
    gt1 = y_true[..., class_val]

    intersection1 = tf.reduce_sum(output1 * gt1)
    union1 = tf.reduce_sum(output1 * output1) + tf.reduce_sum(
        gt1 * gt1)
    if union1 == 0:
        dice = 1.
        dice_u = True
    else:
        dice = (2. * intersection1) / union1
        dice_u = False

    return dice, dice_u

def eval_wrapper(some_inputs_):
    return eval_patch(*some_inputs_)


def eval_patch(path, model):
    import fast

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

    with h5py.File(path, "r") as f:
        image = np.asarray(f["input"])
        gt = np.asarray(f["output"])

    image = image.astype("uint8")

    data_fast = fast.Image.createFromArray(image)
    generator = fast.PatchGenerator.create(2048, 2048, overlapPercent=0.3, maskThreshold=0.02).connect(0, data_fast)
    padder = PadderPO.create(width=2048, height=2048).connect(generator)
    patch_resizer = fast.ImageResizer.create(width=1024, height=1024, useInterpolation=False, preserveAspectRatio=True) \
        .connect(padder)  # @TODO: this the NeuralNetwork PO should do, but adding it to the pipeline surprisingly yields different results
    network = fast.NeuralNetwork.create(modelFilename=model, inferenceEngine="OpenVINO", scaleFactor=0.00392156862) \
        .connect(patch_resizer)
    converter = fast.TensorToSegmentation.create(threshold=0.5).connect(0, network, 5)
    resizer = fast.ImageResizer.create(width=2048, height=2048, useInterpolation=False, preserveAspectRatio=True) \
        .connect(converter)
    stitcher = fast.PatchStitcher.create().connect(resizer)

    for _ in fast.DataStream(stitcher):
        pass

    pred = stitcher.runAndGetOutputData()
    pred = np.asarray(pred)

    del data_fast, generator, padder, network, converter, resizer, stitcher

    gt_shape = gt.shape
    pred = pred[:gt_shape[0], :gt_shape[1]]
    gt = np.argmax(gt, axis=-1).astype("uint8")
    pred = pred[..., 0].astype("uint8")

    # one-hot gt and pred
    gt_back = (gt == 0).astype("float32")
    gt_inv = (gt == 1).astype("float32")
    gt_healthy = (gt == 2).astype("float32")
    gt_inSitu = (gt == 3).astype("float32")
    pred_back = (pred == 0).astype("float32")
    pred_inv = (pred == 1).astype("float32")
    pred_healthy = (pred == 2).astype("float32")
    pred_inSitu = (pred == 3).astype("float32")

    gt = np.stack(
        [gt_back, gt_inv,
         gt_healthy, gt_inSitu], axis=-1)
    pred = np.stack(
        [pred_back, pred_inv,
         pred_healthy, pred_inSitu], axis=-1)

    tp_ = gt + 1  # 0 -> 1, 1 -> 2
    pred_ = pred * 2  # 0 -> 0, 1 -> 2
    tp = (tp_ == pred_).astype("float32")  # true positives  (could also do tf.reduce_sum(tp * pred)
    fn = (gt - tp) * 2  # 1 -> 2
    fp = (pred - tp) * 3  # 1 -> 3
    inv = tp[:, :, 1] + fn[:, :, 1] + fp[:, :, 1]
    ben = tp[:, :, 2] + fn[:, :, 2] + fp[:, :, 2]
    ins = tp[:, :, 3] + fn[:, :, 3] + fp[:, :, 3]
    print(inv.shape)

    if True:
        plt.rcParams.update({'font.size': 28})
        f, axes = plt.subplots(2, 2, figsize=(30, 30))
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Image")
        axes[0, 1].imshow(ben[1500:2500, 1500:2500])
        axes[0, 1].set_title("Benign: tp = blue, fp = yellow, fn = green")
        axes[1, 0].imshow(image[1500:2500, 1500:2500, :])
        axes[1, 0].imshow(gt[1500:2500, 1500:2500, 2], alpha=0.5)
        axes[1, 0].set_title("Benign: ground truth on image")
        axes[1, 1].imshow(image[1500:2500, 1500:2500, :])
        axes[1, 1].imshow(pred[1500:2500, 1500:2500, 2], alpha=0.5)
        axes[1, 1].set_title("Benign: prediction on image")
        plt.show()

    dice_scores = []
    class_names = ["invasive", "benign", "insitu"]
    for i, x in enumerate(class_names):
        c_dice, union_d = class_dice_(gt, pred, class_val=i + 1)
        dice_scores.append(c_dice)

    if True:
        plt.rcParams.update({'font.size': 28})
        f, axes = plt.subplots(2, 2, figsize=(30, 30))
        axes[0, 0].imshow(image)
        axes[0, 0].imshow(gt[:, :, 1], cmap="gray", alpha=0.5)
        axes[0, 0].set_title("Ground truth, invasive")
        axes[0, 1].imshow(image)
        axes[0, 1].imshow(pred[:, :, 1], cmap="gray", alpha=0.5)
        axes[0, 1].set_title("Prediction, invasive, Dice score: " + str(np.asarray(dice_scores[0])))
        axes[1, 0].imshow(image)
        axes[1, 0].imshow(gt[:, :, 2], cmap="gray", alpha=0.5)
        axes[1, 0].set_title("Ground truth, benign")
        axes[1, 1].imshow(image)
        axes[1, 1].imshow(pred[:, :, 2], cmap="gray", alpha=0.5)
        axes[1, 1].set_title("Prediction, benign, Dice score: " + str(np.asarray(dice_scores[1])))
        plt.show()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    path = '/path/to/dataset'
    model_name = '/path/to/model'

    cylinders_paths = os.listdir(path)
    paths_ = np.array([path + x for x in cylinders_paths]).astype("U400")

    for path in paths_:
        if "file_name" in path:  # if specific file is wanted
            inputs_ = [[path, model_name]]
            p = mp.Pool(1)
            p.map(eval_wrapper, inputs_)
            p.terminate()
            p.join()
            del p, inputs_


