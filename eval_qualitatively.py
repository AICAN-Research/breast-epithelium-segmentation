"""
Script to evaluate segmentations qualitatively. Plotting ground truth and segmentation on cylinder level.
"""
import os
import numpy as np
import h5py
import multiprocessing as mp
import matplotlib.pyplot as plt


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
    generator = fast.PatchGenerator.create(2048, 2048, overlapPercent=0.3).connect(0, data_fast)
    padder = PadderPO.create(width=2048, height=2048).connect(generator)
    network = fast.NeuralNetwork.create(modelFilename=model, inferenceEngine="OpenVINO", scaleFactor=0.00392156862) \
        .connect(padder)
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

    plt.rcParams.update({'font.size': 28})
    f, axes = plt.subplots(2, 2, figsize=(30, 30))
    axes[0, 0].imshow(image[1500:2500, 1500:2500, :])
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


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    path = './datasets_tma_cores/230623_141305_level_1_ds_4/ds_test_external/'
    model_name = './output/converted_models/model_030623_224255_agunet_bs_8_as_1_lr_0.0005_d_None_bl_1_br_0.3_h_0.05_s_0.3_st_1.0_fl_1.0_rt_1.0_mp_0_ntb_160_nvb_40.onnx'

    cylinders_paths = os.listdir(path)
    paths_ = np.array([path + x for x in cylinders_paths]).astype("U400")

    for path in paths_:
        inputs_ = [[path, model_name]]
        p = mp.Pool(1)
        p.map(eval_wrapper, inputs_)
        p.terminate()
        p.join()
        del p, inputs_


