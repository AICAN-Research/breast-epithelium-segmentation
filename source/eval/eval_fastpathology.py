"""
Script for evaluating qualitatively the segmentations generated in FastPathology
Plotting ground truth and segmentation on cylinder level.
Specify which exact cylinder to evaluate
"""
import fast
import cv2
from skimage.exposure import equalize_hist
from skimage.registration import phase_cross_correlation
from scipy import ndimage as ndi
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

def cut_image(shift_h, shift_w, shape_h, shape_w):
    """
    Cutting registered images to remove padded areas due to shift
    :param shift_h: shift of moving image (height)
    :param shift_w: shift of moving image (width)
    :param shape_h: height moving image
    :param shape_w:  width moving image
    :return: start and stop height and width when cutting registered images
    """
    start_h = shift_h
    start_w = shift_w
    stop_h = shape_h
    stop_w = shape_w
    if shift_h < 0:
        start_h = 0
        stop_h = shape_h - np.abs(shift_h)
    if shift_w < 0:
        start_w = 0
        stop_w = shape_w - np.abs(shift_w)
    return start_h, start_w, stop_h, stop_w

def eval_wrapper(some_inputs_):
    return eval_patch(*some_inputs_)


def eval_patch(path, pred):

    with h5py.File(path, "r") as f:
        image = np.asarray(f["input"])
        gt = np.asarray(f["output"])

    pred = cv2.resize(pred.astype("uint8"), gt.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    gt = np.argmax(gt, axis=-1).astype("uint8")

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
    seg_path = '/path/to/fastPathology-project/results/'  # path to tiff-result file in fp project
    he_path = '/path/to/file'  # path to he vsi to evaluate
    ck_path = '/path/to/file'  # path to ck vsi to evaluate
    path = '/path/to/image-gt-file'  # path to corresponding hdf5 image and gt

    #level = 1
    dist_limit = 4000
    level = 2

    importer_he = fast.WholeSlideImageImporter.create(
        he_path)  # path to HE image
    importer_ck = fast.WholeSlideImageImporter.create(
        ck_path)  # path to HE image
    importer_seg = fast.TIFFImagePyramidImporter.create(
        seg_path)  # path to annotated image

    seg = importer_seg.runAndGetOutputData()
    access_seg = seg.getAccess(fast.ACCESS_READ)
    annot_he = importer_he.runAndGetOutputData()

    # get HE TMA cores
    extractor = fast.TissueMicroArrayExtractor.create(level=level).connect(importer_he)
    he_tmas = []
    he_stream = fast.DataStream(extractor)
    for j, tma in enumerate(he_stream):
        he_tmas.append(tma)
        if j == -1:
            break
    del extractor, he_stream

    # get CK TMA cores
    extractor = fast.TissueMicroArrayExtractor.create(level=level).connect(importer_ck)
    ck_tmas = []
    ck_stream = fast.DataStream(extractor)
    for j, tma in enumerate(ck_stream):
        ck_tmas.append(tma)
        if j == -1:
            break
    del extractor, ck_stream

    for he_counter in range(len(he_tmas)):
        for ck_counter in range(len(ck_tmas)):
            if he_counter == 7:  # this number must match the cylinder you want to look at when extracting from slide with TMA extractor
                he_tma = he_tmas[he_counter]
                ck_tma = ck_tmas[ck_counter]
                position_he = he_tma.getTransform().getTranslation()  # position of he_tma at level 0
                position_ck = ck_tma.getTransform().getTranslation()  # position of IHC TMA at position IHC_counter.

                position_he_x = position_he[0][0]
                position_he_y = position_he[1][0]
                position_ck_x = position_ck[0][0]
                position_ck_y = position_ck[1][0]

                dist_x = position_he_x - position_ck_x
                dist_y = position_he_y - position_ck_y

                if np.abs(dist_x) < dist_limit and np.abs(dist_y) < dist_limit:  # if positions are close we have a pair

                    try:
                        he_tma = np.asarray(he_tma)
                        ck_tma = np.asarray(ck_tma)
                    except RuntimeError as e:
                        print(e)
                        continue
                    height_he, width_he, _ = he_tma.shape

                    downsample_factor = 4
                    shapes_ck_tma = ck_tma.shape
                    shapes_he_tma = he_tma.shape

                    longest_height = max([shapes_ck_tma[0], shapes_he_tma[0]])
                    longest_width = max([shapes_ck_tma[1], shapes_he_tma[1]])

                    ck_tma_padded = np.ones((longest_height, longest_width, 3), dtype="uint8") * 255
                    he_tma_padded = np.ones((longest_height, longest_width, 3), dtype="uint8") * 255

                    # insert ck and he tma in padded array
                    ck_tma_padded[:ck_tma.shape[0], :ck_tma.shape[1]] = ck_tma
                    he_tma_padded[:he_tma.shape[0], :he_tma.shape[1]] = he_tma

                    curr_shape = ck_tma_padded.shape[:2]
                    ck_tma_padded_ds = cv2.resize(ck_tma_padded,
                                                  np.round(np.array(curr_shape) / downsample_factor).astype("int32"),
                                                  interpolation=cv2.INTER_NEAREST)
                    he_tma_padded_ds = cv2.resize(he_tma_padded,
                                                  np.round(np.array(curr_shape) / downsample_factor).astype("int32"),
                                                  interpolation=cv2.INTER_NEAREST)

                    # detect shift between ck and he, histogram equalization for better shift in tmas with few
                    # distinct landmarks
                    ck_tma_padded_ds_histeq = equalize_hist(ck_tma_padded_ds)
                    shifts, reg_error, phase_diff = phase_cross_correlation(he_tma_padded_ds, ck_tma_padded_ds_histeq,
                                                                            return_error=True)
                    shifts[2] = 0  # set z-axis to zero (should be from beginning)
                    # scale shifts back and apply to original resolution
                    shifts = (np.round(downsample_factor * shifts)).astype("int32")
                    ck_tma_padded_shifted = ndi.shift(ck_tma_padded, shifts, order=0, mode="constant", cval=255,
                                                      prefilter=False)

                    # cut images to remove areas added/removed by shift
                    start_h, start_w, stop_h, stop_w = cut_image(shifts[0], shifts[1], he_tma_padded.shape[0],
                                                                 he_tma_padded.shape[1])
                    he_tma_padded = he_tma_padded[int(start_h):int(stop_h), int(start_w):int(stop_w), :]
                    ck_tma_padded_shifted = ck_tma_padded_shifted[int(start_h):int(stop_h), int(start_w):int(stop_w), :]

                    position_he_x /= (2 ** level)
                    position_he_y /= (2 ** level)
                    try:
                        #seg_annot = access_seg.getPatchAsImage(int(0), int(position_he_x/2), int(position_he_y/2), int(width_he/2),
                        #                                           int(height_he/2), False)  # divide by two since the tiff image from fastpathology does not have a 40x and 20x. (its a pyramidal image from 10x downwards)
                        seg_annot = access_seg.getPatchAsImage(int(0), int(position_he_x), int(position_he_y),
                                                               int(width_he), int(height_he), False)
                    except RuntimeError as e:
                        print(e)
                        continue

                    seg_annot = np.asarray(seg_annot)
                    print(seg_annot.shape)

                    seg_tma_padded = np.zeros((longest_height, longest_width, 1), dtype="uint8")

                    # insert seg tma in padded array
                    seg_tma_padded[:seg_annot.shape[0], :seg_annot.shape[1]] = seg_annot
                    seg_tma_padded_ = seg_tma_padded[int(start_h):int(stop_h), int(start_w):int(stop_w), :]

                    plt.rcParams.update({'font.size': 28})
                    f, axes = plt.subplots(2, 2, figsize=(30, 30))
                    axes[0, 0].imshow(seg_tma_padded)
                    axes[0, 1].imshow(he_tma)
                    axes[1, 0].imshow(seg_annot)
                    axes[1, 1].imshow(seg_tma_padded_)
                    plt.show()

                    inputs_ = [[path, seg_tma_padded_]]
                    p = mp.Pool(1)
                    p.map(eval_wrapper, inputs_)
                    p.terminate()
                    p.join()
                    del p, inputs_

