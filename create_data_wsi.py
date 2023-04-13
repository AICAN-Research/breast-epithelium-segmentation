"""
Script to create patches from whole slide images
Annotations from QuPath of which areas to exclude (from he image)
Put patches in train/val set
"""
from datetime import datetime, date
import h5py
import numpy as np
import fast
import cv2
from skimage.registration import phase_cross_correlation
from scipy import ndimage as ndi
from skimage.exposure import equalize_hist
import matplotlib.pyplot as plt
import os
from source.utils import alignImages, align_optical_flow, align_pyelastix, ImageRegistrationOpticalFlow
from skimage.exposure import equalize_hist
from skimage.transform import warp
from skimage.color import rgb2gray


def minmax(x):
    """
    normalizes intensities to range float [0, 1]
    :param x: intensity image
    :return: normalized x
    """
    # @TODO: Sometimes get error: invalid value encountered (x/=np.argmax(x)).
    #  Is it possible I get one for both argmax and argmin
    x = x.astype("float32")
    if np.amax(x) > 0:
        x -= np.amin(x)
        x /= np.amax(x)
    return x


def create_dataset(he_path, ck_path, roi_annot_path, annot_path, dab_path, dataset_path, level, patch_size, ds_factor,
                   overlap, tissue_level):
    importer_he = fast.WholeSlideImageImporter.create(
        he_path)  # path to CK image
    importer_ck = fast.WholeSlideImageImporter.create(
        ck_path)  # path to CK image
    importer_roi_annot = fast.TIFFImagePyramidImporter.create(
        roi_annot_path)  # path to annotated image or areas to keep
    importer_dab = fast.TIFFImagePyramidImporter.create(
        dab_path)  # path to dab image
    importer_annot = fast.TIFFImagePyramidImporter.create(
        annot_path)  # path to manual annotations of in situ lesions and benign epithelium

    if plot_flag:
        extractor_roi_annot = fast.ImagePyramidLevelExtractor.create(level=4).connect(importer_roi_annot)
        image = extractor_roi_annot.runAndGetOutputData()
        numpy_image = np.asarray(image)
        plt.imshow(numpy_image[..., 0], cmap='gray')
        plt.show()

    if plot_flag:
        extractor_dab = fast.ImagePyramidLevelExtractor.create(level=4).connect(importer_dab)
        image = extractor_dab.runAndGetOutputData()
        numpy_image = np.asarray(image)
        numpy_image = np.flip(numpy_image, axis=0)
        plt.imshow(numpy_image[..., 0], cmap='gray')
        plt.show()

    if plot_flag:
        extractor_ck = fast.ImagePyramidLevelExtractor.create(level=4).connect(importer_ck)
        image = extractor_ck.runAndGetOutputData()
        numpy_image = np.asarray(image)
        plt.imshow(numpy_image[..., 0], cmap='gray')
        plt.show()

    if plot_flag:
        extractor_annot = fast.ImagePyramidLevelExtractor.create(level=4).connect(importer_annot)
        image = extractor_annot.runAndGetOutputData()
        numpy_image = np.asarray(image)
        plt.imshow(numpy_image[..., 0], cmap='gray')
        plt.show()

    # This does not work at level 2
    # Use higher level for registration
    extractor_dab = fast.ImagePyramidLevelExtractor.create(level=4).connect(importer_dab)
    extractor_roi_annot = fast.ImagePyramidLevelExtractor.create(level=4).connect(importer_roi_annot)
    extractor_annot = fast.ImagePyramidLevelExtractor.create(level=4).connect(importer_annot)
    extractor_he = fast.ImagePyramidLevelExtractor.create(level=4).connect(importer_he)
    extractor_ck = fast.ImagePyramidLevelExtractor.create(level=4).connect(importer_ck)

    dab_image = extractor_dab.runAndGetOutputData()
    roi_annot_image = extractor_roi_annot.runAndGetOutputData()
    annot_image = extractor_annot.runAndGetOutputData()
    he_image = extractor_he.runAndGetOutputData()
    ck_image = extractor_ck.runAndGetOutputData()
    
    dab_image = np.asarray(dab_image)
    roi_annot_image = np.asarray(roi_annot_image)
    annot_image = np.asarray(annot_image)
    he_image_l4 = np.asarray(he_image)
    ck_image_l4 = np.asarray(ck_image)

    # tissue segmentation
    """
    intensity_away_from_white_thresh = 40
    ck_image_tissue = (
                np.mean(ck_image, axis=-1) < 255 - intensity_away_from_white_thresh).astype("uint8")
    he_image_tissue = (np.mean(he_image, axis=-1) < 255 - intensity_away_from_white_thresh).astype("uint8")
        # downsample before registration
    curr_shape = ck_image_tissue.shape[:2]
    # @TODO: no padding to largest image per now, since extracting areas
    ck_seg_ds = cv2.resize(ck_image_tissue, np.round(np.array(curr_shape) / ds_factor).astype("int32"),
                             interpolation=cv2.INTER_NEAREST)
    he_seg_ds = cv2.resize(he_image_tissue, np.round(np.array(curr_shape) / ds_factor).astype("int32"),
                             interpolation=cv2.INTER_NEAREST)

    # detect shift between ck and he, histogram equalization for better shift in image with few
    # distinct landmarks
    #ck_seg_ds_hist = equalize_hist(ck_seg_ds)
    # @TODO: z-shift sometimes larger than zero, why?
    shifts, reg_error, phase_diff = phase_cross_correlation(he_seg_ds, ck_seg_ds,
                                                            return_error=True)
    print(shifts)

    # scale shifts back and apply to original resolution
    shifts = (np.round(ds_factor * shifts)).astype("int32")
    shifts_ = np.zeros((3,)).astype("int32")
    shifts_[0:2] = shifts
    ck_shifted = ndi.shift(ck_image, shifts_, order=0, mode="constant", cval=255, prefilter=False)

    print(shifts)
    print(shifts_)
    print()

    dab_image = np.flip(dab_image, axis=0)
    dab_shifted = ndi.shift(dab_image, shifts_, order=0, mode="constant", cval=0, prefilter=False)
    """

    # registration with orb, done on level 4
    # im_reg_l4, h, height, width = alignImages(ck_image_l4, he_image_l4)

    dab_image = importer_dab.runAndGetOutputData()
    roi_annot_image = importer_roi_annot.runAndGetOutputData()
    annot_image = importer_annot.runAndGetOutputData()
    he_image = importer_he.runAndGetOutputData()
    ck_image = importer_ck.runAndGetOutputData()

    height_dab_image = dab_image.getLevelHeight(level)
    width_dab_image = dab_image.getLevelWidth(level)

    height_he_image = he_image.getLevelHeight(level)
    width_he_image = he_image.getLevelWidth(level)

    height_ck_image = ck_image.getLevelHeight(level)
    width_ck_image = ck_image.getLevelWidth(level)

    height_roi_annot_image = roi_annot_image.getLevelHeight(level)
    width_roi_annot_image = roi_annot_image.getLevelWidth(level)

    height_annot_image = annot_image.getLevelHeight(level)
    width_annot_image = annot_image.getLevelWidth(level)

    # @TODO: these are not correct (he and ck larger and same shape),
    # @TODO: somthing is off, and all annotations same size, should not be?
    # @TODO: will probably be a problem also when looking at coordinates from he/ck tma to get annot coordinates
    print("dab: ", height_dab_image, width_dab_image)
    print("he: ", height_he_image, width_he_image)
    print("ck: ", height_ck_image, width_ck_image)
    print("roi annot: ", height_roi_annot_image, width_roi_annot_image)
    print("annot: ", height_annot_image, width_annot_image)

    dab_access = dab_image.getAccess(fast.ACCESS_READ)
    roi_annot_access = roi_annot_image.getAccess(fast.ACCESS_READ)
    annot_access = annot_image.getAccess(fast.ACCESS_READ)
    he_access = he_image.getAccess(fast.ACCESS_READ)
    ck_access = ck_image.getAccess(fast.ACCESS_READ)

    # get shape of he and ck images
    longest_height = max([height_he_image, height_ck_image])
    longest_width = max([width_he_image, width_ck_image])
    smallest_height = min([height_he_image, height_ck_image, height_dab_image, height_annot_image])
    smallest_width = min([width_he_image, width_ck_image, width_dab_image, width_annot_image])

    for i in range(2):
        for j in range(4):
            # @TODO: dab image smaller than ck, annot smaller than he, is this an okay fix:
            large_patch_height = int(smallest_height / 4)
            large_patch_width = int(smallest_width / 2)
            w_from = i * large_patch_width
            h_from = j * large_patch_height

            if w_from + large_patch_width > smallest_width:
                large_patch_width = smallest_width - w_from
            if h_from + large_patch_height > smallest_height:
                large_patch_height = smallest_height - h_from

            he_ = he_access.getPatchAsImage(int(level), int(w_from), int(h_from), int(large_patch_width),
                                            int(large_patch_height), False)
            ck_ = ck_access.getPatchAsImage(int(level), int(w_from), int(h_from), int(large_patch_width),
                                            int(large_patch_height), False)
            dab_ = dab_access.getPatchAsImage(int(level), int(w_from),
                                              int(height_dab_image - h_from - large_patch_height),
                                              int(large_patch_width), int(large_patch_height), False)
            annot_ = annot_access.getPatchAsImage(int(level), int(w_from), int(h_from), int(large_patch_width),
                                                  int(large_patch_height), False)
            roi_annot_ = roi_annot_access.getPatchAsImage(int(level), int(w_from), int(h_from),
                                                          int(large_patch_width), int(large_patch_height), False)
            he_ = np.asarray(he_)
            ck_ = np.asarray(ck_)
            dab_ = np.asarray(dab_)
            dab_ = np.flip(dab_, axis=0)
            annot_ = np.asarray(annot_)
            roi_annot_ = np.asarray(roi_annot_)

            curr_shape = ck_.shape[:2]
            ck_seg_ds = cv2.resize(ck_, np.round(np.array(curr_shape) / ds_factor).astype("int32"),
                                   interpolation=cv2.INTER_NEAREST)
            he_seg_ds = cv2.resize(he_, np.round(np.array(curr_shape) / ds_factor).astype("int32"),
                                   interpolation=cv2.INTER_NEAREST)

            # @TODO: z-shift sometimes larger than zero, why?
            shifts, reg_error, phase_diff = phase_cross_correlation(
                he_seg_ds, ck_seg_ds, return_error=True)

            # scale shifts back and apply to original resolution
            shifts = (np.round(ds_factor * shifts)).astype("int32")
            shifts[2] = 0
            ck_large_reg = ndi.shift(ck_, shifts, order=0, mode="constant", cval=255, prefilter=False)
            dab_large_reg = ndi.shift(dab_, shifts, order=0, mode="constant", cval=0, prefilter=False)

            # differentiate between insitu, benign, invasive
            healthy_ep = ((annot_ == 1) & (dab_large_reg == 1)).astype("float32")
            in_situ_ep = ((annot_ == 2) & (dab_large_reg == 1)).astype("float32")
            invasive_ep = dab_large_reg.copy()
            invasive_ep[healthy_ep == 1] = 0
            invasive_ep[in_situ_ep == 1] = 0

            print("unique: ", np.unique(invasive_ep), np.unique(healthy_ep), np.unique(in_situ_ep))
            continue

            # create patches
            data = [he_, ck_large_reg, healthy_ep, in_situ_ep, invasive_ep, roi_annot_]
            data_fast = [fast.Image.createFromArray(curr) for curr in data]
            generators = [
                fast.PatchGenerator.create(patch_size, patch_size, overlapPercent=overlap).connect(0, curr)
                for curr in data_fast]
            streamers = [fast.DataStream(curr) for curr in generators]

            for patch_idx, (patch_he_, patch_ck_, patch_healthy, patch_in_situ, patch_invasive,
                            patch_roi_annot_) in enumerate(zip(*streamers)):  # get error here sometimes, why?
                patch_he_ = np.asarray(patch_he_)
                patch_ck_ = np.asarray(patch_ck_)
                patch_healthy = np.asarray(patch_healthy)[..., 0]
                patch_in_situ = np.asarray(patch_in_situ)[..., 0]
                patch_invasive = np.asarray(patch_invasive)[..., 0]
                patch_roi_ = np.asarray(patch_roi_annot_)[..., 0]

                # one-hot encode ground truth
                gt_one_hot = np.stack(
                    [1 - (patch_healthy.astype(bool) | patch_in_situ.astype(bool) | patch_invasive.astype(bool)),
                     patch_invasive, patch_healthy, patch_in_situ], axis=-1)

                # skip patches including areas annotated for removal or with tissue below tissue_level percent
                intensity_away_from_white_thresh = 40
                he_tissue = (
                        np.mean(patch_he_, axis=-1) < 255 - intensity_away_from_white_thresh).astype("uint8")
                he_tissue_ = np.sum(he_tissue) / (he_tissue.shape[0] * he_tissue.shape[1])
                if 1 in np.unique(patch_roi_) or he_tissue_ < tissue_level:
                    #print("skip")
                    continue

                # register on patch level
                ck_hist = equalize_hist(patch_ck_)
                shifts, reg_error, phase_diff = phase_cross_correlation(
                    patch_he_, ck_hist, return_error=True)
                shifts[2] = 0
                patch_ck_reg = ndi.shift(patch_ck_, shifts, order=0, mode="constant", cval=255, prefilter=False)
                gt_one_hot = ndi.shift(gt_one_hot, shifts, order=0, mode="constant", cval=0, prefilter=False)

                # cut he and dab image after translation, due to constant padding after shift
                start_x = shifts[0]
                start_y = shifts[1]
                stop_x = patch_size
                stop_y = patch_size
                if shifts[0] < 0:
                    start_x = 0
                    stop_x = patch_size - np.abs(shifts[0])
                if shifts[1] < 0:
                    start_y = 0
                    stop_y = patch_size - np.abs(shifts[1])
                gt_one_hot2 = gt_one_hot[int(start_x):int(stop_x), int(start_y):int(stop_y), :]
                patch_he2 = patch_he_[int(start_x):int(stop_x), int(start_y):int(stop_y), :]


                if len(np.unique(gt_one_hot2)) > 2:
                    print("making figure...")
                    f, axes = plt.subplots(2, 2, figsize=(30, 30))
                    axes[0, 0].imshow(patch_he_)
                    axes[0, 1].imshow(patch_he2)
                    axes[0, 1].imshow(gt_one_hot2[..., 1], cmap="gray", alpha=0.5)
                    axes[1, 0].imshow(patch_he2)
                    axes[1, 0].imshow(gt_one_hot2[..., 2], cmap="gray", alpha=0.5)
                    axes[1, 1].imshow(patch_he2)
                    axes[1, 1].imshow(gt_one_hot2[..., 3], cmap="gray", alpha=0.5)
                    plt.show()

                # @TODO: think about edges, how are they padded to get right shape?

    exit()

    data = [he_cut, invasive_ep, healthy_ep, in_situ_ep, roi_annot_cut]
    data_fast = [fast.Image.createFromArray(curr) for curr in data]
    generators = [fast.PatchGenerator.create(patch_size, patch_size, overlapPercent=overlap).connect(0, curr)
                  for curr in data_fast]
    streamers = [fast.DataStream(curr) for curr in generators]

    # @TODO: find out why the error below sometimes happens
    for patch_idx, (patch_he, patch_invasive, patch_healthy, patch_in_situ, patch_roi) in enumerate(zip(*streamers)):  # get error here sometimes, find out why?
        patch_he = np.asarray(patch_he)
        patch_invasive = np.asarray(patch_invasive)[..., 0]
        patch_healthy = np.asarray(patch_healthy)[..., 0]
        patch_in_situ = np.asarray(patch_in_situ)[..., 0]
        patch_roi = np.asarray(patch_roi)[..., 0]

        if 1 in np.unique(patch_roi):
            print("skip")
            continue

        # normalizing intensities for patches
        patch_invasive = minmax(patch_invasive)
        patch_healthy = minmax(patch_healthy)
        patch_in_situ = minmax(patch_in_situ)

        gt_one_hot = np.stack(
            [1 - (patch_invasive.astype(bool) | patch_healthy.astype(bool) | patch_in_situ.astype(bool)),
             patch_invasive, patch_healthy, patch_in_situ], axis=-1)

        print("gt one hot shape: ", gt_one_hot.shape)

        # for many classes
        if np.any(gt_one_hot[..., 0] < 0):
           [print(np.mean(gt_one_hot[..., iii])) for iii in range(4)]
           raise ValueError("Negative values occurred in the background class, check the segmentations...")

        if np.any(np.sum(gt_one_hot, axis=-1) > 1):
           raise ValueError("One-hot went wrong - multiple classes in the same pixel...")

        # check if either of the shapes are empty, if yes, continue
        if (len(patch_he) == 0) or (len(patch_invasive) == 0):
           continue

        # @TODO: pad patches with incorrect shape

        if plot_flag:
            fig, ax = plt.subplots(2, 2, figsize=(20, 20))
            ax[0, 0].imshow(gt_one_hot[:, :, 1])
            ax[0, 1].imshow(gt_one_hot[:, :, 2])
            ax[1, 0].imshow(gt_one_hot[:, :, 3])
            ax[1, 1].imshow(patch_he)
            plt.show()

        if plot_flag:
            fig, ax = plt.subplots(1, 2, figsize=(30, 30))
            ax[0].imshow(patch_he)
            ax[1].imshow(patch_he)
            ax[1].imshow(gt_one_hot[:, :, 2], cmap="gray", alpha=0.5)
            plt.show()

#if patch doesn't include areas in roi annotated image -> skip


if __name__ == "__main__":
    level = 2
    patch_size = 1024
    ds_factor = 4
    plot_flag = False
    overlap = 0.25
    tissue_level = 0.25

    data_split_path = ""  # split train/val/test

    curr_date = "".join(date.today().strftime("%d/%m").split("/")) + date.today().strftime("%Y")[2:]
    curr_time = "".join(str(datetime.now()).split(" ")[1].split(".")[0].split(":"))
    dataset_path = "./datasets/" + curr_date + "_" + curr_time + \
                   "_level_" + str(level) + \
                   "_psize_" + str(patch_size) + \
                   "_ds_" + str(ds_factor) + "/"

    # @TODO: cannot divide slides into train/val before? How to divide now?
    # @TODO: could I use a random change of train/val (0.6 and 0.4) for each patch? Though that would make
    # @TODO: patches from same slide in both sets. All in train instead?

    # go through files in train/val/test -> create_dataset()
    he_path_ = '/data/Maren_P1/data/WSI/'
    ck_path_ = '/data/Maren_P1/data/WSI/'
    roi_annot_path_ = '/data/Maren_P1/data/annotations_converted/patches_WSI/'
    annot_path_ = '/data/Maren_P1/data/annotations_converted/WSI/'
    dab_path_ = '/data/Maren_P1/data/annotations_converted/dab_channel_WSI_tiff/'

    for file in os.listdir(annot_path_):
        id_ = file.split("-labels.ome.tif")[0]
        he_path = he_path_ + id_ + ".vsi"
        annot_path = annot_path_ + file
        roi_annot_path = roi_annot_path_ + id_ + ".vsi - 40x-labels.ome.tif"

        for file in os.listdir(dab_path_):
            id_2 = file.split(".tiff")[0]
            if id_2 in id_:
                dab_path = dab_path_ + id_2 + ".tiff"
                ck_path = ck_path_ + id_2 + " CK.vsi"

        print(he_path)
        print(ck_path)
        print(roi_annot_path)
        print(annot_path)
        print(dab_path)

        create_dataset(he_path, ck_path, roi_annot_path, annot_path, dab_path, dataset_path, level, patch_size,
                       ds_factor, overlap, tissue_level)