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
                   overlap):
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

    """ this does not work at level 2
    # @TODO: is it best to do this or use importer and .getAccess()
    extractor_dab = fast.ImagePyramidLevelExtractor.create(level=level).connect(importer_dab)
    extractor_roi_annot = fast.ImagePyramidLevelExtractor.create(level=level).connect(importer_roi_annot)
    extractor_annot = fast.ImagePyramidLevelExtractor.create(level=level).connect(importer_annot)
    extractor_he = fast.ImagePyramidLevelExtractor.create(level=level).connect(importer_he)
    extractor_ck = fast.ImagePyramidLevelExtractor.create(level=level).connect(importer_ck)

    dab_image = extractor_dab.runAndGetOutputData()
    roi_annot_image = extractor_roi_annot.runAndGetOutputData()
    annot_image = extractor_annot.runAndGetOutputData()
    he_image = extractor_he.runAndGetOutputData()
    ck_image = extractor_ck.runAndGetOutputData()
    
    dab_image = np.asarray(dab_image)
    roi_annot_image = np.asarray(roi_annot_image)
    annot_image = np.asarray(annot_image)
    he_image = np.asarray(he_image)
    ck_image = np.asarray(ck_image)
    """

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

    print(longest_height, longest_width)

    nbr_h = int(longest_height / 10000)
    nbr_w = int(longest_width / 10000)

    print(nbr_h, nbr_w)

    # @TODO: fix edge cases, will stops now
    for i in range(nbr_w + 1):
        for j in range(nbr_h + 1):
            w_from = 10000 * i
            h_from = 10000 * j

            dab_ = dab_access.getPatchAsImage(int(level), int(w_from), int(height_dab_image - h_from - 10000), int(10000),
                                              int(10000), False)

            he_ = he_access.getPatchAsImage(int(level), int(w_from), int(h_from), int(10000),
                                            int(10000), False)

            ck_ = ck_access.getPatchAsImage(int(level), int(w_from), int(h_from), int(10000),
                                            int(10000), False)

            annot_ = annot_access.getPatchAsImage(int(level), int(w_from), int(h_from), int(10000),
                                                  int(10000), False)

            rot_annot_ = roi_annot_access.getPatchAsImage(int(level), int(w_from), int(h_from), int(10000),
                                                          int(10000), False)

            dab_ = np.asarray(dab_)
            dab_ = np.flip(dab_, axis=0)
            he_ = np.asarray(he_)
            ck_ = np.asarray(ck_)
            annot_ = np.asarray(annot_)
            roi_annot_ = np.asarray(rot_annot_)

            if plot_flag:
                f, axes = plt.subplots(2, 3, figsize=(30, 30))
                axes[0, 0].imshow(dab_, cmap="gray")
                axes[0, 1].imshow(he_)
                axes[0, 2].imshow(ck_)
                axes[1, 0].imshow(annot_)
                axes[1, 1].imshow(roi_annot_, cmap="gray")
                axes[1, 2].imshow(roi_annot_, cmap="gray")
                plt.show()

            # downsample before registration
            curr_shape = ck_.shape[:2]
            # @TODO: no padding to largest image per now, since extracting areas
            ck_image_ds = cv2.resize(ck_, np.round(np.array(curr_shape) / ds_factor).astype("int32"),
                                     interpolation=cv2.INTER_NEAREST)
            he_image_ds = cv2.resize(he_, np.round(np.array(curr_shape) / ds_factor).astype("int32"),
                                     interpolation=cv2.INTER_NEAREST)

            # detect shift between ck and he, histogram equalization for better shift in image with few
            # distinct landmarks
            ck_image_ds_hist = equalize_hist(ck_image_ds)
            # @TODO: z-shift sometimes larger than zero, why?
            shifts, reg_error, phase_diff = phase_cross_correlation(he_image_ds, ck_image_ds_hist,
                                                                    return_error=True)
            shifts[2] = 0  # set z-axis to zero (should be from beginning, but is not always)

            # scale shifts back and apply to original resolution
            shifts = (np.round(ds_factor * shifts)).astype("int32")
            ck_shifted = ndi.shift(ck_, shifts, order=0, mode="constant", cval=255, prefilter=False)
            dab_shifted = ndi.shift(dab_, shifts, order=0, mode="constant", cval=0, prefilter=False)

            # @TODO: should not need to cut all sides
            ck_shifted_cut = ck_shifted[np.abs(shifts[0]):10000 - np.abs(shifts[0]),
                              np.abs(shifts[1]):10000 - np.abs(shifts[1]), :]
            dab_shifted_cut = dab_shifted[np.abs(shifts[0]):10000 - np.abs(shifts[0]),
                              np.abs(shifts[1]):10000 - np.abs(shifts[1]), :]
            he_cut = he_[np.abs(shifts[0]):10000 - np.abs(shifts[0]), np.abs(shifts[1]):10000 - np.abs(shifts[1]), :]
            annot_cut = annot_[np.abs(shifts[0]):10000 - np.abs(shifts[0]), np.abs(shifts[1]):10000 - np.abs(shifts[1]), :]
            roi_annot_cut = roi_annot_[np.abs(shifts[0]):10000 - np.abs(shifts[0]), np.abs(shifts[1]):10000 - np.abs(shifts[1]), :]

            if plot_flag:
                f, axes = plt.subplots(1, 2, figsize=(30, 30))
                axes[0].imshow(he_)
                axes[0].imshow(ck_, cmap="gray", alpha=0.5)
                axes[1].imshow(he_cut)
                axes[1].imshow(ck_shifted_cut, cmap="gray", alpha=0.5)
                plt.show()

            if plot_flag:
                f, axes = plt.subplots(1, 2, figsize=(30, 30))
                axes[0].imshow(he_)
                axes[0].imshow(annot_, alpha=0.5)
                axes[1].imshow(he_)
                axes[1].imshow(roi_annot_, cmap="gray", alpha=0.5)
                plt.show()

            if plot_flag:
                f, axes = plt.subplots(1, 2, figsize=(30, 30))
                axes[0].imshow(he_)
                axes[0].imshow(dab_, cmap="gray", alpha=0.5)
                axes[1].imshow(he_cut)
                axes[1].imshow(dab_shifted_cut, cmap="gray", alpha=0.5)
                plt.show()

            # differentiate between insitu, benign, invasive
            healthy_ep = ((annot_cut == 1) & (dab_shifted_cut == 1)).astype("float32")
            in_situ_ep = ((annot_cut == 2) & (dab_shifted_cut == 1)).astype("float32")
            invasive_ep = dab_shifted_cut.copy()
            invasive_ep[healthy_ep == 1] = 0
            invasive_ep[in_situ_ep == 1] = 0

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

                if True:
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

        create_dataset(he_path, ck_path, roi_annot_path, annot_path, dab_path, dataset_path, level, patch_size, ds_factor, overlap)