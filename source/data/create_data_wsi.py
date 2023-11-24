"""
Script to create patches from whole slide images
Annotations from QuPath of which areas to exclude (from he image)
Put patches in train/val set
"""
from datetime import datetime, date
import h5py
import numpy as np
#import fast
import cv2
from skimage.registration import phase_cross_correlation
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import os
from skimage.exposure import equalize_hist
import multiprocessing as mp


def create_datasets_wrapper(some_inputs_):
    # give all arguements from wrapper to create_datasets to begin processing WSI
    create_dataset(*some_inputs_)


def minmax(x):
    """
    normalizes intensities to range float [0, 1]
    :param x: intensity image
    :return: normalized x
    """
    # @TODO: Sometimes get error: invalid value encountered in divide (x/=np.argmax(x)).
    # @TODO: is this really a good idea, will increase contrast?
    #  Is it possible I get one for both argmax and argmin
    x = x.astype("float32")
    if np.amax(x) > 0:
        if np.amax(x) == 1. and np.amin(x) == 1.:  # @TODO: it this okay then? what it both ex 255 or other nbr?
            return x
        else:
            x -= np.amin(x)
            x /= np.amax(x)
    return x


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


def create_dataset(he_path, ck_path, roi_annot_path, annot_path, dab_path, dataset_path, level, patch_size, ds_factor,
                   overlap, tissue_level, wsi_idx, i, j, square_idx, set_name):

    import fast

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

    height_annot_image = annot_image.getLevelHeight(level)
    width_annot_image = annot_image.getLevelWidth(level)

    dab_access = dab_image.getAccess(fast.ACCESS_READ)
    roi_annot_access = roi_annot_image.getAccess(fast.ACCESS_READ)
    annot_access = annot_image.getAccess(fast.ACCESS_READ)
    he_access = he_image.getAccess(fast.ACCESS_READ)
    ck_access = ck_image.getAccess(fast.ACCESS_READ)

    # get shape of he and ck images
    smallest_height = min([height_he_image, height_ck_image, height_dab_image, height_annot_image])
    smallest_width = min([width_he_image, width_ck_image, width_dab_image, width_annot_image])

    # @TODO: dab image smaller than ck, annot smaller than he, is this an okay fix:


    large_patch_height = int(smallest_height / 4)
    large_patch_width = int(smallest_width / 2)
    w_from = i * large_patch_width
    h_from = j * large_patch_height

    if w_from + large_patch_width > smallest_width:
        large_patch_width = smallest_width - w_from
    if h_from + large_patch_height > smallest_height:
        large_patch_height = smallest_height - h_from
    if height_dab_image - h_from - large_patch_height + large_patch_height > smallest_height:
        large_patch_height = smallest_height - (height_dab_image - h_from - large_patch_height)

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
    # axis order for shift vector "is consistent with the axis order of the input array"
    # https://scikit-image.org/docs/stable/api/skimage.registration.html
    shifts, reg_error, phase_diff = phase_cross_correlation(
        he_seg_ds, ck_seg_ds, return_error=True)

    # scale shifts back and apply to original resolution
    shifts = (np.round(ds_factor * shifts)).astype("int32")
    shifts[2] = 0
    ck_large_reg = ndi.shift(ck_, shifts, order=0, mode="constant", cval=255, prefilter=False)
    dab_large_reg = ndi.shift(dab_, shifts, order=0, mode="constant", cval=0, prefilter=False)

    # cut area that has been padded:
    # cut he and dab image after translation, due to constant padding after shift
    start_h, start_w, stop_h, stop_w = cut_image(shifts[0], shifts[1], large_patch_height, large_patch_width)

    ck_large_reg = ck_large_reg[int(start_h):int(stop_h), int(start_w):int(stop_w), :]
    dab_large_reg = dab_large_reg[int(start_h):int(stop_h), int(start_w):int(stop_w), :]
    he_ = he_[int(start_h):int(stop_h), int(start_w):int(stop_w), :]
    annot_ = annot_[int(start_h):int(stop_h), int(start_w):int(stop_w), :]
    roi_annot_ = roi_annot_[int(start_h):int(stop_h), int(start_w):int(stop_w), :]

    # differentiate between insitu, benign, invasive
    healthy_ep = ((annot_ == 1) & (dab_large_reg == 1)).astype("uint8")
    in_situ_ep = ((annot_ == 2) & (dab_large_reg == 1)).astype("uint8")
    invasive_ep = dab_large_reg.copy().astype("uint8")
    invasive_ep[healthy_ep == 1] = 0
    invasive_ep[in_situ_ep == 1] = 0

    print("unique: ", np.unique(invasive_ep), np.unique(healthy_ep), np.unique(in_situ_ep))
    # create patches
    data = [he_, ck_large_reg, healthy_ep, in_situ_ep, invasive_ep, roi_annot_]
    data_fast = [fast.Image.createFromArray(curr) for curr in data]
    generators = [
        fast.PatchGenerator.create(patch_size, patch_size, overlapPercent=overlap).connect(0, curr)
        for curr in data_fast]
    streamers = [fast.DataStream(curr) for curr in generators]

    for patch_idx, (patch_he_, patch_ck_, patch_healthy, patch_in_situ, patch_invasive,
                    patch_roi_annot_) in enumerate(zip(*streamers)):  # get error here sometimes, why?
        try:
            # convert from FAST image to numpy array
            # @TODO: why is this necessary? otherwise get error sometimes
            patch_he_ = np.asarray(patch_he_)
            patch_ck_ = np.asarray(patch_ck_)
            patch_healthy = np.asarray(patch_healthy)[..., 0]
            patch_in_situ = np.asarray(patch_in_situ)[..., 0]
            patch_invasive = np.asarray(patch_invasive)[..., 0]
            patch_roi_ = np.asarray(patch_roi_annot_)[..., 0]
        except RuntimeError as e:
            print(e)
            continue

        # one-hot encode ground truth
        gt_one_hot = np.stack(
            [1 - (patch_healthy.astype(bool) | patch_in_situ.astype(bool) | patch_invasive.astype(bool)),
             patch_invasive, patch_healthy, patch_in_situ], axis=-1)
        if np.any(gt_one_hot[..., 0] < 0):
            [print(np.mean(gt_one_hot[..., iii])) for iii in range(4)]
            raise ValueError("Negative values occurred in the background class, check the segmentations...")

        if np.any(np.sum(gt_one_hot, axis=-1) > 1):
            raise ValueError("One-hot went wrong - multiple classes in the same pixel...")

        # check if either of the shapes are empty, if yes, continue
        if (len(patch_he_) == 0) or (len(patch_invasive) == 0):
            continue

        # pad patches that are not shape patch_size
        if np.array(patch_he_).shape[0] < patch_size or np.array(patch_he_).shape[1] < patch_size:
            patch_he_padded = np.ones((patch_size, patch_size, 3), dtype="uint8") * 255
            patch_ck_padded = np.ones((patch_size, patch_size, 3), dtype="uint8") * 255
            patch_gt_padded = np.zeros((patch_size, patch_size, 4), dtype="float32")

            patch_he_padded[:patch_he_.shape[0], :patch_he_.shape[1]] = patch_he_
            patch_ck_padded[:patch_ck_.shape[0], :patch_ck_.shape[1]] = patch_ck_
            patch_gt_padded[:gt_one_hot.shape[0], :gt_one_hot.shape[1]] = gt_one_hot

            patch_he_ = patch_he_padded
            patch_ck_ = patch_ck_padded
            gt_one_hot = patch_gt_padded

        # skip patches including areas annotated for removal or with tissue below tissue_level percent
        if 1. in np.unique(patch_roi_):
            continue

        # register on patch level
        ck_hist = equalize_hist(patch_ck_)
        shifts, reg_error, phase_diff = phase_cross_correlation(
            patch_he_, ck_hist, return_error=True)
        shifts[2] = 0
        gt_one_hot = ndi.shift(gt_one_hot, shifts, order=0, mode="constant", cval=0, prefilter=False)

        # cut he and dab image after translation, due to constant padding after shift
        start_h, start_w, stop_h, stop_w = cut_image(shifts[0], shifts[1], patch_size, patch_size)
        gt_one_hot = gt_one_hot[int(start_h):int(stop_h), int(start_w):int(stop_w), :]
        patch_he_ = patch_he_[int(start_h):int(stop_h), int(start_w):int(stop_w), :]

        # pad patches that are not shape patch_size
        if np.array(patch_he_).shape[0] < patch_size or np.array(patch_he_).shape[1] < patch_size:
            patch_he_padded = np.ones((patch_size, patch_size, 3), dtype="uint8") * 255
            patch_gt_padded = np.zeros((patch_size, patch_size, 4), dtype="float32")

            patch_he_padded[:patch_he_.shape[0], :patch_he_.shape[1]] = patch_he_
            patch_gt_padded[:gt_one_hot.shape[0], :gt_one_hot.shape[1]] = gt_one_hot

            patch_he_ = patch_he_padded
            gt_one_hot = patch_gt_padded

        if plot_flag:
            print("making figure...")
            f, axes = plt.subplots(2, 2, figsize=(30, 30))
            axes[0, 0].imshow(patch_he_)
            axes[0, 1].imshow(patch_he_)
            axes[0, 1].imshow(gt_one_hot[..., 0], cmap="gray", alpha=0.5)
            axes[1, 0].imshow(patch_he_)
            axes[1, 0].imshow(gt_one_hot[..., 1], cmap="gray", alpha=0.5)
            axes[1, 1].imshow(patch_he_)
            axes[1, 1].imshow(gt_one_hot[..., 2], cmap="gray", alpha=0.5)
            plt.show()


        # save patches as hdf5
        if np.count_nonzero(patch_in_situ) > 0:
            add_to_path = 'inSitu/'
        elif np.count_nonzero(patch_healthy) > 0:
            add_to_path = 'benign/'
        else:
            add_to_path = 'invasive/'

        # create folder if not exists
        os.makedirs(dataset_path + set_name + add_to_path, exist_ok=True)
        with h5py.File(dataset_path + set_name + add_to_path + "wsi_" + str(wsi_idx) + "_" + str(square_idx) + "_" +
                       str(patch_idx) + ".h5", "w") as f:
            f.create_dataset(name="input", data=patch_he_.astype("uint8"))
            f.create_dataset(name="output", data=gt_one_hot.astype("float32"))

    # delete streamers and stuff to potentially avoid threading issues in FAST
    del data_fast, generators, streamers, data
    del he_, ck_, dab_, annot_, roi_annot_, ck_large_reg, dab_large_reg
    del healthy_ep, in_situ_ep, invasive_ep
    del patch_he_, patch_ck_, patch_healthy, patch_in_situ, patch_invasive, patch_roi_annot_, patch_roi_
    del patch_he_padded, patch_gt_padded, gt_one_hot
    del ck_seg_ds, he_seg_ds,
    del ck_hist
    import gc
    gc.collect()
    print("end of square")

if __name__ == "__main__":
    level = 2  # image level
    patch_size = 1024
    ds_factor = 4  # downsample before finding shift in large patches
    plot_flag = False
    overlap = 0.25  # overlap when creating patches
    tissue_level = 0.25  # patches with less tissue will be skipped
    wsi_idx = 0

    curr_date = "".join(date.today().strftime("%d/%m").split("/")) + date.today().strftime("%Y")[2:]
    curr_time = "".join(str(datetime.now()).split(" ")[1].split(".")[0].split(":"))
    dataset_path = "/path/to/dataset/" + curr_date + "_" + curr_time + \
                   "_wsi" + \
                   "_level_" + str(level) + \
                   "_psize_" + str(patch_size) + \
                   "_ds_" + str(ds_factor) + "/"

    # go through files in train/val/test -> create_dataset()
    he_path_ = '/path/to/he/'
    ck_path_ = '/path/to/ck/'
    roi_annot_path_ = '/path/to/patches/wsi/annot/'
    annot_path_ = '/path/to/annots/'
    dab_path_ = '/path/to/dab/'
    wsi1_split_path = './path/to/datasplit/h5'
    wsi2_split_path = './path/to/datasplit/h5'

    # define datasets (train/val/test) - always uses predefined dataset
    with h5py.File(wsi1_split_path, "r") as f:
        train_set1 = np.array(f['train']).astype(int)
        val_set1 = np.array(f['val']).astype(int)

    with h5py.File(wsi2_split_path, "r") as f:
        train_set2 = np.array(f['train']).astype(int)
        val_set2 = np.array(f['val']).astype(int)

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

        square_idx = 0

        for i in range(2):
            for j in range(4):

                if wsi_idx == 0:
                    if square_idx in train_set1:
                        set_name = "ds_train/"
                    elif square_idx in val_set1:
                        set_name = "ds_val/"
                    else:
                        raise ValueError("Error in wsi division for train and val set")
                elif wsi_idx == 1:
                    if square_idx in train_set2:
                        set_name = "ds_train/"
                    elif square_idx in val_set2:
                        set_name = "ds_val/"
                    else:
                        raise ValueError("Error in wsi division for train and val set")
                else:
                    raise ValueError("More wsis than accounted for")

                inputs_ = [[he_path, ck_path, roi_annot_path, annot_path, dab_path, dataset_path, level, patch_size,
                           ds_factor, overlap, tissue_level, wsi_idx, i, j, square_idx, set_name]]
                p = mp.Pool(1)
                output = p.map(create_datasets_wrapper, inputs_)
                p.terminate()
                p.join()
                del p, inputs_
                square_idx += 1

        wsi_idx += 1
