"""
Script for creating patches from histopathological images and saving them in hdf5 format.
The patches are either placed in an invasive, benign or inSitu folder depending on their content.
"""

#import fast
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from skimage.registration import phase_cross_correlation
from scipy import ndimage as ndi
import h5py
from datetime import datetime, date
import multiprocessing as mp
from skimage.exposure import equalize_hist
from skimage.morphology import remove_small_holes, binary_dilation, disk


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
            x /= np.amax(x)  #@TODO: got error here still once
    return x


def create_datasets_wrapper(some_inputs_):
    # give all arguements from wrapper to create_datasets to begin processing WSI
    create_datasets(*some_inputs_)


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


def dsc(pred, target):
    intersection = np.sum(pred * target)
    union = np.sum(pred * pred) + np.sum(target * target)
    return np.clip((2 * intersection + 1) / (union + 1), 0, 1)


def create_datasets(he_path, ck_path, mask_path, annot_path, remove_path, triplet_path, dataset_path, set_name,
                    plot_flag, level, nb_iters, patch_size, downsample_factor, wsi_idx, dist_limit, overlap,
                    file_front, id_):

    #fast.Reporter.setGlobalReportMethod(fast.Reporter.COUT)  # verbose

    # import fast here to free memory when this is done (if running in a separate process)
    import fast
    test = False

    # import images:
    importer_he = fast.WholeSlideImageImporter.create(
        he_path)  # path to CK image
    importer_ck = fast.WholeSlideImageImporter.create(
        ck_path)  # path to CK image
    importer_mask = fast.TIFFImagePyramidImporter.create(
        mask_path)  # path to dab image
    importer_annot = fast.TIFFImagePyramidImporter.create(
        annot_path)  # path to annotated image
    importer_remove = fast.TIFFImagePyramidImporter.create(
        remove_path)  # path to annotated remove cores image
    importer_triplet = fast.TIFFImagePyramidImporter.create(
        triplet_path)  # path to annotated triplet image

    # access annotated mask (generated from QuPath)
    mask = importer_mask.runAndGetOutputData()
    annot = importer_annot.runAndGetOutputData()
    annot_remove = importer_remove.runAndGetOutputData()
    annot_for_width = importer_annot.runAndGetOutputData()
    annot_triplet = importer_triplet.runAndGetOutputData()

    height_mask = mask.getLevelHeight(level)
    width_mask = mask.getLevelWidth(level)

    height_annot = annot_for_width.getLevelHeight(level)
    width_annot = annot_for_width.getLevelWidth(level)

    access = mask.getAccess(fast.ACCESS_READ)
    access_annot = annot.getAccess(fast.ACCESS_READ)
    access_remove = annot_remove.getAccess(fast.ACCESS_READ)
    access_triplet = annot_triplet.getAccess(fast.ACCESS_READ)

    # plot whole TMA image (does not work on level 0-3, image level too large to convert to FAST image)
    if plot_flag:
        extractor = fast.ImagePyramidLevelExtractor.create(level=4).connect(importer_mask)
        image = extractor.runAndGetOutputData()
        numpy_image = np.asarray(image)
        plt.imshow(numpy_image[..., 0], cmap='gray')
        plt.show()

    if plot_flag:
        extractor = fast.ImagePyramidLevelExtractor.create(level=4).connect(importer_ck)
        image = extractor.runAndGetOutputData()
        numpy_image = np.asarray(image)
        plt.imshow(numpy_image[..., 0], cmap='gray')
        plt.show()

    if plot_flag:
        extractor = fast.ImagePyramidLevelExtractor.create(level=4).connect(importer_annot)
        image = extractor.runAndGetOutputData()
        numpy_image = np.asarray(image)
        plt.imshow(numpy_image[..., 0], cmap='jet', interpolation="none")
        plt.show()

    if plot_flag:
        extractor = fast.ImagePyramidLevelExtractor.create(level=4).connect(importer_remove)
        image = extractor.runAndGetOutputData()
        numpy_image = np.asarray(image)
        plt.imshow(numpy_image[..., 0], cmap='jet', interpolation="none")
        plt.show()

    # get CK TMA cores
    extractor = fast.TissueMicroArrayExtractor.create(level=level).connect(importer_ck)
    ck_tmas = []
    ck_stream = fast.DataStream(extractor)
    for j, tma in enumerate(ck_stream):
        ck_tmas.append(tma)
        if j == nb_iters:
            break
    del extractor, ck_stream

    # get HE TMA cores
    extractor = fast.TissueMicroArrayExtractor.create(level=level).connect(importer_he)
    he_tmas = []
    he_stream = fast.DataStream(extractor)
    for j, tma in enumerate(he_stream):
        he_tmas.append(tma)
        if j == nb_iters:
            break
    del extractor, he_stream

    print("length HE TMAs:", len(he_tmas))
    print("length CK TMAs:", len(ck_tmas))
    tma_idx = 0

    count_invasive = 0
    count_benign = 0
    count_inSitu = 0
    count = 0

    for he_counter in range(len(he_tmas)):
        for ck_counter in range(len(ck_tmas)):

            he_tma = he_tmas[he_counter]
            ck_tma = ck_tmas[ck_counter]

            # positions will change slightly depending on level of TMA extractor even though the positions
            # are at level 0. Due to upscaling to level 0?
            position_he = he_tma.getTransform().getTranslation()  # position of he_tma at level 0
            position_ck = ck_tma.getTransform().getTranslation()  # position of ck_tma at level 0

            position_he_x = position_he[0][0]
            position_he_y = position_he[1][0]

            position_ck_x = position_ck[0][0]
            position_ck_y = position_ck[1][0]

            dist_x = position_he_x - position_ck_x
            dist_y = position_he_y - position_ck_y

            if np.abs(dist_x) < dist_limit and np.abs(dist_y) < dist_limit:  # if positions are close we have a pair
                count += 1
                try:
                    ck_tma = np.asarray(ck_tma)
                    he_tma = np.asarray(he_tma)
                except RuntimeError as e:
                    print(e)
                    continue

                shapes_ck_tma = ck_tma.shape
                shapes_he_tma = he_tma.shape

                height_ck, width_ck, _ = ck_tma.shape  # need when finding TMA in mask slide
                height_he, width_he, _ = he_tma.shape

                longest_height = max([shapes_ck_tma[0], shapes_he_tma[0]])
                longest_width = max([shapes_ck_tma[1], shapes_he_tma[1]])

                ck_tma_padded = np.ones((longest_height, longest_width, 3), dtype="uint8") * 255
                he_tma_padded = np.ones((longest_height, longest_width, 3), dtype="uint8") * 255

                ck_tma_padded[:ck_tma.shape[0], :ck_tma.shape[1]] = ck_tma
                he_tma_padded[:he_tma.shape[0], :he_tma.shape[1]] = he_tma

                # skip cores that should be removed
                position_ck_x /= (2 ** level)
                position_ck_y /= (2 ** level)
                # @TODO: get warning from TIFFReadTile for edge cases, ok? They are padded when too small
                try:
                    remove_annot = access_remove.getPatchAsImage(int(level), int(position_ck_x), int(position_ck_y),
                                                                 int(width_ck), int(height_ck), False)
                except RuntimeError as e:
                    print(e)
                    continue
                tma_remove = np.asarray(remove_annot)
                if np.count_nonzero(tma_remove) > 0:
                    continue

                # get id number for triplet
                try:
                    triplet_annot = access_triplet.getPatchAsImage(int(level), int(position_ck_x), int(position_ck_y),
                                                                 int(width_ck), int(height_ck), False)
                except RuntimeError as e:
                    print(e)
                    continue
                triplet = np.asarray(triplet_annot)
                triplet_nbr = np.amax(triplet)

                # downsample image before registration
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

                # remove cylinders with dice score below threshold, remove cores with different tissue
                # ex due to only parts of one stain extracted with TMA extractor or very broken TMAs in one stain
                he_tma_padded_ = minmax(he_tma_padded)  # to account for intensity differences in staining
                ck_tma_padded_ = minmax(ck_tma_padded_shifted)
                intensity_away_from_white_thresh_he = 0.20
                intensity_away_from_white_thresh_ck = 0.20
                he_tissue = (
                        np.mean(he_tma_padded_, axis=-1) < 1. - intensity_away_from_white_thresh_he).astype("uint8")
                ck_tissue = (
                        np.mean(ck_tma_padded_, axis=-1) < 1. - intensity_away_from_white_thresh_ck).astype("uint8")

                he_tissue = binary_dilation(he_tissue, disk(radius=5))
                ck_tissue = binary_dilation(ck_tissue, disk(radius=5))
                he_tissue = remove_small_holes(he_tissue, 8000)
                ck_tissue = remove_small_holes(ck_tissue, 8000)

                # skip matches with very different dice, destroyed tissue
                if dsc(he_tissue, ck_tissue) < 0.80:
                    continue

                # Get TMA from annot slide
                position_he_x /= (2 ** level)
                position_he_y /= (2 ** level)
                position_ck_y = height_mask - position_ck_y - height_ck

                # get corresponding TMA core in the dab image as in the CK:
                # get corresponding TMA core in manual annotated image as in the HE:
                # skip TMA cores when area is outside mask area
                if position_ck_x + width_ck > width_mask or position_ck_y + height_ck > height_mask:
                    continue
                if position_he_x + width_he > width_annot or position_he_y + height_he > height_annot:
                    continue

                # get dab and manual annotation tma cores as images
                dab_core = access.getPatchAsImage(int(level), int(position_ck_x), int(position_ck_y), int(width_ck),
                                                  int(height_ck), False)
                annot_core = access_annot.getPatchAsImage(int(level), int(position_he_x), int(position_he_y),
                                                          int(width_he), int(height_he), False)
                dab_core = np.asarray(dab_core)
                annot_core = np.asarray(annot_core)

                dab_core = dab_core[..., 0]
                annot_core = annot_core[..., 0]
                dab_core = np.flip(dab_core, axis=0)  # since dab annotation is flipped

                annot_core_padded = np.zeros((longest_height, longest_width), dtype="uint8")
                dab_core_padded = np.zeros((longest_height, longest_width), dtype="uint8")

                # the correctly placed dab and manual annotation:
                dab_core_padded[:dab_core.shape[0], :dab_core.shape[1]] = dab_core
                annot_core_padded[:annot_core.shape[0], :annot_core.shape[1]] = annot_core

                dab_core_padded_shifted = ndi.shift(dab_core_padded, shifts[:2], order=0, mode="constant", cval=0, prefilter=False)

                # cut to match ck and he
                annot_core_padded = annot_core_padded[int(start_h):int(stop_h), int(start_w):int(stop_w)]
                dab_core_padded_shifted = dab_core_padded_shifted[int(start_h):int(stop_h), int(start_w):int(stop_w)]

                # get each GT annotation as its own binary image + fix manual annotations
                manual_annot = np.asarray(annot_core_padded)
                healthy_ep = ((manual_annot == 1) & (dab_core_padded_shifted == 1)).astype("uint8")
                in_situ_ep = ((manual_annot == 2) & (dab_core_padded_shifted == 1)).astype("uint8")

                if class_ == "multiclass":
                    dab_core_padded_shifted[healthy_ep == 1] = 0
                    dab_core_padded_shifted[in_situ_ep == 1] = 0

                data = [he_tma_padded, ck_tma_padded_shifted, dab_core_padded_shifted, healthy_ep, in_situ_ep]
                data_fast = [fast.Image.createFromArray(curr) for curr in data]
                generators = [fast.PatchGenerator.create(patch_size, patch_size, overlapPercent=overlap).connect(0, curr) for curr in data_fast]
                streamers = [fast.DataStream(curr) for curr in generators]

                for patch_idx, (patch_he, patch_ck, patch_mask, patch_healthy, patch_in_situ) in enumerate(zip(*streamers)):  # get error here sometimes, find out why?
                    try:
                        # convert from FAST image to numpy array
                        patch_he = np.asarray(patch_he)
                        patch_ck = np.asarray(patch_ck)
                        patch_mask = np.asarray(patch_mask)[..., 0]
                        patch_healthy = np.asarray(patch_healthy)[..., 0]
                        patch_in_situ = np.asarray(patch_in_situ)[..., 0]
                    except RuntimeError as e:
                        print(e)
                        continue


                    # create one-hot, one channel for each class
                    if class_ == "multiclass":
                        gt_one_hot = np.stack([1 - (patch_mask.astype(bool) | patch_healthy.astype(bool) | patch_in_situ.astype(bool)), patch_mask, patch_healthy, patch_in_situ], axis=-1)
                        if np.any(gt_one_hot[..., 0] < 0):
                            [print(np.mean(gt_one_hot[..., iii])) for iii in range(4)]
                            raise ValueError("Negative values occurred in the background class, check the segmentations...")

                        if np.any(np.sum(gt_one_hot, axis=-1) > 1):
                            raise ValueError("One-hot went wrong - multiple classes in the same pixel...")

                        # check if either of the shapes are empty, if yes, continue
                        if (len(patch_he) == 0) or (len(patch_mask) == 0):
                            continue

                    # for all epithelium as one class:
                    if class_ == "singleclass":
                        gt_one_hot = np.stack([1 - (patch_mask.astype(bool)), patch_mask], axis=-1)  # for dataset with all epithelium as one class
                        if np.any(gt_one_hot[..., 0] < 0):
                            [print(np.mean(gt_one_hot[..., iii])) for iii in range(2)]
                            raise ValueError("Negative values occurred in the background class, check the segmentations...")

                        if np.any(np.sum(gt_one_hot, axis=-1) > 1):
                            raise ValueError("One-hot went wrong - multiple classes in the same pixel...")

                        # check if either of the shapes are empty, if yes, continue
                        if (len(patch_he) == 0) or (len(patch_mask) == 0):
                            continue

                    if np.array(patch_he).shape[0] < patch_size or np.array(patch_he).shape[1] < patch_size:
                        patch_he_padded = np.ones((patch_size, patch_size, 3), dtype="uint8") * 255
                        patch_ck_padded = np.ones((patch_size, patch_size, 3), dtype="uint8") * 255
                        if class_ == "multiclass":
                            patch_gt_padded = np.zeros((patch_size, patch_size, 4), dtype="float32")
                        if class_ == "singleclass":
                            patch_gt_padded = np.zeros((patch_size, patch_size, 2), dtype="float32")
                        patch_he_padded[:patch_he.shape[0], :patch_he.shape[1]] = patch_he
                        patch_ck_padded[:patch_ck.shape[0], :patch_ck.shape[1]] = patch_ck
                        patch_gt_padded[:gt_one_hot.shape[0], :gt_one_hot.shape[1]] = gt_one_hot

                        patch_he = patch_he_padded
                        patch_ck = patch_ck_padded
                        gt_one_hot = patch_gt_padded

                    # skip patches with less than 0.25 percent tissue
                    intensity_away_from_white_thresh = 40
                    he_tissue = (
                            np.mean(patch_he, axis=-1) < 255 - intensity_away_from_white_thresh).astype("uint8")
                    he_tissue_ = np.sum(he_tissue) / (he_tissue.shape[0] * he_tissue.shape[1])

                    if he_tissue_ < skip_percentage:
                        continue

                    # register on patch level
                    ck_hist = equalize_hist(patch_ck)
                    shifts, reg_error, phase_diff = phase_cross_correlation(
                        patch_he, ck_hist, return_error=True)
                    shifts[2] = 0
                    # @TODO: fix for single class too
                    patch_ck_ = ndi.shift(patch_ck, shifts, order=0, mode="constant", cval=255, prefilter=False)
                    gt_one_hot_ = ndi.shift(gt_one_hot, shifts, order=0, mode="constant", cval=0., prefilter=False)

                    start_h, start_w, stop_h, stop_w = cut_image(shifts[0], shifts[1], patch_size, patch_size)
                    gt_one_hot_ = gt_one_hot_[int(start_h):int(stop_h), int(start_w):int(stop_w), :]
                    patch_he = patch_he[int(start_h):int(stop_h), int(start_w):int(stop_w), :]

                    if np.array(patch_he).shape[0] < patch_size or np.array(patch_he).shape[1] < patch_size:
                        patch_he_padded = np.ones((patch_size, patch_size, 3), dtype="uint8") * 255
                        if class_ == "multiclass":
                            patch_gt_padded = np.zeros((patch_size, patch_size, 4), dtype="float32")
                        patch_he_padded[:patch_he.shape[0], :patch_he.shape[1]] = patch_he
                        patch_gt_padded[:gt_one_hot_.shape[0], :gt_one_hot_.shape[1]] = gt_one_hot_

                        patch_he = patch_he_padded
                        gt_one_hot = patch_gt_padded

                    # check if patch includes benign or in situ
                    if np.count_nonzero(patch_in_situ) > 0:
                        add_to_path = 'inSitu/'
                        count_inSitu += 1
                    elif np.count_nonzero(patch_healthy) > 0:
                        add_to_path = 'benign/'
                        count_benign += 1
                    else:
                        add_to_path = 'invasive/'
                        count_invasive += 1

                    triplet_nbr = str(triplet_nbr)

                    # create folder if not exists
                    if class_ == "multiclass":
                        os.makedirs(dataset_path + set_name + "/" + add_to_path, exist_ok=True)
                        with h5py.File(dataset_path + set_name + "/" + add_to_path + "/" + "wsi_" + str(wsi_idx) +
                                       "_" + str(tma_idx) + "_" + str(patch_idx) + "_" + str(file_front) + "_" +
                                       "_" + str(id_) + "_" + triplet_nbr + ".h5", "w") as f:
                            f.create_dataset(name="input", data=patch_he.astype("uint8"))
                            f.create_dataset(name="output", data=gt_one_hot.astype("float32"))
                    if class_ == "singleclass":
                        os.makedirs(dataset_path + set_name + "/", exist_ok=True)
                        with h5py.File(dataset_path + set_name + "/" + "wsi_" + str(wsi_idx) + "_" + str(tma_idx) + "_"
                                       + str(patch_idx) + "_" + str(file_front) + "_" +
                                       "_" + str(id_) + "_" + triplet_nbr + ".h5", "w") as f:
                            f.create_dataset(name="input", data=patch_he.astype("uint8"))
                            f.create_dataset(name="output", data=gt_one_hot.astype("uint8"))

                # delete streamers and stuff to potentially avoid threading issues in FAST
                del data_fast, generators, streamers
    
                tma_idx += 1


    he_tmas.clear()
    ck_tmas.clear()
    del he_tmas, ck_tmas
    del mask, annot, annot_remove, annot_for_width
    del remove_annot
    del importer_he, importer_ck, importer_mask, importer_annot, importer_remove
    del access, access_annot, access_remove
    del he_tma, ck_tma, he_tma_padded, ck_tma_padded
    # del data

    import gc
    gc.collect()

    #pbar.close()
    print("count", count)


if __name__ == "__main__":
    import os
    # from multiprocessing import Process
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # --- HYPER PARAMS
    plot_flag = False
    plot_flag_test = False
    level = 2  # image pyramid level
    nb_iters = -1
    patch_size = 1024
    downsample_factor = 4  # tested with 8, but not sure if better
    wsi_idx = 0
    dist_limit = 2000  # / 2 ** level  # distance shift between HE and IHC TMA allowed  # @TODO: Check if okay
    overlap = 0.25
    class_ = "multiclass"  # singleclass
    skip_percentage = 0.25

    HE_CK_dir_path = '/data/Maren_P1/data/TMA/cohorts/'

    # paths to wsis included in train and validation sets
    data_splits_path = "./data_splits/250123_093254/dataset_split.h5"

    # path to wsis included in test set
    # @TODO: create patches for test set when time for it

    curr_date = "".join(date.today().strftime("%d/%m").split("/")) + date.today().strftime("%Y")[2:]
    curr_time = "".join(str(datetime.now()).split(" ")[1].split(".")[0].split(":"))
    dataset_path = "/mnt/EncryptedSSD1/maren/datasets/" + curr_date + "_" + curr_time + \
                   "_level_" + str(level) + \
                   "_psize_" + str(patch_size) + \
                   "_ds_" + str(downsample_factor) + "/"

    # define datasets (train/val/test) - always uses predefined dataset
    with h5py.File(data_splits_path, "r") as f:
        train_set = np.array(f['train']).astype(str)
        val_set = np.array(f['val']).astype(str)
        test_set = np.array(f['test']).astype(str)

    # get elements in each dataset
    N_train = len(list(train_set))
    N_val = len(list(val_set))

    file_set = train_set, val_set
    set_names = ['ds_train', 'ds_val']

    print("file set", file_set)
    print("length file set", len(file_set))
    count = 0

    for files in tqdm(file_set, "Cohort"):
        set_name = set_names[count]  # ds_train or ds_val
        for file in tqdm(files, "WSI"):

            file_front = file.split("_EFI_CK")[0]
            id_ = file.split("BC_")[1].split(".tiff")[0]

            HE_path = HE_CK_dir_path + str(file_front) + "/" + str(file_front) + '_EFI_HE_BC_' + str(id_) + '.vsi'
            CK_path = HE_CK_dir_path + str(file_front) + "/" + str(file_front) + '_EFI_CK_BC_' + str(id_) + '.vsi'

            mask_path = '/data/Maren_P1/data/annotations_converted/blue_channel_tiff/' + str(file_front) + \
                        '_EFI_CK_BC_' + str(id_) + '.tiff'
            annot_path = '/data/Maren_P1/data/annotations_converted/TMA/' + str(file_front) + \
                         '_EFI_HE_BC_' + str(id_) + '-labels.ome.tif'
            remove_path = '/data/Maren_P1/data/annotations_converted/remove_TMA/' + str(file_front) \
                          + '_EFI_CK_BC_' + str(id_) + '.vsi - EFI 40x-remove.ome.tif'
            triplet_path = '/data/Maren_P1/data/annotations_converted/triplets_TMA_id/' + str(file_front) \
                           + '_EFI_CK_BC_' + str(id_) + '.vsi - EFI 40x-labels.ome.tif'


            # create dataset for current WSI in a separate process
            # this process will be killed when it is done, hence, all memory will be freed
            inputs_ = [[HE_path, CK_path, mask_path, annot_path, remove_path, triplet_path, dataset_path, set_name,
                       plot_flag, level, nb_iters, patch_size, downsample_factor, wsi_idx, dist_limit, overlap,
                        file_front, id_]]
            p = mp.Pool(1)
            output = p.map(create_datasets_wrapper, inputs_)
            p.terminate()
            p.join()
            del p, inputs_

            wsi_idx += 1

        count += 1