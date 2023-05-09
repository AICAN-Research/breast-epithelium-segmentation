"""
Script to create TMA-core pairs from test set and save on disk for evaluation of models
The TMAs cores are saved on "level" resolution (image pyramid level)
The cores are registrerd and saved in hdf5 format.
"""
import fast
import numpy as np
from tqdm import tqdm
from skimage.exposure import equalize_hist
from skimage.morphology import remove_small_holes, binary_dilation, disk
import cv2
from skimage.registration import phase_cross_correlation
from scipy import ndimage as ndi
from datetime import datetime, date
import h5py
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


def create_tma_pairs(he_path, ck_path, mask_path, annot_path, remove_path, dataset_path, set_name,
                    plot_flag, nb_iters, level, downsample_factor, wsi_idx, dist_limit, class_):

    # import fast here to free memory when this is done (if running in a separate process)
    import fast

    # import CK and annotated (in QuPath) image
    importer_he = fast.WholeSlideImageImporter.create(
        he_path)  # path to CK image
    importer_ck = fast.WholeSlideImageImporter.create(
        ck_path)  # path to CK image
    importer_mask = fast.TIFFImagePyramidImporter.create(
        mask_path)  # path to dab image
    importer_annot = fast.TIFFImagePyramidImporter.create(
        annot_path)  # path to annotated image
    importer_remove = fast.TIFFImagePyramidImporter.create(
        remove_path)  # path to remove cores image

    # access annotated mask (generated from QuPath)
    mask = importer_mask.runAndGetOutputData()
    annot = importer_annot.runAndGetOutputData()
    annot_for_width = importer_annot.runAndGetOutputData()
    annot_remove = importer_remove.runAndGetOutputData()

    height_mask = mask.getLevelHeight(level)
    width_mask = mask.getLevelWidth(level)

    height_annot = annot_for_width.getLevelHeight(level)
    width_annot = annot_for_width.getLevelWidth(level)

    access = mask.getAccess(fast.ACCESS_READ)
    access_annot = annot.getAccess(fast.ACCESS_READ)
    access_remove = annot_remove.getAccess(fast.ACCESS_READ)

    # get CK TMA cores at level 0
    extractor = fast.TissueMicroArrayExtractor.create(level=level).connect(importer_ck)
    ck_tmas = []
    ck_stream = fast.DataStream(extractor)
    for j, TMA in enumerate(ck_stream):
        ck_tmas.append(TMA)
        if j == nb_iters:
            break
    del extractor, ck_stream

    # get HE TMA cores at level 0
    extractor = fast.TissueMicroArrayExtractor.create(level=level).connect(importer_he)
    he_tmas = []
    he_stream = fast.DataStream(extractor)
    for j, TMA in enumerate(he_stream):
        he_tmas.append(TMA)
        if j == nb_iters:
            break
    del extractor, he_stream

    print("length HE TMAs:", len(he_tmas))
    print("length CK TMAs:", len(ck_tmas))
    tma_idx = 0

    count = 0

    for he_counter in range(len(he_tmas)):
        for ck_counter in range(len(ck_tmas)):

            he_tma = he_tmas[he_counter]
            ck_tma = ck_tmas[ck_counter]

            position_he = he_tma.getTransform().getTranslation()  # position of HE TMA at position HE_counter.
            position_ck = ck_tma.getTransform().getTranslation()  # position of IHC TMA at position IHC_counter.

            position_he_x = position_he[0][0]
            position_he_y = position_he[1][0]

            position_ck_x = position_ck[0][0]
            position_ck_y = position_ck[1][0]

            dist_x = position_he_x - position_ck_x
            dist_y = position_he_y - position_ck_y

            if np.abs(dist_x) < dist_limit and np.abs(dist_y) < dist_limit:  # if positions are close we have a pair
                count += 1

                # @TODO: why do I need to do this, should not be necessary
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

                # find size of rectangle to fit both ck and he tma core
                longest_height = max([shapes_ck_tma[0], shapes_he_tma[0]])
                longest_width = max([shapes_ck_tma[1], shapes_he_tma[1]])

                ck_tma_padded = np.ones((longest_height, longest_width, 3), dtype="uint8") * 255
                he_tma_padded = np.ones((longest_height, longest_width, 3), dtype="uint8") * 255

                # insert ck and he tma in padded array
                ck_tma_padded[:ck_tma.shape[0], :ck_tma.shape[1]] = ck_tma
                he_tma_padded[:he_tma.shape[0], :he_tma.shape[1]] = he_tma

                # skip cores that should be removed
                position_he_x /= (2 ** level)
                position_he_y /= (2 ** level)
                try:
                    remove_annot = access_remove.getPatchAsImage(int(level), int(position_he_x), int(position_he_y),
                                                                 int(width_ck), int(height_ck), False)
                except RuntimeError as e:
                    print(e)
                    print("HOPPER OVER FOR MYE")
                    continue

                tma_remove = np.asarray(remove_annot)
                # continue to next core if the current tma core should be skipped
                if np.count_nonzero(tma_remove) > 0:
                    continue
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
                # @TODO: it is possible that one stain has half of a cylinder and the other a whole still (not shifted)
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

                # annotation image is RGB (gray) -> keep only first dim to get intensity image -> single class uint8
                dab_core = dab_core[..., 0]
                annot_core = annot_core[..., 0]
                dab_core = np.flip(dab_core, axis=0)  # since dab annotation is flipped

                annot_core_padded = np.zeros((longest_height, longest_width), dtype="uint8")
                dab_core_padded = np.zeros((longest_height, longest_width), dtype="uint8")

                # the correctly placed dab and manual annotation:
                dab_core_padded[:dab_core.shape[0], :dab_core.shape[1]] = dab_core
                annot_core_padded[:annot_core.shape[0], :annot_core.shape[1]] = annot_core

                dab_core_padded_shifted = ndi.shift(dab_core_padded, shifts[:2], order=0, mode="constant", cval=0,
                                                    prefilter=False)

                # cut to match ck and he
                annot_core_padded = annot_core_padded[int(start_h):int(stop_h), int(start_w):int(stop_w)]
                dab_core_padded_shifted = dab_core_padded_shifted[int(start_h):int(stop_h),
                                          int(start_w):int(stop_w)]

                # @TODO: think about whether just keep padded, otherwise one will be "cut"
                # @TODO: then that will be the same for he_tma_padded that is cut below too
                # @TODO: is it possible that the padding will be top left, then below is wrong
                # the correctly placed dab and manual annot:
                # dab_core_correctly_placed = dab_core_padded_shifted[:annot_core.shape[0], :annot_core.shape[1]]
                # annot_core_correctly_placed = annot_core_padded[:annot_core.shape[0], :annot_core.shape[1]]
                # he_core_correctly_placed = he_tma_padded[:annot_core.shape[0], :annot_core.shape[1]]  # is this corrrect??
                # ck_tma_padded_shifted = ck_tma_padded_shifted[:annot_core.shape[0], :annot_core.shape[1]] # is this corrrect??

                # get each GT annotation as its own binary image + fix manual annotations
                manual_annot = np.asarray(annot_core_padded)
                healthy_ep = ((manual_annot == 1) & (dab_core_padded_shifted == 1)).astype("float32")
                in_situ_ep = ((manual_annot == 2) & (dab_core_padded_shifted == 1)).astype("float32")

                if class_ == "multiclass":
                    dab_core_padded_shifted[healthy_ep == 1] = 0
                    dab_core_padded_shifted[in_situ_ep == 1] = 0

                    gt_one_hot = np.stack(
                        [1 - (dab_core_padded_shifted.astype(bool) | healthy_ep.astype(bool) | in_situ_ep.astype(bool)),
                         dab_core_padded_shifted, healthy_ep, in_situ_ep], axis=-1)
                    if np.any(gt_one_hot[..., 0] < 0):
                        [print(np.mean(gt_one_hot[..., iii])) for iii in range(4)]
                        raise ValueError("Negative values occurred in the background class, check the segmentations...")

                    if np.any(np.sum(gt_one_hot, axis=-1) > 1):
                        raise ValueError("One-hot went wrong - multiple classes in the same pixel...")

                    # check if either of the shapes are empty, if yes, continue
                    if (len(healthy_ep) == 0) or (len(dab_core_padded_shifted) == 0):
                        continue

                    # create folder if not exists
                if class_ == "multiclass":
                    os.makedirs(dataset_path + set_name + "/", exist_ok=True)
                    with h5py.File(
                            dataset_path + set_name + "/" + "/" + "wsi_" + str(wsi_idx) + "_" + str(
                                    tma_idx) + ".h5", "w") as f:
                        f.create_dataset(name="input", data=he_tma_padded.astype("uint8"))
                        f.create_dataset(name="output", data=gt_one_hot.astype("float32"))
                if class_ == "singleclass":
                    os.makedirs(dataset_path + set_name + "/", exist_ok=True)
                    with h5py.File(
                            dataset_path + set_name + "/" + "wsi_" + str(wsi_idx) + "_" + str(tma_idx) + "_" +
                            ".h5", "w") as f:
                        f.create_dataset(name="input", data=he_tma_padded.astype("uint8"))
                        f.create_dataset(name="output", data=gt_one_hot.astype("uint8"))


if __name__ == "__main__":
    level = 2
    downsample_factor = 4
    nb_iters = 5
    plot_flag = True
    dist_limit = 2000
    wsi_idx = 0
    class_ = "multiclass"

    he_ck_dir_path = '/data/Maren_P1/data/TMA/cohorts/'

    # paths to wsis included in train and validation sets
    data_splits_path = "./data_splits/250123_093254/dataset_split.h5"

    # path to wsis included in test set

    curr_date = "".join(date.today().strftime("%d/%m").split("/")) + date.today().strftime("%Y")[2:]
    curr_time = "".join(str(datetime.now()).split(" ")[1].split(".")[0].split(":"))
    dataset_path = "./datasets_tma_cores/" + curr_date + "_" + curr_time + \
                    "_level_" + str(level) + \
                    "_ds_" + str(downsample_factor) + "/"

    with h5py.File(data_splits_path, "r") as f:
        test_set = np.array(f['test']).astype(str)

    set_name = test_set
    n_test = len(list(test_set))
    print("n_test: ", n_test)

    for file in tqdm(set_name, "WSI"):

        file_front = file.split("_EFI_CK")[0]
        id_ = file.split("BC_")[1].split(".tiff")[0]

        he_path = he_ck_dir_path + str(file_front) + "/" + str(file_front) + '_EFI_HE_BC_' + str(id_) + '.vsi'
        ck_path = he_ck_dir_path + str(file_front) + "/" + str(file_front) + '_EFI_CK_BC_' + str(id_) + '.vsi'

        mask_path = '/data/Maren_P1/data/annotations_converted/blue_channel_tiff/' + str(file_front) + \
                     '_EFI_CK_BC_' + str(id_) + '.tiff'
        annot_path = '/data/Maren_P1/data/annotations_converted/TMA/' + str(file_front) + \
                      '_EFI_HE_BC_' + str(id_) + '-labels.ome.tif'
        remove_path = '/data/Maren_P1/data/annotations_converted/remove_TMA/' + str(file_front) \
                       + '_EFI_CK_BC_' + str(id_) + '.vsi - EFI 40x-remove.ome.tif'

        dataset_path = dataset_path + "/" + file_front + "/" + file + "/"

        create_tma_pairs(he_path, ck_path, mask_path, annot_path, remove_path, dataset_path, set_name,
                          plot_flag, nb_iters, level, downsample_factor, wsi_idx, dist_limit, class_)

        wsi_idx += 1
