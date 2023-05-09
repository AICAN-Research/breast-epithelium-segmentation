"""
Script to create dataset of whole TMA-cores
For evaluation on triplet level
"""

import fast
import numpy as np
import cv2
from skimage.registration import phase_cross_correlation
from scipy import ndimage as ndi
from skimage.exposure import equalize_hist
import h5py
import os


def create_data(he_path, ck_path, dab_mask_path, class_annot_path, triplet_annot_path, remove_annot_path, level,
                nb_iters, dist_limit, downsample_factor, dataset_path, set_name):
    importer_he = fast.WholeSlideImageImporter.create(
        he_path)  # path to he image
    importer_ck = fast.WholeSlideImageImporter.create(
        ck_path)  # path to ck image
    importer_dab_mask = fast.TIFFImagePyramidImporter.create(
        dab_mask_path)  # path to dab mask
    importer_class_annot = fast.TIFFImagePyramidImporter.create(
        class_annot_path)  # path to annotated image of in situ/benign
    importer_triplet_annot = fast.TIFFImagePyramidImporter.create(
        triplet_annot_path)  # path to annotated image of triplets
    importer_remove_annot = fast.TIFFImagePyramidImporter.create(
        remove_annot_path)  # path to annotated image of cores to be removed

    # access annotated mask (generated from QuPath)
    dab_mask = importer_dab_mask.runAndGetOutputData()
    class_annot = importer_class_annot.runAndGetOutputData()
    triplet_annot = importer_triplet_annot.runAndGetOutputData()
    remove_annot = importer_remove_annot.runAndGetOutputData()

    height_mask = dab_mask.getLevelHeight(level)
    width_mask = dab_mask.getLevelWidth(level)

    height_annot = class_annot.getLevelHeieght(level)
    width_annot = class_annot.getLevelWidth(level)

    access_dab_mask = dab_mask.getAccess(fast.ACCESS_READ)
    access_class_annot = class_annot.getAccess(fast.ACCESS_READ)
    access_triplet_annot = triplet_annot.getAccess(fast.ACCESS_READ)
    access_remove_annot = remove_annot.getAccess(fast.ACCESS_READ)

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

    count = 0  # counting number of he/ck TMA pairs that are found

    # loop over HE and CK TMA cores
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

                longest_height = max([shapes_ck_tma[0], shapes_he_tma[0]])
                longest_width = max([shapes_ck_tma[1], shapes_he_tma[1]])

                ck_tma_padded = np.ones((longest_height, longest_width, 3), dtype="uint8") * 255
                he_tma_padded = np.ones((longest_height, longest_width, 3), dtype="uint8") * 255

                ck_tma_padded[:ck_tma.shape[0], :ck_tma.shape[1]] = ck_tma
                he_tma_padded[:he_tma.shape[0], :he_tma.shape[1]] = he_tma

                # skip cores that should be removed
                position_ck_x /= (2 ** level)
                position_ck_y /= (2 ** level)

                try:
                    remove_annot = access_remove_annot.getPatchAsImage(int(level), int(position_ck_x),
                                                                       int(position_ck_y), int(width_ck),
                                                                       int(height_ck), False)
                except RuntimeError as e:
                    print(e)
                    continue
                tma_remove = np.asarray(remove_annot)
                tma_remove = tma_remove[..., :3]
                remove_tma_padded = np.zeros((longest_height, longest_width, 3), dtype="uint8")
                remove_tma_padded[:tma_remove.shape[0], :tma_remove.shape[1]] = tma_remove
                remove_tma_padded = remove_tma_padded[:tma_remove.shape[0], :tma_remove.shape[1]]
                if np.count_nonzero(remove_tma_padded) > 0:
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

                # find dice score and shift to determine if core pair should be skipped
                def dsc(pred, target):
                    intersection = np.sum(pred * target)
                    union = np.sum(pred * pred) + np.sum(target * target)
                    return np.clip((2 * intersection + 1) / (union + 1), 0, 1)

                intensity_away_from_white_thresh = 20

                # tissue segmentation
                ck_tma_padded_shifted_tissue = (np.mean(ck_tma_padded_shifted, axis=-1) < 255 - intensity_away_from_white_thresh).astype("uint8")
                he_tma_padded_tissue = (np.mean(he_tma_padded, axis=-1) < 255 - intensity_away_from_white_thresh).astype("uint8")

                # dice value for tissue segmentation of he and ck
                dsc_value = dsc(he_tma_padded_tissue, ck_tma_padded_shifted_tissue)

                shift_thresh = 200
                # skip tma core if above shift threshold or dsc too low
                if np.abs(shifts[0]) > shift_thresh or np.abs(shifts[1]) > shift_thresh:
                    if dsc_value < 80:
                        continue

                # @TODO: is incorrect? or should I do it:
                #he_core_correctly_placed = he_tma_padded[:longest_height, :longest_width]
                #ck_core_correctly_placed = ck_tma_padded_shifted[:longest_height, :longest_width]

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
                dab_core = access_dab_mask.getPatchAsImage(int(level), int(position_ck_x), int(position_ck_y),
                                                           int(width_ck), int(height_ck), False)
                annot_core = access_class_annot.getPatchAsImage(int(level), int(position_he_x), int(position_he_y),
                                                                int(width_he), int(height_he), False)
                dab_core = np.asarray(dab_core)
                annot_core = np.asarray(annot_core)

                dab_core = dab_core[..., 0]
                annot_core = annot_core[..., 0]  # annotation image is RGB (gray) ->
                                                 # keep only first dim to get intensity image -> single class uint8
                dab_core = np.flip(dab_core, axis=0)  # since dab annotation is flipped

                annot_core_padded = np.zeros((longest_height, longest_width), dtype="uint8")
                dab_core_padded = np.zeros((longest_height, longest_width), dtype="uint8")

                # the correctly placed dab and manual annotation:
                dab_core_padded[:dab_core.shape[0], :dab_core.shape[1]] = dab_core
                annot_core_padded[:annot_core.shape[0], :annot_core.shape[1]] = annot_core

                dab_core_padded_shifted = ndi.shift(dab_core_padded, shifts[:2], order=0, mode="constant", cval=0,
                                                    prefilter=False)

                # @TODO: think about whether just keep padded, otherwise one will be "cut"
                # @TODO: then that will be the same for he_tma_padded that is cut below too
                # the correctly placed dab and manual annot:
                dab_core_correctly_placed = dab_core_padded_shifted[:annot_core.shape[0], :annot_core.shape[1]]
                annot_core_correctly_placed = annot_core_padded[:annot_core.shape[0], :annot_core.shape[1]]

                # get each GT annotation as its own binary image + fix manual annotations
                manual_annot = np.asarray(annot_core_correctly_placed)
                healthy_ep = ((manual_annot == 1) & (dab_core_correctly_placed == 1)).astype("float32")
                in_situ_ep = ((manual_annot == 2) & (dab_core_correctly_placed == 1)).astype("float32")

                # subtract fixed healthy and in situ from invasive tissue
                dab_core_correctly_placed[healthy_ep == 1] = 0
                dab_core_correctly_placed[in_situ_ep == 1] = 0

                one_hot = np.stack(1 - (dab_core_correctly_placed.astype(bool) | healthy_ep.astype(bool) |
                                        in_situ_ep.astype(bool)), dab_core_correctly_placed, healthy_ep, in_situ_ep)

                os.makedirs(dataset_path + set_name + "/", exist_ok=True)  # + add_to_path, exist_ok=True)

                # insert saving patches as hdf5 (h5py) here:
                with h5py.File(dataset_path + set_name + "/" + "wsi_" + str(wsi_idx) + "_" + str(tma_idx) + "_" +
                               str(patch_idx) + ".h5", "w") as f:
                    f.create_dataset(name="input", data=he_tma_padded.astype("uint8"))
                    f.create_dataset(name="output", data=one_hot.astype("uint8"))

            tma_idx += 1