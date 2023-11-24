"""
Script to create TMA pairs from slide that has been corrected by pathologists
Similar to create_tma_pairs.py, but no CK stained slide is necessary, as the invasive, benign and in situ masks
have been created with a preliminary model.
"""

import numpy as np
from datetime import datetime, date
import h5py
import os
import matplotlib.pyplot as plt


def create_tma_pairs(he_path, mask_path, remove_path, triplet_path, dataset_path, plot_flag, nb_iters, level, set_name):

    # import fast here to free memory when this is done (if running in a separate process)
    import fast

    # import CK and annotated (in QuPath) image
    importer_he = fast.WholeSlideImageImporter.create(
        he_path)  # path to CK image
    importer_mask = fast.TIFFImagePyramidImporter.create(
        mask_path)  # path to dab image
    importer_remove = fast.TIFFImagePyramidImporter.create(
        remove_path)  # path to remove cores image
    importer_triplet = fast.TIFFImagePyramidImporter.create(
        triplet_path)  # path to annotated triplet image

    # access annotated mask (generated from QuPath)
    mask = importer_mask.runAndGetOutputData()
    annot_remove = importer_remove.runAndGetOutputData()
    annot_triplet = importer_triplet.runAndGetOutputData()

    access = mask.getAccess(fast.ACCESS_READ)
    access_remove = annot_remove.getAccess(fast.ACCESS_READ)
    access_triplet = annot_triplet.getAccess(fast.ACCESS_READ)

    if plot_flag:
        triplet_annot_ = annot_triplet.getAccess(fast.ACCESS_READ).getLevelAsImage(level=4)
        mask_annot_ = mask.getAccess(fast.ACCESS_READ).getLevelAsImage(level=4)
        remove_annot_ = annot_remove.getAccess(fast.ACCESS_READ).getLevelAsImage(level=4)

        plt.rcParams.update({'font.size': 28})
        f, axes = plt.subplots(2, 2, figsize=(30, 30))
        axes[0, 0].imshow(np.asarray(mask_annot_))
        axes[0, 1].imshow(np.asarray(triplet_annot_))
        axes[1, 0].imshow(np.asarray(remove_annot_))
        plt.show()

    # get HE TMA cores at level 1 (20x)
    extractor = fast.TissueMicroArrayExtractor.create(level=level).connect(importer_he)
    he_tmas = []
    he_stream = fast.DataStream(extractor)
    for j, TMA in enumerate(he_stream):
        he_tmas.append(TMA)
        if j == nb_iters:
            break
    del extractor, he_stream

    tma_idx = 0

    for he_counter in range(len(he_tmas)):
        he_tma = he_tmas[he_counter]
        position_he = he_tma.getTransform().getTranslation()
        position_he_x = position_he[0][0]
        position_he_y = position_he[1][0]

        try:
            he_tma = np.asarray(he_tma)
        except RuntimeError as e:
            print(e)
            continue

        shapes_he_tma = he_tma.shape
        height_he, width_he, _ = he_tma.shape

        position_he_x /= (2 ** level)
        position_he_y /= (2 ** level)

        # skip cores that should be removed
        try:
            remove_annot = access_remove.getPatchAsImage(int(level), int(position_he_x), int(position_he_y),
                                                         int(width_he), int(height_he), False)
        except RuntimeError as e:
            print(e)
            continue

        tma_remove = np.asarray(remove_annot)
        # continue to next core if the current tma core should be skipped
        if np.count_nonzero(tma_remove) > 0:
            continue

        # get id number for triplet
        try:
            triplet_annot = access_triplet.getPatchAsImage(int(level), int(position_he_x),
                                                           int(position_he_y),
                                                           int(width_he), int(height_he), False)
        except RuntimeError as e:
            print(e)
            continue
        triplet = np.asarray(triplet_annot)
        triplet_nbr = np.amax(triplet)

        # if no triplet is marked, ex if cylinder moved so much it is difficult to determine triplet
        if triplet_nbr == 0:
            continue

        seg_core = access.getPatchAsImage(int(level), int(position_he_x), int(position_he_y), int(width_he),
                                          int(height_he), False)
        seg_core = np.asarray(seg_core)

        # annotation image is RGB (gray) -> keep only first dim to get intensity image -> single class uint8
        seg_core = seg_core[..., 0]

        # get each GT annotation as its own binary image + fix manual annotations
        invasive_ep = (seg_core == 1).astype("uint8")
        healthy_ep = (seg_core == 2).astype("uint8")
        in_situ_ep = (seg_core == 3).astype("uint8")

        gt_one_hot = np.stack(
            [1 - (invasive_ep.astype(bool) | healthy_ep.astype(bool) | in_situ_ep.astype(bool)),
             invasive_ep, healthy_ep, in_situ_ep], axis=-1)

        if plot_flag:
            print("gt one hot shape: ", gt_one_hot.shape)
            plt.rcParams.update({'font.size': 28})
            f, axes = plt.subplots(2, 2, figsize=(30, 30))
            axes[0, 0].imshow(he_tma)
            axes[0, 1].imshow(gt_one_hot[..., 1])
            axes[1, 0].imshow(gt_one_hot[..., 2])
            axes[1, 1].imshow(gt_one_hot[..., 3])
            plt.show()

        if np.any(gt_one_hot[..., 0] < 0):
            [print(np.mean(gt_one_hot[..., iii])) for iii in range(4)]
            raise ValueError("Negative values occurred in the background class, check the segmentations...")

        if np.any(np.sum(gt_one_hot, axis=-1) > 1):
            raise ValueError("One-hot went wrong - multiple classes in the same pixel...")

        # check if either of the shapes are empty, if yes, continue
        if (len(healthy_ep) == 0) or (len(invasive_ep) == 0):
            continue

        # create folder if not exists
        os.makedirs(dataset_path + set_name + "/" , exist_ok=True)
        with h5py.File(dataset_path + set_name + "/" + "wsi_" +
                       "_" + str(tma_idx) + "_" + str(file_front) +
                       "_" + str(triplet_nbr) + ".h5", "w") as f:
            f.create_dataset(name="input", data=he_tma.astype("uint8"))
            f.create_dataset(name="output", data=gt_one_hot.astype("float32"))

        tma_idx += 1


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    level = 1
    nb_iters = -1
    plot_flag = True
    set_name = "test_corrected"

    curr_date = "".join(date.today().strftime("%d/%m").split("/")) + date.today().strftime("%Y")[2:]
    curr_time = "".join(str(datetime.now()).split(" ")[1].split(".")[0].split(":"))
    dataset_path = "./datasets_tma_cores/" + curr_date + "_" + curr_time + \
                   "_level_" + str(level) + "/"

    he_dir_path = '/path/to/he/'
    remove_path_ = '/path/to/remove/'

    for file in os.listdir(he_dir_path):
        if ".vsi" in file and "Overview" not in file:
            print("file: ", file)
            file_front = file.split(".vsi")[0]
            he_path = he_dir_path + str(file)
            mask_path = '/path/to/mask//' + str(file_front) \
                        + '.vsi - EFI 40x-labels.ome.tif'
            remove_path = remove_path_ + str(file_front) + '.vsi - EFI 40x-remove.ome.tif'
            triplet_path = '/path/to/triplets/' + str(file_front) \
                           + '.vsi - EFI 40x-labels.ome.tif'

            print(he_path)
            print(mask_path)
            print(remove_path)
            print(triplet_path)

            create_tma_pairs(he_path, mask_path, remove_path, triplet_path, dataset_path, plot_flag, nb_iters, level,
                             set_name)


