"""
Script to create TMA-core paris and save on disk
The TMAs cores are saved on 40x resolution (image pyramid level 0)
The cores are registrerd and saved as pyrdamidal tiffs.
"""
import fast
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from skimage.registration import phase_cross_correlation
from scipy import ndimage as ndi
from datetime import datetime, date
import h5py


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


def create_tma_pairs(he_path, ck_path, mask_path, annot_path, remove_path, dataset_path, set_name,
                    plot_flag, nb_iters, level, downsample_factor, wsi_idx, dist_limit):

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
    # init tqdm
    # pbar = tqdm(total=max([len(CK_TMAs), len(HE_TMAs)]))
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

                ck_tma_padded = np.zeros((longest_height, longest_width, 3), dtype="uint8")
                he_tma_padded = np.ones((longest_height, longest_width, 3), dtype="uint8") * 255

                # insert ck and he tma in padded array
                ck_tma_padded[:ck_tma.shape[0], :ck_tma.shape[1]] = ck_tma
                he_tma_padded[:he_tma.shape[0], :he_tma.shape[1]] = he_tma

                # skip cores that should be removed
                position_he_x /= (2 ** level)
                position_he_y /= (2 ** level)
                try:
                    remove_annot = access_remove.getPatchAsImage(int(level), int(position_he_x), int(position_he_y),
                                                                 int(width_he), int(height_he), False)
                except RuntimeError as e:
                    print(e)
                    continue

                patch_remove = np.asarray(remove_annot)
                patch_remove = patch_remove[..., :3]  #@TODO: check if this is correct
                remove_tma_padded = np.zeros((longest_height, longest_width, 3), dtype="uint8")
                remove_tma_padded[:patch_remove.shape[0], :patch_remove.shape[1]] = patch_remove
                remove_tma_padded = remove_tma_padded[:patch_remove.shape[0], :patch_remove.shape[1]]

                # continue to next core if the current tma core should be skipped
                if np.count_nonzero(remove_tma_padded) > 0:
                    continue

                # downsample image before registration
                curr_shape = ck_tma_padded.shape[:2]
                ck_tma_padded_ds = cv2.resize(ck_tma_padded,
                                              np.round(np.array(curr_shape) / downsample_factor).astype("int32"),
                                              interpolation=cv2.INTER_NEAREST)
                he_tma_padded_ds = cv2.resize(he_tma_padded, np.round(np.array(curr_shape) / downsample_factor).astype("int32"),
                                              interpolation=cv2.INTER_NEAREST)

                # detect shift between IHC and HE
                #@TODO: try other registration methods. Also, check output from this, what are the two last outputs
                detected_shift = phase_cross_correlation(he_tma_padded_ds, ck_tma_padded_ds)
                shifts = detected_shift[0]
                shifts[2] = 0

                # scale shifts back and apply to original resolution
                shifts = (np.round(downsample_factor * shifts)).astype("int32")

                tma_padded_shifted = ndi.shift(ck_tma_padded, shifts, order=0, mode="constant", cval=0, prefilter=False)

                # Pad TMAs:
                x = he_tma_padded[:ck_tma.shape[0], :ck_tma.shape[1]]
                y = tma_padded_shifted[:ck_tma.shape[0], :ck_tma.shape[1]]

                # Get TMA from mask slide
                position_ck_x /= (2 ** level)
                position_ck_y /= (2 ** level)

                position_ck_y = height_mask - position_ck_y - height_ck

                # get corresponding TMA core in the annotated image as in the CK:
                # get corresponding TMA core in manual annotated image as in the HE:
                # skip TMA cores when area is outside mask area
                if position_ck_x + width_ck > width_mask or position_ck_y + height_ck > height_mask:
                    # print("TMA core boundary outside mask boundary")
                    continue
                if position_he_x + width_he > width_annot or position_he_y + height_he > height_annot:
                    # print("TMA core boundary outside mask boundary")
                    continue

                patch = access.getPatchAsImage(int(level), int(position_ck_x), int(position_ck_y), int(width_ck),
                                               int(height_ck), False)
                patch_annot = access_annot.getPatchAsImage(int(level), int(position_he_x), int(position_he_y),
                                                           int(width_he), int(height_he), False)
                patch = np.asarray(patch)
                patch_annot = np.asarray(patch_annot)

                patch = patch[..., 0]
                patch_annot = patch_annot[..., 0]  # annotation image is RGB (gray) -> keep only first dim to get intensity image -> single class uint8
                patch = np.flip(patch, axis=0)  # since annotation is flipped

                annot_tma_padded = np.zeros((longest_height, longest_width), dtype="uint8")
                mask_tma_padded = np.zeros((longest_height, longest_width), dtype="uint8")

                # the correctly placed manual annotation and dab mask:
                annot_tma_padded[:patch_annot.shape[0], :patch_annot.shape[1]] = patch_annot
                mask_tma_padded[:patch.shape[0], :patch.shape[1]] = patch

                # shift mask
                mask_padded_shifted = ndi.shift(mask_tma_padded, shifts[:2], order=0, mode="constant", cval=0, prefilter=False)

                # the correctly placed dab channel mask:
                mask = mask_padded_shifted[:patch.shape[0], :patch.shape[1]]  # should I have CK_TMA.shape here instead?

                # do the same for manual annotations
                annot_tma_padded = annot_tma_padded[:patch.shape[0], :patch.shape[1]]  # @TODO: is this necessary?

                # get each GT annotation as its own binary image + fix manual annotations
                manual_annot = np.asarray(annot_tma_padded)
                healthy_ep = ((manual_annot == 1) & (mask == 1)).astype("float32")
                in_situ = ((manual_annot == 2) & (mask == 1)).astype("float32")

                # subtract fixed healthy and in situ from invasive tissue
                mask[healthy_ep == 1] = 0
                mask[in_situ == 1] = 0

                # create one-hot encoded image of ground truth core
                mask = minmax(mask)
                healthy_ep = minmax(healthy_ep)
                in_situ = minmax(in_situ)
                gt_one_hot = np.stack([1 - (mask.astype(bool) | healthy_ep.astype(bool) | in_situ.astype(bool)),
                                       mask, healthy_ep, in_situ], axis=-1)

                if plot_flag:
                    fig, ax = plt.subplots(2, 2, figsize=(30, 30))
                    ax[0, 0].imshow(x)
                    ax[0, 1].imshow(gt_one_hot[..., 1], cmap="gray")
                    ax[1, 0].imshow(gt_one_hot[..., 2], cmap="gray")
                    ax[1, 1].imshow(gt_one_hot[..., 3], cmap="gray")
                    plt.show()

                he_path = dataset_path + "tmas_he/" + "_wsi_idx_" + str(wsi_idx) + "_tma_idx_" + str(tma_idx)
                ck_path = dataset_path + "tmas_ck/" + "_wsi_idx_" + str(wsi_idx) + "_tma_idx_" + str(tma_idx)

                # os.makedirs(he_path, exist_ok=True)
                # os.makedirs(ck_path, exist_ok=True)

                # save cores as tiff images
                #he_image = pyvips.Image.new_from_array(large_image)
                #he_image.tiffsave(f'{args.out_dir}/{tiffname}.tif', tile=True, tile_width=args.psize,
                #                     tile_height=args.psize, pyramid=True, compression='lzw', bigtiff=True)

                #ck_image = pyvips.Image.new_from_array(large_image)
                #ck_image.tiffsave(f'{args.out_dir}/{tiffname}.tif', tile=True, tile_width=args.psize,
                #                     tile_height=args.psize, pyramid=True, compression='lzw', bigtiff=True)

                #del he_image, ck_image


if __name__ == "__main__":
    level = 0
    downsample_factor = 4
    nb_iters = 5
    plot_flag = True
    dist_limit = 2000
    wsi_idx = 0

    he_ck_dir_path = '/data/Maren_P1/data/TMA/cohorts/'

    # paths to wsis included in train and validation sets
    data_splits_path = "./data_splits/250123_093254/dataset_split.h5"

    # path to wsis included in test set
    # @TODO: create patches for test set when time for it

    curr_date = "".join(date.today().strftime("%d/%m").split("/")) + date.today().strftime("%Y")[2:]
    curr_time = "".join(str(datetime.now()).split(" ")[1].split(".")[0].split(":"))
    dataset_path = "./datasets_tma_cores/" + curr_date + "_" + curr_time + \
                   "_level_" + str(level) + \
                   "_ds_" + str(downsample_factor) + "/"

    #os.makedirs(dataset_path, exist_ok=True)

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
                             plot_flag, nb_iters, level, downsample_factor, wsi_idx, dist_limit)

            wsi_idx += 1

        count += 1