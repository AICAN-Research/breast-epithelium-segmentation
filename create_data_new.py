import fast
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from skimage.registration import phase_cross_correlation
from scipy import ndimage as ndi
import h5py
from datetime import datetime, date
import os
import pandas as pd


def create_datasets(HE_path, CK_path, mask_path, annot_path,
                    file_name, plot_flag, level, nb_iters, patch_size, downsample_factor):

    # fast.Reporter.setGlobalReportMethod(fast.Reporter.COUT)  # verbose

    # import CK and annotated (in qupath) image:
    importerHE = fast.WholeSlideImageImporter.create(
        HE_path)  # path to CK image
    importerCK = fast.WholeSlideImageImporter.create(
        CK_path)  # path to CK image
    importerMask = fast.TIFFImagePyramidImporter.create(
        mask_path)  # path to annotated image
    importerAnnot = fast.TIFFImagePyramidImporter.create(
        annot_path)  # path to annotated image

    # access annotated mask (generated from qupath)
    mask = importerMask.runAndGetOutputData()
    annot = importerAnnot.runAndGetOutputData()
    height_mask = mask.getLevelHeight(level)
    width_mask = mask.getLevelWidth(level)
    access = mask.getAccess(fast.ACCESS_READ)
    accessAnnot = annot.getAccess(fast.ACCESS_READ)

    # plot whole TMA image (does not work on level 0-3, image level too large to convert to FAST image)
    if plot_flag:
        extractor = fast.ImagePyramidLevelExtractor.create(level=4).connect(importerMask)
        image = extractor.runAndGetOutputData()
        numpy_image = np.asarray(image)
        plt.imshow(numpy_image[..., 0], cmap='gray')
        plt.show()

    if plot_flag:
        extractor = fast.ImagePyramidLevelExtractor.create(level=4).connect(importerCK)
        image = extractor.runAndGetOutputData()
        numpy_image = np.asarray(image)
        plt.imshow(numpy_image[..., 0], cmap='gray')
        plt.show()

    if plot_flag:
        extractor = fast.ImagePyramidLevelExtractor.create(level=4).connect(importerAnnot)
        image = extractor.runAndGetOutputData()
        numpy_image = np.asarray(image)
        plt.imshow(numpy_image[..., 0], cmap='jet', interpolation="none")
        plt.show()

    # get CK TMA cores
    extractor = fast.TissueMicroArrayExtractor.create(level=level).connect(importerCK)
    CK_TMAs = []
    for j, TMA in tqdm(enumerate(fast.DataStream(extractor)), "CK TMA:"):
        CK_TMAs.append(TMA)
        # CK_TMAs.append(get_positions(TMA))
        if j == nb_iters:
            break

    # get HE TMA cores
    extractor = fast.TissueMicroArrayExtractor.create(level=level).connect(importerHE)
    HE_TMAs = []
    for j, TMA in tqdm(enumerate(fast.DataStream(extractor)), "HE TMA:"):
        HE_TMAs.append(TMA)
        # HE_TMAs.append(get_positions(TMA))
        if j == nb_iters:
            break

    # init tqdm
    pbar = tqdm(total=max([len(CK_TMAs), len(HE_TMAs)]))
    wsi_idx = 0
    tma_idx = 0
    some_counter = 0
    HE_counter = 0
    CK_counter = 0
    shifts = []
    dist_limit = 2000  # / 2 ** level  # distance shift between HE and IHC TMA allowed
    skip_indx = 0

    """
    # remove badly stained cores:
    def remove_cores():
        pd.read_excel(remove_cores_path)
        print("read")
    
    remove_cores()
    exit()
    """
    while True:
        some_counter += 1
        if HE_counter == len(HE_TMAs) or CK_counter == len(CK_TMAs):
            break

        # update tqdm
        pbar.update(1)
        # position_HE_x, position_HE_y, position_HE_z = HE_TMAs[HE_counter]  # HE TMA at place HE_counter in HE_TMAs. HE_TMA and IHC_TMA are Image objects
        # position_CK_x, position_CK_y, position_CK_z = CK_TMAs[CK_counter]  # IHC TMA at place IHC_counter in IHC_TMAs

        HE_TMA = HE_TMAs[HE_counter]
        CK_TMA = CK_TMAs[CK_counter]

        position_HE = HE_TMA.getTransform().getTranslation()  # position of HE TMA at position HE_counter.
        position_CK = CK_TMA.getTransform().getTranslation()  # position of IHC TMA at position IHC_counter.

        position_HE_x = position_HE[0][0]
        position_HE_y = position_HE[1][0]
        position_HE_z = position_HE[2][0]

        position_CK_x = position_CK[0][0]
        position_CK_y = position_CK[1][0]
        position_CK_z = position_CK[2][0]

        dist_x = position_HE_x - position_CK_x
        dist_y = position_HE_y - position_CK_y

        if np.abs(dist_x) < dist_limit and np.abs(dist_y) < dist_limit:  # if positions are close we have a pair

            HE_counter += 1
            CK_counter += 1

            # skip if one of the "position" cores
            if position_HE_y < 5000:
                continue

            # @TODO: need to get TMA from coordinates
            # CK_TMA = access.getPatchAsImage(int(level), int(position_CK_x), int(position_CK_y), int(width), int(height),
            #                               False)

            CK_TMA = np.asarray(CK_TMA)
            HE_TMA = np.asarray(HE_TMA)

            if (CK_TMA.dtype == "object") or (HE_TMA.dtype == "object"):
                #print("TMA was 'corrupt', either HE or CK")
                continue

            shapes_CK_TMA = CK_TMA.shape
            shapes_HE_TMA = HE_TMA.shape

            height, width, _ = CK_TMA.shape  # need when finding TMA in mask slide
            height_HE, width_HE, _ = HE_TMA.shape

            longest_height = max([shapes_CK_TMA[0], shapes_HE_TMA[0]])
            longest_width = max([shapes_CK_TMA[1], shapes_HE_TMA[1]])

            CK_TMA_padded = np.zeros((longest_height, longest_width, 3), dtype="uint8")
            HE_TMA_padded = np.ones((longest_height, longest_width, 3), dtype="uint8") * 255

            CK_TMA_padded[:CK_TMA.shape[0], :CK_TMA.shape[1]] = CK_TMA
            HE_TMA_padded[:HE_TMA.shape[0], :HE_TMA.shape[1]] = HE_TMA

            # downsample image before registration
            curr_shape = CK_TMA_padded.shape[:2]
            CK_TMA_padded_ds = cv2.resize(CK_TMA_padded,
                                          np.round(np.array(curr_shape) / downsample_factor).astype("int32"),
                                          interpolation=cv2.INTER_NEAREST)
            HE_TMA_padded_ds = cv2.resize(HE_TMA_padded, np.round(np.array(curr_shape) / downsample_factor).astype("int32"),
                                          interpolation=cv2.INTER_NEAREST)

            detected_shift = phase_cross_correlation(HE_TMA_padded_ds, CK_TMA_padded_ds)  # detect shift between IHC and HE

            # print(detected_shift)
            shifts = detected_shift[0]
            shifts[2] = 0

            # scale shifts back and apply to original resolution
            shifts = (np.round(downsample_factor * shifts)).astype("int32")

            tma_padded_shifted = ndi.shift(CK_TMA_padded, shifts, order=0, mode="constant", cval=0, prefilter=False)

            # Pad TMAs:
            x = HE_TMA_padded[:CK_TMA.shape[0], :CK_TMA.shape[1]]
            y = tma_padded_shifted[:CK_TMA.shape[0], :CK_TMA.shape[1]]

            # Get TMA from mask slide
            position_CK_x /= (2 ** level)  # why do I need to do this now, when I didn't before?
            position_CK_y /= (2 ** level)  # why do I need to do this now, when I didn't before?

            position_HE_x /= (2 ** level)  # why do I need to do this now, when I didn't before?
            position_HE_y /= (2 ** level)  # why do I need to do this now, when I didn't before?

            position_CK_y = height_mask - position_CK_y - height

            # get corresponding TMA core in the annotated image as in the CK:
            # get corresponding TMA core in manual annotated image as in the HE:
            # skip TMA cores when area is outside mask area
            if position_CK_x + width > width_mask or position_CK_y + height > height_mask:
                # print("TMA core boundary outside mask boundary")
                continue
            patch = access.getPatchAsImage(int(level), int(position_CK_x), int(position_CK_y), int(width), int(height),
                                           False)
            patch_annot = accessAnnot.getPatchAsImage(int(level), int(position_HE_x), int(position_HE_y), int(width_HE),
                                                      int(height_HE),
                                                      False)
            patch = np.asarray(patch)
            patch_annot = np.asarray(patch_annot)

            # patch = patch[..., 0]  # used to do this, and probably still should
            patch = patch[..., 0:3]
            patch_annot = patch_annot[..., 0:3]
            patch = np.flip(patch, axis=0)  # since annotation is flipped

            mask_TMA_padded = np.zeros((longest_height, longest_width, 3), dtype="uint8")
            annot_TMA_padded = np.zeros((longest_height, longest_width, 3), dtype="uint8")
            mask_TMA_padded[:patch.shape[0], :patch.shape[1]] = patch

            # the correctly placed manual annotation:
            annot_TMA_padded[:patch_annot.shape[0], :patch_annot.shape[1]] = patch_annot

            mask_padded_shifted = ndi.shift(mask_TMA_padded, shifts, order=0, mode="constant", cval=0, prefilter=False)

            # the correctly placed blue channel threshold:
            mask = mask_padded_shifted[:patch.shape[0], :patch.shape[1]]  # should I have CK_TMA.shape here instead?

            # do the same for manual annotations
            annot_TMA_padded = annot_TMA_padded[:patch.shape[0], :patch.shape[1]]

            if plot_flag:
                f, axes = plt.subplots(2, 2, figsize=(30, 30))  # Figure of TMAs
                axes[0, 0].imshow(y)
                axes[0, 1].imshow(mask[..., 0], cmap="gray")
                axes[1, 0].imshow(annot_TMA_padded[..., 0], cmap='jet', interpolation="none")
                axes[1, 0].imshow(x, alpha=0.5)
                axes[1, 1].imshow(mask[..., 0], cmap='gray', interpolation="none")
                axes[1, 1].imshow(annot_TMA_padded[..., 0], alpha=0.5)
                plt.show()

            # Visualize TMAs:
            if plot_flag:
                fig, axes = plt.subplots(2, 2, figsize=(30, 30))  # Figure of TMAs
                axes[0, 0].imshow(y)
                axes[0, 1].imshow(x)
                axes[0, 1].imshow(y, alpha=0.5)
                axes[1, 0].imshow(mask[..., 0], cmap="gray")
                axes[1, 1].imshow(x)
                axes[1, 1].imshow(mask[..., 0], alpha=0.5, cmap="gray")
                plt.show()

            # get each GT annotation as its own binary image + fix manual annotations
            marit_annot = np.asarray(annot_TMA_padded)
            healthy_ep = ((marit_annot == 1) & (mask == 1)).astype("float32")
            in_situ = ((marit_annot == 2) & (mask == 1)).astype("float32")

            # substract fixed healthy and in-sity from invasive tissue
            mask[healthy_ep == 1] = 0
            mask[in_situ == 1] = 0

            data = [x, mask, healthy_ep, in_situ]
            data_fast = [fast.Image.createFromArray(curr) for curr in data]
            generators = [fast.PatchGenerator.create(patch_size, patch_size).connect(0, curr) for curr in data_fast]
            streamers = [fast.DataStream(curr) for curr in generators]

            for patch_idx, (patch_HE, patch_mask, patch_healthy, patch_in_situ) in enumerate(zip(*streamers)):
                # fast to np array
                patch_HE = np.array(patch_HE)
                patch_mask = np.array(patch_mask)[..., 0]
                patch_healthy = np.array(patch_healthy)[..., 0]
                patch_in_situ = np.array(patch_in_situ)[..., 0]

                # create one-hot, one channel for each class
                gt_one_hot = np.stack([1 - patch_mask, patch_mask, patch_healthy, patch_in_situ], axis=-1)

                # check if either of the shapes are empty, if yes, continue
                if (len(patch_HE) == 0) or (len(patch_mask) == 0):
                    continue

                # pad patches with incorrect shape
                if np.array(patch_HE).shape[0] < patch_size or np.array(patch_HE).shape[1] < patch_size:
                    continue
                    patch_HE_padded = np.ones((patch_size, patch_size, 3), dtype="uint8") * 255
                    patch_mask_padded = np.zeros((patch_size, patch_size, 3), dtype="uint8")

                    patch_HE_padded[:patch_HE.shape[0], :patch_HE.shape[1]] = patch_HE.astype("uint8")
                    patch_mask_padded[:patch_mask.shape[0], :patch_mask.shape[1]] = patch_mask.astype("uint8")

                    patch_HE = patch_HE_padded
                    patch_mask = patch_mask_padded

                if plot_flag:
                    fig, ax = plt.subplots(2, 3, figsize=(30, 30))  # Figure of the two patches on top of each other
                    ax[0, 0].imshow(patch_HE)
                    ax[0, 1].imshow(gt_one_hot[..., 0], cmap="gray")
                    ax[0, 2].imshow(gt_one_hot[..., 1], cmap="gray")
                    ax[1, 1].imshow(gt_one_hot[..., 2], cmap="gray")
                    ax[1, 2].imshow(gt_one_hot[..., 3], cmap="gray")
                    plt.show()  # Show the two images on top of each other

                # insert saving patches as hdf5 (h5py) here:
                #with h5py.File(dataset_path + str(wsi_idx) + "_" + str(tma_idx) + "_" + str(patch_idx) + ".h5", "w") as f:
                #    f.create_dataset(name="input", data=patch_HE.astype("uint8"))
                #    f.create_dataset(name="output", data=gt_one_hot.astype("uint8"))

            tma_idx += 1

        elif dist_x > dist_limit and dist_y < dist_limit:  # if HE position has passed IHC position
            CK_counter += 1
        elif dist_y > dist_limit:
            CK_counter += 1
        elif dist_x < -dist_limit and dist_y < dist_limit:
            HE_counter += 1
        elif dist_y < -dist_limit:  # if IHC position has passed HE position
            HE_counter += 1
        else:
            raise ValueError("Logical error in distance calculation")

    pbar.close()


if __name__ == "__main__":
    print("HERHERHER")
    print('------........------')

    # --- HYPER PARAMS
    plot_flag = False
    level = 2  # image pyramid level
    nb_iters = 5
    patch_size = 512
    downsample_factor = 4  # tested with 8, but not sure if better

    mask_path = '/home/maren/workspace/qupath-ck-seg/pyramidal_tiff/'

    files = os.listdir(mask_path)
    nbr_files = len(files)

    # shuffle
    order = np.arange(nbr_files)
    np.random.shuffle(order)
    shuffled_files = [files[x] for x in order]

    # split into train, val and test set
    N = nbr_files
    N_train = int(np.round(N * 0.6))
    N_val = int(np.round(N * 0.2))
    N_test = N - N_train - N_val

    # files in train, validation and test set
    train_set = shuffled_files[:N_train]
    val_set = shuffled_files[N_train:(N_train + N_val)]
    test_set = shuffled_files[(N_train + N_val):]

    # curr_date = "".join(date.today().strftime("%d/%m").split("/")) + date.today().strftime("%Y")[2:]
    # curr_time = "".join(str(datetime.now()).split(" ")[1].split(".")[0].split(":"))

    # dataset path name
    # dataset_path = "./datasets/" + curr_date + "_" + curr_time + \
    #               "_TMASegmentation_level_" + str(level) + \
    #               "_psize_" + str(patch_size) + \
    #               "_ds_" + str(downsample_factor) + "/"

    # create folder if not exists
    # os.makedirs(dataset_path, exist_ok=True)

    file_set = train_set, val_set, test_set
    print("file set", file_set)
    print("length file set", len(file_set))
    file_names = ['train_set', 'val_set', 'test_set']
    count = 0
    for files in file_set:
        file_name = file_names[count]
        for file in files:
            file_front = file.split("_CK")[0]
            id_ = file.split("BC_")[1].split(".tiff")[0]

            HE_path = '/data/Maren_P1/epithelium/HE/' + str(file_front) + '_HE_BC_' + str(id_) + '.vsi'
            CK_path = '/data/Maren_P1/epithelium/CK/' + str(file_front) + '_CK_BC_' + str(id_) + '.vsi'
            mask_path = '/home/maren/workspace/qupath-ck-seg/pyramidal_tiff/' + str(file_front) + '_CK_BC_' + str(
                id_) + '.tiff'
            annot_path = '/home/maren/workspace/qupath-ck-seg/export_annotations_pyramidal_tiff_291122/' \
                         + str(file_front) + '_HE_BC_' + str(id_) + '-labels.ome.tif'

            create_datasets(HE_path, CK_path, mask_path, annot_path,
                    file_name, plot_flag, level, nb_iters, patch_size, downsample_factor)
        count += 1