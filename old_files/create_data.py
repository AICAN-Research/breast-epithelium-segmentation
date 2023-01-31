import os
import fast
import matplotlib.pyplot as plt
import numpy as np
from skimage.registration import phase_cross_correlation
from scipy import ndimage as ndi
import cv2
from tqdm import tqdm
from skimage.morphology import binary_dilation, binary_erosion, disk, \
    remove_small_objects, remove_small_holes
from datetime import datetime, date


#  Import IHC and HE image:
importerIHC = fast.WholeSlideImageImporter.create(
    '/data/Maren_P1/TMA/TP02  pan CK AE1-AE3_01.vsi')
importerHE = fast.WholeSlideImageImporter.create(
    '/data/Maren_P1/TMA/H2 TP02 HE helsnittscan.vsi')

# --- HYPER PARAMS FOR PATCH GEN
plot_flag = True
plot_flag_check_overlap = False
patch_size = 512
downsample_factor = 4  # tested with 8, but not sure if better
level = 2  # used to be 4, changed to 2 050822
dist_limit = 2000 / 2 ** level  # distance shift between HE and IHC TMA allowed
# curr_tma_size = int(16384 / 2 ** level)
TMA_pairs = []
# ---

# level 2 - optimal values
# changed min_size to check, optimal 3000? so far.
dilation_radius = 5  # 1 for level 3
area_threshold = 700  # 200 for level 3  # remove_small_holes
min_size = 1000  # 300 for level 3  # remove_small_objects
erosion_radius = 5  # 1 for level 3
hsv_th = 50

# get HE TMA
extractor = fast.TissueMicroArrayExtractor.create(level=level).connect(importerHE)
HE_TMAs = []
for i, TMA in tqdm(enumerate(fast.DataStream(extractor)), "HE TMA"):
    HE_TMAs.append(TMA)
    if i == 20:
        break

# Get IHC TMA
extractor = fast.TissueMicroArrayExtractor.create(level=level).connect(importerIHC)
IHC_TMAs = []
for j, TMA in tqdm(enumerate(fast.DataStream(extractor)), "IHC TMA:"):
    IHC_TMAs.append(TMA)
    if j == 20:
        break

# HE_TMAs = HE_TMAS[3:]  # <- do this to remove silly three redundant TMAs at the top of the WSI

# HE_TMAs = HE_TMAs[53:]
# IHC_TMAs = IHC_TMAs[53:]

print("IHC_TMAs length", len(IHC_TMAs))
print("HE_TMAs length", len(HE_TMAs))

HE_counter = 0
IHC_counter = 0
shifts = []

# init tqdm
pbar = tqdm(total=max([len(IHC_TMAs), len(HE_TMAs)]))

curr_date = "".join(date.today().strftime("%d/%m").split("/")) + date.today().strftime("%Y")[2:]
curr_time = "".join(str(datetime.now()).split(" ")[1].split(".")[0].split(":"))

# dataset path name
dataset_path = "./datasets/" + curr_date + "_" + curr_time + \
               "_TMASegmentation_level_" + str(level) + \
               "_psize_" + str(patch_size) + \
               "_ds_" + str(downsample_factor) + "/"

os.makedirs(dataset_path, exist_ok=True)  # creates if not exists already
wsi_idx = 0
tma_idx = 0
some_counter = 0
while True:  # Use this, HE_counter < 4 just for testing
    # while HE_counter < 4:
    some_counter += 1
    if HE_counter == len(HE_TMAs) or IHC_counter == len(IHC_TMAs):
        break

    # update tqdm
    pbar.update(1)

    HE_TMA = HE_TMAs[HE_counter]  # HE TMA at place HE_counter in HE_TMAs. HE_TMA and IHC_TMA are Image objects
    IHC_TMA = IHC_TMAs[IHC_counter]  # IHC TMA at place IHC_counter in IHC_TMAs

    position_HE = HE_TMA.getTransform().getTranslation()  # position of HE TMA at position HE_counter. just zero? Should it be center of TMA?
    position_IHC = IHC_TMA.getTransform().getTranslation()  # position of IHC TMA at position IHC_counter. just zero, why?

    position_HE_x = position_HE[0]
    position_HE_y = position_HE[1]
    position_HE_z = position_HE[2]

    position_IHC_x = position_IHC[0]
    position_IHC_y = position_IHC[1]
    position_IHC_z = position_IHC[2]

    dist_x = position_HE_x - position_IHC_x
    dist_y = position_HE_y - position_IHC_y

    if np.abs(dist_x) < dist_limit and np.abs(dist_y) < dist_limit:  # if positions are close we have a pair
        TMA_pairs.append((position_HE, position_IHC))
        HE_counter += 1
        IHC_counter += 1

        IHC_TMA = np.asarray(IHC_TMA)
        HE_TMA = np.asarray(HE_TMA)

        if (IHC_TMA.dtype == "object") or (HE_TMA.dtype == "object"):
            print("TMA was 'corrupt', either HE or CK")
            continue

        print("\n---Counter:")
        print(IHC_TMA.shape, IHC_TMA.dtype)
        print(HE_TMA.shape, HE_TMA.dtype)

        shapes_IHC_TMA = IHC_TMA.shape
        shapes_HE_TMA = HE_TMA.shape

        longest_height = max([shapes_IHC_TMA[0], shapes_HE_TMA[0]])
        longest_width = max([shapes_IHC_TMA[1], shapes_HE_TMA[1]])

        IHC_TMA_padded = np.zeros((longest_height, longest_width, 3), dtype="uint8")
        HE_TMA_padded = np.ones((longest_height, longest_width, 3), dtype="uint8")*255
        #HE_TMA_padded = IHC_TMA_padded.copy()
        print("CK padded shape", IHC_TMA_padded.shape)
        print("CK unique", np.unique(IHC_TMA_padded))
        print("HE padded shape", HE_TMA_padded.shape)
        print("HE unique", np.unique(HE_TMA_padded))
        #exit()


        IHC_TMA_padded[:IHC_TMA.shape[0], :IHC_TMA.shape[1]] = IHC_TMA
        HE_TMA_padded[:HE_TMA.shape[0], :HE_TMA.shape[1]] = HE_TMA

        # downsample image before registration
        curr_shape = IHC_TMA_padded.shape[:2]

        IHC_TMA_padded_ds = cv2.resize(IHC_TMA_padded,
                                       np.round(np.array(curr_shape) / downsample_factor).astype("int32"),
                                       interpolation=cv2.INTER_NEAREST)
        HE_TMA_padded_ds = cv2.resize(HE_TMA_padded, np.round(np.array(curr_shape) / downsample_factor).astype("int32"),
                                      interpolation=cv2.INTER_NEAREST)

        detected_shift = phase_cross_correlation(HE_TMA_padded_ds, IHC_TMA_padded_ds)  # detect shift between IHC and HE

        # print(detected_shift)
        shifts = detected_shift[0]
        shifts[2] = 0

        # scale shifts back and apply to original resolution
        shifts = (np.round(downsample_factor * shifts)).astype("int32")

        tma_padded_shifted = ndi.shift(IHC_TMA_padded, shifts, order=0, mode="constant", cval=0, prefilter=False)

        # Pad TMAs:
        x = HE_TMA_padded[:IHC_TMA.shape[0], :IHC_TMA.shape[1]]
        y = tma_padded_shifted[:IHC_TMA.shape[0], :IHC_TMA.shape[1]]
        y_ck = y.copy()
        y_copy = y.copy()

        #if plot_flag:
        #    fig, ax = plt.subplots(1, 1)  # Figure of the two TMAs on top of each other
        #    ax.imshow(x)
        #    ax.imshow(y, alpha=0.5)  # Add opacity
        #    plt.show()  # Show the two images on top of each other

        # Threshold TMA (IHC)
        y_rgb2hsv = cv2.cvtColor(y, cv2.COLOR_RGB2HSV)  # rgb to hsv color space
        y_bgr2hsv = cv2.cvtColor(y, cv2.COLOR_RGB2HSV)  # bgr to hsv color space, not now, needed to compare

        # Saturation channel from the two different hsv images (created from cv2.COLOR_RGB2HSV
        # and cv2.COLOR_BGR2HSV):
        y_rgb2hsv_1 = y_rgb2hsv[:, :, 1]  # saturation channel of hsv from rgb
        y_bgr2hsv_1 = y_bgr2hsv[:, :, 1]  # saturation channel of hsv from bgr

        # Gaussian blur:
        y_rgb2hsv_1 = cv2.GaussianBlur(y_rgb2hsv_1, (5, 5), 0)

        # Otsu thresholding:

        # Thresholding saturation channel from the two different hsv images:
        y_rgb2hsv = (y_rgb2hsv_1 > hsv_th).astype('uint8')  # thresholding hsv from rgb
        y_bgr2hsv = (y_bgr2hsv_1 > hsv_th).astype('uint8')  # thresholding hsv from bgr

        # post-process thresholded rgb2hsv and bgr2hsv CK to generate annotation
        curr_annotation_rgb2hsv = np.array(y_rgb2hsv)
        curr_annotation_bgr2hsv = np.array(y_bgr2hsv)

        result_rgb2hsv = binary_dilation(curr_annotation_rgb2hsv, disk(radius=dilation_radius))
        result_bgr2hsv = binary_dilation(curr_annotation_bgr2hsv, disk(radius=dilation_radius))

        result1_rgb2hsv = remove_small_holes(result_rgb2hsv, area_threshold=area_threshold)
        result1_bgr2hsv = remove_small_holes(result_bgr2hsv, area_threshold=area_threshold)

        result2_rgb2hsv = remove_small_objects(result1_rgb2hsv, min_size=min_size)
        result2_bgr2hsv = remove_small_objects(result1_bgr2hsv, min_size=min_size)

        y_rgb2hsv = binary_erosion(result2_rgb2hsv, disk(radius=erosion_radius)).astype("uint8")
        y_bgr2hsv = binary_erosion(result2_bgr2hsv, disk(radius=erosion_radius)).astype("uint8")

        # Plot RGB2HSV with post-processing:
        if plot_flag:
            plt.rcParams.update({'font.size': 28})
            fig, ax = plt.subplots(2, 3, figsize=(30,30))
            titles = ["CK TMA", "RGB2HSV CK thresholded", "dilation",
                          "fill holes", "remove small obj.", "erosion"]
            ax[0, 0].imshow(y_ck)
            ax[0, 1].imshow(curr_annotation_rgb2hsv, cmap="gray", interpolation='none')
            ax[0, 2].imshow(result_rgb2hsv, cmap="gray", interpolation='none')
            ax[1, 0].imshow(result1_rgb2hsv, cmap="gray", interpolation='none')
            ax[1, 1].imshow(result2_rgb2hsv, cmap="gray", interpolation='none')
            ax[1, 2].imshow(y_rgb2hsv, cmap="gray", interpolation='none')
            cnts = 0
            for i in range(2):
                for j in range(3):
                    ax[i, j].set_title(titles[cnts])
                    cnts += 1

            plt.tight_layout()
            plt.show()

        # Plot BGR2HSV with post-processing:
        if plot_flag:
            plt.rcParams.update({'font.size': 28})
            fig, ax = plt.subplots(2, 3, figsize=(30, 30))
            titles = ["CK TMA", "BGR2HSV CK thresholded", "dilation",
                      "fill holes", "remove small obj.", "erosion"]
            ax[0, 0].imshow(y_ck)
            ax[0, 1].imshow(curr_annotation_bgr2hsv, cmap="gray", interpolation='none')
            ax[0, 2].imshow(result_bgr2hsv, cmap="gray", interpolation='none')
            ax[1, 0].imshow(result1_bgr2hsv, cmap="gray", interpolation='none')
            ax[1, 1].imshow(result2_bgr2hsv, cmap="gray", interpolation='none')
            ax[1, 2].imshow(y_bgr2hsv, cmap="gray", interpolation='none')
            cnts = 0
            for i in range(2):
                for j in range(3):
                    ax[i, j].set_title(titles[cnts])
                    cnts += 1

            plt.tight_layout()
            plt.show()


        # Test red channel instead of hsv saturation
        #y_red_ch = y_ck[:, :, 0]  # red channel (?) from ck image
        #y_red_ch_test = y_ck.copy()
        #y_red_ch_test[:, :, 1] = 0  # R G B
        #y_red_ch_test[:, :, 2] = 0
        #print(y_red_ch_test.shape)

        #y_red_ch = np.array(y_red_ch)
        #y_blue_ch_test = y_ck.copy()
        #y_blue_ch_test[:, :, 0] = 0
        #y_blue_ch_test[:, :, 1] = 0

        # post-processing ck with red channel extraced:
        curr_annotation_red_ch = np.array(y_ck[:, :, 0] < 130)

        curr_annotation_green_ch = np.array(y_ck[:, :, 1] < 130)

        # post-process ck with blue channel extracted:
        curr_annotation_blue_ch = np.array(y_ck[:, :, 2] < 150)
        result_blue_ch = binary_dilation(curr_annotation_blue_ch, disk(radius=dilation_radius))
        result1_blue_ch = remove_small_holes(result_blue_ch, area_threshold=area_threshold)
        result2_blue_ch = remove_small_objects(result1_blue_ch, min_size=min_size)
        y_blue_ch = binary_erosion(result2_blue_ch, disk(radius=erosion_radius)).astype("uint8")

        # Plot blue channel with post-processing steps:
        if plot_flag:
            plt.rcParams.update({'font.size': 28})

            f, axes = plt.subplots(2, 4, figsize=(30, 30))  # Figure of patches
            # print(np.unique(patch_HE))
            # print(np.unique(np.array(patch_CK)[:,:,1]))
            titles = ["CK TMA", "R", "G", "B", "", "R th", "G th",
                      "B th", "erosion"]
            # axes[0, 0].imshow(x, interpolation='none')  # patch 1
            axes[0, 0].imshow(y_ck, interpolation='none')
            axes[0, 1].imshow(y_ck[..., 0], cmap="gray", interpolation='none')  # patch 1
            axes[0, 2].imshow(y_ck[..., 1], cmap="gray", interpolation='none')
            axes[0, 3].imshow(y_ck[..., 2], cmap="gray", interpolation='none')

            axes[1, 1].imshow(curr_annotation_red_ch, cmap="gray", interpolation='none')  # patch 2
            #axes[0, 2].imshow(result_red_ch, cmap="gray", interpolation='none')  # post-procssed segmentation
            axes[1, 2].imshow(curr_annotation_green_ch, cmap="gray", interpolation='none')  #curr_annotation_green_ch
            axes[1, 3].imshow(curr_annotation_blue_ch, cmap="gray", interpolation='none')  #

            cnts = 0
            for i in range(2):
                for j in range(3):
                    axes[i, j].set_title(titles[cnts])
                    cnts += 1

            plt.tight_layout()
            plt.show()
            exit()

        exit()
            # @TODO: use fig.savefig() instead to save figures on disk (set dpi=900 maybe?)
        #exit()
        # Visualize TMAs:
        # if plot_flag:
        #    f, axes = plt.subplots(1, 3)  # Figure of TMAs
        #    axes[0].imshow(HE_TMA)
        #    axes[1].imshow(IHC_TMA)
        #    axes[2].imshow(np.array(y)[:,:,1],cmap="gray")
        #    plt.show()

        # patch generator. Change input y corresponding to which channel or rgb2hsv or bgr2hsv:
        #x = fast.Image.createFromArray(np.asarray(x))
        x = fast.Image.createFromArray(np.asarray(y_copy))
        y = fast.Image.createFromArray(np.asarray(y_blue_ch))

        # tissue_HE = fast.TissueSegmentation.create().connect(x)

        # Use overlap for test, not train
        # Add overlap in .create(, overlapPercent=0.25)
        generator_x = fast.PatchGenerator.create(patch_size, patch_size)\
            .connect(0, x)  # .connect(1, tissue_HE)  # try adding overlap
        generator_y = fast.PatchGenerator.create(patch_size, patch_size)\
            .connect(0, y)  # .connect(1, tissue_HE)  # get error when adding level


        patch_list_HE = []  # list of patches to plot to check overlap
        patch_list_CK = []
        for patch_idx, (patch_HE, patch_CK) in enumerate(
                zip(fast.DataStream(generator_x), fast.DataStream(generator_y))):
            # fast to np array
            patch_HE = np.array(patch_HE)
            patch_CK = np.array(patch_CK)

            # check if either of the shapes are empty, if yes, continue
            if (len(patch_HE) == 0) or (len(patch_CK) == 0):
                print("shape patch", np.array(patch_HE).shape)
                print("shape patch CK", np.array(patch_CK).shape)
                print("Patch was empty in either HE or CK:")
                continue

            # remove redundant channel axis on GT
            patch_CK = np.squeeze(patch_CK, axis=-1)

            # pad patches with incorrect shape
            if np.array(patch_HE).shape[0] < patch_size or np.array(patch_HE).shape[1] < patch_size:
                patch_HE_padded = np.ones((patch_size, patch_size, 3), dtype="uint8")*255
                patch_CK_padded = np.zeros((patch_size, patch_size), dtype="uint8")

                patch_HE_padded[:patch_HE.shape[0], :patch_HE.shape[1]] = patch_HE.astype("uint8")
                patch_CK_padded[:patch_CK.shape[0], :patch_CK.shape[1]] = patch_CK.astype("uint8")

                patch_HE = patch_HE_padded
                patch_CK = patch_CK_padded

            curr_annotation = np.array(patch_CK)

            # One-hot TMA (IHC) binary, 01
            final_gt = np.stack([1 - curr_annotation, curr_annotation], axis=-1)
            patch_list_HE.append(patch_HE)  # add HE patch to list for plot to check overlap
            patch_list_CK.append(patch_CK)

            # Plot to check overlap
            print(len(patch_list_HE))
            if plot_flag_check_overlap and len(patch_list_HE) > 8:
                fig, ax = plt.subplots(3, 3)  # Figure of the two patches on top of each other
                ax[0, 0].imshow(patch_list_HE[0])
                ax[0, 1].imshow(patch_list_HE[1])
                ax[0, 2].imshow(patch_list_HE[2])
                ax[1, 0].imshow(patch_list_HE[3])
                ax[1, 1].imshow(patch_list_HE[4])
                ax[1, 2].imshow(patch_list_HE[5])
                ax[2, 0].imshow(patch_list_HE[6])
                ax[2, 1].imshow(patch_list_HE[7])
                ax[2, 2].imshow(patch_list_HE[8])
                plt.show()  # Show the two images on top of each other
                patch_list_HE.clear()
            if plot_flag_check_overlap and len(patch_list_CK) > 8:
                fig, ax = plt.subplots(3, 3)  # Figure of the two patches on top of each other
                ax[0, 0].imshow(patch_list_CK[0], cmap="gray")
                ax[0, 1].imshow(patch_list_CK[1], cmap="gray")
                ax[0, 2].imshow(patch_list_CK[2], cmap="gray")
                ax[1, 0].imshow(patch_list_CK[3], cmap="gray")
                ax[1, 1].imshow(patch_list_CK[4], cmap="gray")
                ax[1, 2].imshow(patch_list_CK[5], cmap="gray")
                ax[2, 0].imshow(patch_list_CK[6], cmap="gray")
                ax[2, 1].imshow(patch_list_CK[7], cmap="gray")
                ax[2, 2].imshow(patch_list_CK[8], cmap="gray")
                plt.show()  # Show the two images on top of each other
                patch_list_CK.clear()

            print("shape ck patch: ", patch_CK.shape)
            fig, ax = plt.subplots(1, 1)  # Figure of the two patches on top of each other
            ax.imshow(patch_HE)
            ax.imshow(np.array(patch_CK),cmap="gray", alpha=0.5)  # Add opacity
            plt.show()  # Show the two images on top of each other

            # insert saving patches as hdf5 (h5py) here:
            #with h5py.File(dataset_path + str(wsi_idx) + "_" + str(tma_idx) + "_" + str(patch_idx) + ".h5", "w") as f:
            #    f.create_dataset(name="input", data=patch_HE.astype("uint8"))
            #    f.create_dataset(name="output", data=final_gt.astype("uint8"))
            #    f.create_dataset(name="orig_CK", data=patch_CK.astype("uint8"))

        tma_idx += 1
        exit()
        # add stupid first dim
        # x = np.expand_dims(x, axis=0)
        # y = np.expand_dims(y, axis=0)

        # done writing for current patch

    elif dist_x > dist_limit and dist_y < dist_limit:  # if HE position has passed IHC position
        IHC_counter += 1
    elif dist_y > dist_limit:
        IHC_counter += 1
    elif dist_x < -dist_limit and dist_y < dist_limit:
        HE_counter += 1
    elif dist_y < -dist_limit:  # if IHC position has passed HE position
        HE_counter += 1
    else:
        raise ValueError("Logical error in distance calculation")

pbar.close()

print(len(TMA_pairs))
print()
print(shifts)
