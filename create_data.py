import os

import fast
import matplotlib.pyplot as plt
import utils
import numpy as np
from skimage.registration import phase_cross_correlation
from scipy import ndimage as ndi
import cv2
import h5py
import scipy.ndimage.morphology as morph
from skimage.transform import rescale
from tqdm import tqdm

#  Import IHC and HE image:
importerIHC = fast.WholeSlideImageImporter.create(
    '/data/Maren_P1/TMA/TP02  pan CK AE1-AE3_01.vsi')
importerHE = fast.WholeSlideImageImporter.create(
    '/data/Maren_P1/TMA/H2 TP02 HE helsnittscan.vsi')

# --- HYPER PARAMS FOR PATCH GEN
plot_flag = True
patch_size = 512
level = 2  # used to be 4, changed to 2 050822
curr_tma_size = int(16384 / 2 ** level)
TMA_pairs = []
# ---

# get HE TMA
extractor = fast.TissueMicroArrayExtractor.create(level=level).connect(importerHE)
HE_TMAs = []
for TMA in tqdm(fast.DataStream(extractor), "HE TMA"):
    HE_TMAs.append(TMA)

# Get IHC TMA
extractor = fast.TissueMicroArrayExtractor.create(level=level).connect(importerIHC)
IHC_TMAs = []
for TMA in tqdm(fast.DataStream(extractor), "IHC TMA:"):
    IHC_TMAs.append(TMA)

print("IHC_TMAs length", len(IHC_TMAs))
print("HE_TMAs length", len(HE_TMAs))

HE_counter = 0
IHC_counter = 0
shifts = []

#
dataset_path = './datasets/TMASegmentation050822/'
os.makedirs(dataset_path, exist_ok=True)  # creates if not exists already
cnt = 1
some_counter = 0
while True:   # Use this, HE_counter < 4 just for testing
#while HE_counter < 4:
    some_counter += 1
    print(some_counter)
    if HE_counter == len(HE_TMAs) or IHC_counter == len(IHC_TMAs):
        break
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

    dist_limit = 500  # distance shift between HE and IHC TMA allowed

    if np.abs(dist_x) < dist_limit and np.abs(dist_y) < dist_limit:  # if positions are close we have a pair
        TMA_pairs.append((HE_TMA, IHC_TMA))
        HE_counter += 1
        IHC_counter += 1

        IHC_TMA = np.asarray(IHC_TMA)
        HE_TMA = np.asarray(HE_TMA)
        print(IHC_TMA.shape, IHC_TMA.dtype)
        print(HE_TMA.shape, HE_TMA.dtype)
        #exit()
        # THIS NEXT PART, until tma_padded_shifted = ndi... is VERY time consuming at level 0
        #when level = 4, use:
        #IHC_TMA_padded = np.zeros((1200, 1200, 3), dtype=IHC_TMA.dtype).copy()
        #when level = 2, use:
        IHC_TMA_padded = np.zeros((curr_tma_size, curr_tma_size, 3), dtype=IHC_TMA.dtype).copy()
        HE_TMA_padded = IHC_TMA_padded.copy()

        IHC_TMA_padded[:IHC_TMA.shape[0], :IHC_TMA.shape[1]] = IHC_TMA
        HE_TMA_padded[:HE_TMA.shape[0], :HE_TMA.shape[1]] = HE_TMA

        detected_shift = phase_cross_correlation(HE_TMA_padded, IHC_TMA_padded)  # detect shift between IHC and HE

        shifts.append(detected_shift)

        tmp = detected_shift[0]
        tmp[2] = 0

        tma_padded_shifted = ndi.shift(IHC_TMA_padded, tmp)

        # Threshold TMA (IHC)
        #tma_padded_shifted = (tma_padded_shifted[..., 2] > 127.5).astype(np.uint8)

        print(HE_TMA_padded.shape, tma_padded_shifted.shape)

        x = HE_TMA_padded[:IHC_TMA.shape[0], :IHC_TMA.shape[1]]
        y = tma_padded_shifted[:IHC_TMA.shape[0], :IHC_TMA.shape[1]]

        print(x.shape, y.shape)
        print()

        if plot_flag:
            fig, ax = plt.subplots(1, 1)  # Figure of the two TMAs on top of each other
            ax.imshow(x)
            ax.imshow(y, alpha=0.5)  # Add opacity
            plt.show()  # Show the two images on top of each other

        # Threshold TMA (IHC)
        #tma_padded_shifted = (tma_padded_shifted[..., 2] > 127.5).astype(np.uint8)
        y_hsv = cv2.cvtColor(y, cv2.COLOR_RGB2HSV)  # rgb to hsv color space
        y_hsv = y_hsv[:, :, 1]  # hue, saturation, value
        y = (y_hsv > 60).astype('uint8')  # threshold, but which channel?

        # resize both to fixed size ex: (512, 512), image bilinear, gt nearest
        #x = cv2.resize(x, (512, 512), interpolation=cv2.INTER_LINEAR)
        #y = cv2.resize(y, (512, 512), interpolation=cv2.INTER_NEAREST)

        # One-hot TMA (IHC) binary, 01
        y = np.stack([1 - y, y], axis=-1)

        # Visualize TMAs:
        if plot_flag:
            f, axes = plt.subplots(1, 3)  # Figure of TMAs
            axes[0].imshow(HE_TMA)
            axes[1].imshow(IHC_TMA)
            axes[2].imshow(np.array(y)[:,:,1],cmap="gray")
            plt.show()

        # patch generator
        x = fast.Image.createFromArray(np.asarray(x))
        y = fast.Image.createFromArray(np.asarray(y))

        generator_x = fast.PatchGenerator.create(patch_size, patch_size).connect(x) # try adding overlap
        generator_y = fast.PatchGenerator.create(patch_size, patch_size).connect(y)  # get error when adding level

        for patch_HE, patch_CK in zip(fast.DataStream(generator_x), fast.DataStream(generator_y)):
            print("HEI")
            print("shape patch", np.array(patch_HE).shape)
            print("shape patch CK", np.array(patch_CK).shape)

            #pad patches with incorrect shape
            if np.array(patch_HE).shape[0] < patch_size or np.array(patch_HE).shape[1] < patch_size:
                patch_HE_padded = np.zeros((patch_size, patch_size, 3),dtype="uint8")
                patch_CK_padded = np.zeros((patch_size, patch_size, 2),dtype="uint8")

                patch_HE = np.asarray(patch_HE)
                patch_CK = np.asarray(patch_CK)

                patch_HE_padded[:patch_HE.shape[0], :patch_HE.shape[1]] = patch_HE.astype("uint8")
                patch_CK_padded[:patch_CK.shape[0], :patch_CK.shape[1]] = patch_CK.astype("uint8")

                patch_HE = patch_HE_padded
                patch_CK = patch_CK_padded

            # attempting to fix mask -> create human-esque annotations
            from skimage.morphology import binary_dilation, binary_erosion, disk,\
                remove_small_objects, remove_small_holes

            curr_annotation = np.array(patch_CK)[:,:,1]
            result = binary_dilation(curr_annotation, disk(radius=1))
            result1 = remove_small_holes(result, area_threshold=200)
            result2 = remove_small_objects(result1, min_size=300)
            result3 = binary_erosion(result2, disk(radius=1))

            if plot_flag:
                f, axes = plt.subplots(2, 3)  # Figure of patches
                #print(np.unique(patch_HE))
                #print(np.unique(np.array(patch_CK)[:,:,1]))
                titles = ["HE Patch", "IHC thresholded", "dilation",
                          "fill holes", "remove small obj.", "erosion"]
                axes[0, 0].imshow(patch_HE)  # patch 1
                axes[0, 1].imshow(curr_annotation, cmap="gray")  # patch 2
                axes[0, 2].imshow(result, cmap="gray")  # post-procssed segmentation
                axes[1, 0].imshow(result1, cmap="gray")  #
                axes[1, 1].imshow(result2, cmap="gray")  #
                axes[1, 2].imshow(result3, cmap="gray")  #

                cnts = 0
                for i in range(2):
                    for j in range(3):
                        axes[i, j].set_title(titles[cnts])
                        cnts += 1

                plt.tight_layout()
                plt.show()
                #exit()
            """
            fig, ax = plt.subplots(1, 1)  # Figure of the two patches on top of each other
            ax.imshow(patch_HE)
            ax.imshow(np.array(patch_CK)[:,:,1],cmap="gray", alpha=0.5)  # Add opacity
            plt.show()  # Show the two images on top of each other
            """
            #insert saving patches as hdf5 (h5py) here
            #with h5py.File(dataset_path + str(cnt) + ".h5", "w") as f:
            #    f.create_dataset(name="input", data=patch_HE)
            #    f.create_dataset(name="output", data=patch_CK)
            
            #cnt += 1

        # add stupid first dim
        #x = np.expand_dims(x, axis=0)
        #y = np.expand_dims(y, axis=0)
        exit()
        print(x.shape)
        print(y.shape)
        exit()
        # save TMAs on disk
        """ Comment out 050822
        #with h5py.File(dataset_path + str(cnt) + ".h5", "w") as f:
        #    f.create_dataset(name="input", data=x)
        #     f.create_dataset(name="output", data=y)
        
        # cnt += 1
        """
        print(cnt)
        print()

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

print(len(TMA_pairs))
print()
print(shifts)




