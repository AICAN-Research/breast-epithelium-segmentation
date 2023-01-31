import fast
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.ndimage.morphology as morph

from source import utils

#  Import IHC and HE image:
importer = fast.WholeSlideImageImporter.create(
    '/data/Maren_P1/TMA/TP02  pan CK AE1-AE3_01_plane_0_cm_lzw_jpeg_Q_85.tif')
importer_HE = fast.WholeSlideImageImporter.create(
    '/data/Maren_P1/TMA/H2 TP02 HE helsnittscan_plane_0_cm_lzw_jpeg_Q_85.tif')

#extractor = fast.TissueMicroArrayExtractor.create(level=0).connect(importer)

# Want only one segmentation, otherwise the patches will not match for HE and IHC, that is why tissueSegmentation_HE is commented out
tissueSegmentation = fast.TissueSegmentation.create().connect(importer)
# tissueSegmentation_HE = fast.TissueSegmentation.create().connect(importer_HE)

img_size = 512  # 3000
patch_level = 2  # 0


#  Patch generators, one for IHC and one for HE:
#  Level 0 is the whole image, and level 1 has lower resolution etc..
#  It is possible to add overlap in generator for edges (check FAST website)
patchGenerator = fast.PatchGenerator.create(img_size, img_size, level=patch_level) \
    .connect(0, importer) \
    .connect(1, tissueSegmentation)

patchGenerator_HE = fast.PatchGenerator.create(img_size, img_size, level=patch_level) \
    .connect(0, importer_HE) \
    .connect(1, tissueSegmentation)



# Create subplots of patches with different preprocessing
patch_list = []  # list of patches to display in figure with subplots
for patch, patch_HE in zip(fast.DataStream(patchGenerator), fast.DataStream(patchGenerator_HE)):
    patch = np.asarray(patch)  # original IHC patch
    patch_HE = np.asarray(patch_HE)  # original HE patch

    ret, h = utils.alignImages(patch_HE, patch)

    patch_list.append(patch)  # patch list
    patch_list.append(patch_HE)
    # print(patch.shape)
    # print(np.unique(patch))

    patch = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)  # rgb to hsv color space
    # patch_list.append(patch)

    patch = patch[:, :, 1]  # hue, saturation, value
    # print(np.amin(patch), np.amax(patch))
    patch_temp = (patch > 40).astype('uint8')  # threshold
    # patch_list.append(patch_temp)

    patch_gauss = cv2.GaussianBlur(patch, (7, 7), 0)  # blur before threshold with gaussian blur
    print(np.amin(patch_gauss), np.amax(patch_gauss))
    patch_gauss = (patch_gauss > 20).astype('uint8')  # threshold
    # patch_list.append(patch_gauss)

    patch_median = cv2.medianBlur(patch, 7)  # blur before threshold with median blur
    patch_median1 = (patch_median > 60).astype('uint8')  # threshold

    patch_median2 = (patch_median > 80).astype('uint8')  # threshold
    patch_list.append(patch_median2)

    #  fill holes
    patch_fillHoles = morph.binary_fill_holes(patch_median2)
    patch_list.append(patch_fillHoles)

    # opencv binaryfillholes
    cmaps = ["viridis", "viridis", "gray", "gray"]
    cmaps1 = ["viridis", "gray"]
    cmaps3 = ["viridis", "viridis", "gray"]
    cmaps4 = ["viridis", "gray", "gray"]

    # Make subplots for different processing
    fig, axes = plt.subplots(2, 3, figsize=(10, 10))
    for i, cmap in zip(range(2), cmaps1):
        for j, cmap in zip(range(2), cmaps1):
            axes[i, j].imshow(patch_list[i + j * 2], cmap=cmap, interpolation='none')
    axes[0, 2].imshow(ret, interpolation='none')
    plt.show()
    patch_list.clear()

    # Make subplots for HE, IHC and blurred before threshold (patch_list3 and cmaps3):
    #  or make subplots for IHC and two different types of thresholds (patch_list4 and cmaps4):
    # fig, axes = plt.subplots(1, 3, figsize = (10,10))
    # for i, cmap in zip(range(3), cmaps4):
    #   axes[i].imshow(patch_list4[i], cmap=cmap, interpolation='none')
    # plt.show()
    # patch_list4.clear()

    # Make subplots for original and blurred before threshold processing
    # fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    # for i, cmap in zip(range(2), cmaps1):
    #    axes[i].imshow(patch_list2[i], cmap=cmap, interpolation='none')
    # plt.show()
    # patch_list2.clear()

    ##  Old subplots of 9 patches
    # patch_list.append(patch)
    # if len(patch_list) == 9:
    #    # Display the 9 last patches
    #    f, axes = plt.subplots(3, 3, figsize=(10, 10))
    #    for i in range(3):
    #        for j in range(3):
    #            axes[i, j].imshow(patch_list[i + j * 3], cmap="gray")  # cmap only works with grayscale
    #    plt.show()
    #    patch_list.clear()
