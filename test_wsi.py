import fast
import matplotlib.pyplot as plt
import numpy as np
import cv2

#  Import IHC and HE image:
importer = fast.WholeSlideImageImporter.create(
    '/data/Maren_P1/TMA/TP02  pan CK AE1-AE3_01_plane_0_cm_lzw_jpeg_Q_85.tif')
importer_HE = fast.WholeSlideImageImporter.create('/data/Maren_P1/TMA/H2 TP02 HE helsnittscan_plane_0_cm_lzw_jpeg_Q_85.tif')

tissueSegmentation = fast.TissueSegmentation.create().connect(importer)
#tissueSegmentation_HE = fast.TissueSegmentation.create().connect(importer_HE)

#  Patch generators, one for IHC and one for HE:
#  Level 0 is the whole image, and level 1 has lower resolution etc..
#  It is possible to add overlap in generator for edges (check FAST website)
patchGenerator = fast.PatchGenerator.create(1000, 1000, level=0) \
    .connect(0, importer) \
    .connect(1, tissueSegmentation)

patchGenerator_HE = fast.PatchGenerator.create(1000, 1000, level=0) \
    .connect(0, importer_HE) \
    .connect(1, tissueSegmentation)


# Create subplots of patches with different preprocessing
patch_list = []
patch_list2 = []
patch_list_HE = []
for patch, patch_HE in zip(fast.DataStream(patchGenerator), fast.DataStream(patchGenerator_HE)):
    patch = np.asarray(patch)
    patch_list.append(patch)
    patch_list2.append(patch)
    # print(patch.shape)
    # print(np.unique(patch))
    patch = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)  # rgb to hsv
    patch_list.append(patch)
    # print('---')
    patch = patch[:, :, 1]
    #print(np.amin(patch), np.amax(patch))
    patch_temp = (patch > 40).astype('uint8') # threshold
    patch_list.append(patch_temp)

    patch = cv2.GaussianBlur(patch, (7, 7), 0)  # blur before threshold
    print(np.amin(patch), np.amax(patch))
    patch = (patch > 80).astype('uint8')  # threshold
    patch_list.append(patch)
    patch_list2.append(patch)

    #opencv binaryfillholes
    cmaps = ["viridis", "viridis", "gray", "gray"]
    cmaps1 = ["viridis", "gray"]


    #  Make subplots for different processing
    fig, axes = plt.subplots(2,2, figsize=(10,10))
    for i, cmap in zip(range(2), cmaps1):
        for j, cmap in zip(range(2),cmaps1):
             axes[i,j].imshow(patch_list[i + j * 2], cmap=cmap, interpolation='none')
    plt.show()
    patch_list.clear()


    #  Make subplots for original and blurred before threshold processing
    #fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    #for i, cmap in zip(range(2), cmaps1):
    #    axes[i].imshow(patch_list2[i], cmap=cmap, interpolation='none')
    #plt.show()
    #patch_list2.clear()



    ##  Old subplots of 9 patches
    #patch_list.append(patch)
    #if len(patch_list) == 9:
    #    # Display the 9 last patches
    #    f, axes = plt.subplots(3, 3, figsize=(10, 10))
    #    for i in range(3):
    #        for j in range(3):
    #            axes[i, j].imshow(patch_list[i + j * 3], cmap="gray")  # cmap only works with grayscale
    #    plt.show()
    #    patch_list.clear()
