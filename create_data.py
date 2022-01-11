#  Create data patches to train
import numpy as np
import matplotlib.pyplot as plt
import h5py
import fast
import cv2
import os

#storage_path = '/home/maren/workspace/ep-segmentation/data/'
#data_path_HE = '/data/Maren_P1/TMA/ #H2 TP02 HE helsnittscan_plane_0_cm_lzw_jpeg_Q_85.tif'
#data_path_IHC = '/data/Maren_P1/TMA/ #TP02  pan CK AE1-AE3_01_plane_0_cm_lzw_jpeg_Q_85.tif'

#os.makedirs(storage_path, exist_ok=True)
dataset070121 =

importer_IHC = fast.WholeSlideImageImporter.create(data_path_IHC)
importer_HE = fast.WholeSlideImageImporter.create(data_path_HE)

# Want only one segmentation, otherwise the patches will not match for HE and IHC, that is why tissueSegmentation_HE is commented out
tissueSegmentation = fast.TissueSegmentation.create().connect(importer_IHC)

img_size = 512  # 3000
patch_level = 2  # 0

#  Patch generators, one for IHC and one for HE:
#  Level 0 is the whole image, and level 1 has lower resolution etc..
#  It is possible to add overlap in generator for edges (check FAST website)
patchGenerator_IHC = fast.PatchGenerator.create(img_size, img_size, level=patch_level) \
    .connect(0, importer_IHC) \
    .connect(1, tissueSegmentation)

patchGenerator_HE = fast.PatchGenerator.create(img_size, img_size, level=patch_level) \
    .connect(0, importer_HE) \
    .connect(1, tissueSegmentation)

patch_list = []
for patch, patch_HE in zip(fast.DataStream(patchGenerator_IHC), fast.DataStream(patchGenerator_HE)):
    patch = np.asarray(patch)  # original IHC patch
    patch_HE = np.asarray(patch_HE)  # original HE patch
    patch_list.append(patch_HE)

    # Preprocess IHC to make mask
    patch = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)  # rgb to hsv color space
    patch = patch[:, :, 1]  # hue, saturation, value
    patch_median = cv2.medianBlur(patch, 7)  # blur before threshold with median blur
    patch_median = (patch_median > 80).astype('uint8')  # threshold
    patch_list.append(patch_median)

    cmaps = ["viridis", "gray"]

    # Make subplot
    fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    for i, cmap in zip(range(2), cmaps):
       axes[i].imshow(patch_list[i], cmap=cmap, interpolation='none')
    plt.show()
    patch_list.clear()
'''
# Store data per subject in hd5 file
input_data = np.stack(patches_HE)
output_data = np.stack(patches_IHC)
print(input_data.shape)
print(output_data.shape)
file = h5py.File(join(storage_path, subject + '.hd5'), 'w')
file.create_dataset('input/0', data=input_data)
file.create_dataset('output/0', data=output_data)
file.close()
'''