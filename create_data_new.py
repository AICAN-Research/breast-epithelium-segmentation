import fast
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

#fast.Reporter.setGlobalReportMethod(fast.Reporter.COUT)  # verbose

# --- HYPER PARAMS
plot_flag = True
level = 0  # image pyramid level
nb_iters = 10

# want to read all images (oslistdir). Loop over.
# match against HE images
CK_path = '/data/Maren_P1/epithelium/CK/ECD_EFI_CK_BC_4.vsi'
# CK_path = '/home/maren/workspace/qupath-ck-seg/vsi_to_tif/ECD_EFI_CK_BC_4.tif'
mask_path = '/home/maren/workspace/qupath-ck-seg/pyramidal_tiff/ECD_EFI_CK_BC_4.tiff'

# import CK and annotated (in qupath) image:
importerCK = fast.WholeSlideImageImporter.create(
    CK_path)  # path to CK image
importerMask = fast.WholeSlideImageImporter.create(
    mask_path)  # path to annotated image

# access annotated mask (generated from qupath)
mask = importerMask.runAndGetOutputData()
height_mask = mask.getLevelHeight(level)
width_mask = mask.getLevelWidth(level)
print("height_mask, width_mask", height_mask, width_mask)
print(height_mask/512, width_mask/512)
access = mask.getAccess(fast.ACCESS_READ)

# Find height of image
importerCK_temp = fast.WholeSlideImageImporter.create(CK_path)
image_temp = importerCK_temp.runAndGetOutputData()

height_slide = image_temp.getLevelHeight(level)
width_slide = image_temp.getLevelWidth(level)
print("height_slide, width_slide", height_slide, width_slide)
print(height_slide/512, width_slide/512)

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

# get CK TMA cores
extractor = fast.TissueMicroArrayExtractor.create(level=level).connect(importerCK)
CK_TMAs = []
for j, TMA in tqdm(enumerate(fast.DataStream(extractor)), "CK TMA:"):
    CK_TMAs.append(TMA)
    if j == nb_iters:
        break

position_y = height_slide
CK_counter = 0
for element in CK_TMAs:
    print('---------')

    CK_TMA = CK_TMAs[CK_counter]
    position_CK = CK_TMA.getTransform().getTranslation()  # position of IHC TMA at position IHC_counter. just zero, why?

    position_CK_x = position_CK[0].astype("int32")[0]  # ex [123.] -> 123, needs to be int for getPatchAsImage()
    position_CK_y = position_CK[1].astype("int32")[0]
    position_CK_z = position_CK[2].astype("int32")[0]

    # print(CK_TMA.getTransform().getMatrix())

    CK_TMA = np.asarray(CK_TMA)
    height, width, _ = CK_TMA.shape

    # position_y = position_y - height  # update y-position due to flip

    position_CK_x /= (2 ** level)
    position_CK_y /= (2 ** level)

    position_CK_y = height_mask - position_CK_y - height

    print(width_mask, height_mask)
    print(width, height)
    print(position_CK_x, position_CK_y, )

    print("CK AND segCK sizes (original):", height_slide, width_slide, height_mask, width_mask)

    # get corresponding TMA core in the annotated image as in the CK:
    patch = access.getPatchAsImage(int(level), int(position_CK_x), int(position_CK_y), int(width), int(height), False)
    patch = np.asarray(patch)

    print(patch.shape)
    patch = patch[..., 0]

    patch = np.flip(patch, axis=0)  # since annotation is flipped

    print(CK_TMA.shape)
    print(np.unique(patch), patch.dtype)
    print("postition y", position_y)

    # plot CK tma core and mask:
    if plot_flag:
        plt.rcParams.update({'font.size': 28})

        f, axes = plt.subplots(1, 3, figsize=(30, 30))  # Figure of patches

        titles = ["CK TMA core", "mask from QuPath, blue channel"]
        axes[0].imshow(CK_TMA, interpolation='none')
        axes[1].imshow(patch, cmap="gray", interpolation='none')
        axes[2].imshow(CK_TMA)
        axes[2].imshow(patch, alpha=0.5, cmap='gray', interpolation='none')

        cnts = 0
        for i in range(2):
                axes[i].set_title(titles[cnts])
                cnts += 1

        plt.tight_layout()
        plt.show()
        # exit()

    CK_counter += 1
    if CK_counter > nb_iters:
        exit()
