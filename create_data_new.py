import fast
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

#fast.Reporter.setGlobalReportMethod(fast.Reporter.COUT)  # verbose

# --- HYPER PARAMS
plot_flag = True
level = 2  # image pyramid level

# want to read all images (oslistdir). Loop over.
# match against HE images

# import CK and annotated (in qupath) image:
importerCK = fast.WholeSlideImageImporter.create(
    '/data/Maren_P1/epithelium/CK/ECD_EFI_CK_BC_4.vsi')  # path to CK image
importerMask = fast.TIFFImagePyramidImporter.create(
    '/home/maren/workspace/qupath-ck-seg/pyramidal_tiff/ECD_EFI_CK_BC_4.tiff')  # path to annotated image

# access annotated mask (generated from qupath)
mask = importerMask.runAndGetOutputData()
access = mask.getAccess(fast.ACCESS_READ)

# plot whole TMA image (does not work on level 0-3, image level too large to convert to FAST image)
if plot_flag:
    extractor = fast.ImagePyramidLevelExtractor.create(level=4).connect(importerMask)
    image = extractor.runAndGetOutputData()
    numpy_image = np.asarray(image)
    plt.imshow(numpy_image[...,0], cmap='gray')
    plt.show()
    exit()

# get CK TMA cores
extractor = fast.TissueMicroArrayExtractor.create(level=level).connect(importerCK)
CK_TMAs = []
for j, TMA in tqdm(enumerate(fast.DataStream(extractor)), "CK TMA:"):
    CK_TMAs.append(TMA)
    if j == 5:
        break

CK_counter = 0
for element in CK_TMAs:

    CK_TMA = CK_TMAs[CK_counter]
    position_CK = CK_TMA.getTransform().getTranslation()  # position of IHC TMA at position IHC_counter. just zero, why?

    position_CK_x = position_CK[0].astype("int32")[0]  # ex [123.] -> 123, needs to be int for getPatchAsImage()
    position_CK_y = position_CK[1].astype("int32")[0]
    position_CK_z = position_CK[2].astype("int32")[0]

    CK_TMA = np.asarray(CK_TMA)
    height, width, _ = CK_TMA.shape
    print(level, position_CK_x, position_CK_y, width, height)
    # get corresponding TMA core in the annotated image as in the CK:
    mask = access.getPatchAsImage(int(level), int(position_CK_x), int(position_CK_y), int(width), int(height), False)
    mask = np.asarray(mask)
    print(mask.shape)
    #exit()

    # plot CK tma core and mask:
    if plot_flag:
        plt.rcParams.update({'font.size': 28})

        f, axes = plt.subplots(1, 2, figsize=(30, 30))  # Figure of patches

        titles = ["CK TMA core", "mask from QuPath, blue channel"]
        axes[0].imshow(CK_TMA, interpolation='none')
        axes[1].imshow(mask, cmap="gray", interpolation='none')

        cnts = 0
        for i in range(2):
                axes[i].set_title(titles[cnts])
                cnts += 1

        plt.tight_layout()
        plt.show()
        exit()

    CK_counter += 1
    if CK_counter > 3:
        exit()
