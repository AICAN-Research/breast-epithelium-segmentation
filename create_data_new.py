import fast
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

#fast.Reporter.setGlobalReportMethod(fast.Reporter.COUT)  # verbose

# --- HYPER PARAMS
plot_flag = True
level = 0  # image pyramid level
nb_iters = 5

# want to read all images (oslistdir). Loop over.
# match against HE images
HE_path = '/data/Maren_P1/epithelium/HE/ECD_EFI_HE_BC_4.vsi'
CK_path = '/data/Maren_P1/epithelium/CK/ECD_EFI_CK_BC_4.vsi'
# CK_path = '/home/maren/workspace/qupath-ck-seg/vsi_to_tif/ECD_EFI_CK_BC_4.tif'
mask_path = '/home/maren/workspace/qupath-ck-seg/pyramidal_tiff/ECD_EFI_CK_BC_4_new.tiff'

# import CK and annotated (in qupath) image:
importerHE = fast.WholeSlideImageImporter.create(
    HE_path)  # path to CK image
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

# get HE TMA cores
extractor = fast.TissueMicroArrayExtractor.create(level=level).connect(importerHE)
HE_TMAs = []
for j, TMA in tqdm(enumerate(fast.DataStream(extractor)), "HE TMA:"):
    HE_TMAs.append(TMA)
    if j == nb_iters:
        break

# NEW (from create_data.py):
downsample_factor = 4  # tested with 8, but not sure if better
from skimage.registration import phase_cross_correlation
from scipy import ndimage as ndi
# init tqdm
pbar = tqdm(total=max([len(CK_TMAs), len(HE_TMAs)]))
wsi_idx = 0
tma_idx = 0
some_counter = 0
HE_counter = 0
CK_counter = 0
shifts = []
dist_limit = 2000 / 2 ** level  # distance shift between HE and IHC TMA allowed

while True:  # Use this, HE_counter < 4 just for testing
    # while HE_counter < 4:
    some_counter += 1
    if HE_counter == len(HE_TMAs) or CK_counter == len(CK_TMAs):
        break

    # update tqdm
    pbar.update(1)

    HE_TMA = HE_TMAs[HE_counter]  # HE TMA at place HE_counter in HE_TMAs. HE_TMA and IHC_TMA are Image objects
    CK_TMA = CK_TMAs[CK_counter]  # IHC TMA at place IHC_counter in IHC_TMAs

    position_HE = HE_TMA.getTransform().getTranslation()  # position of HE TMA at position HE_counter.
    position_CK = CK_TMA.getTransform().getTranslation()  # position of IHC TMA at position IHC_counter.

    position_HE_x = position_HE[0]
    position_HE_y = position_HE[1]
    position_HE_z = position_HE[2]

    position_CK_x = position_CK[0]
    position_CK_y = position_CK[1]
    position_CK_z = position_CK[2]

    dist_x = position_HE_x - position_CK_x
    dist_y = position_HE_y - position_CK_y

    if np.abs(dist_x) < dist_limit and np.abs(dist_y) < dist_limit:  # if positions are close we have a pair
        HE_counter += 1
        CK_counter += 1

        CK_TMA = np.asarray(CK_TMA)
        HE_TMA = np.asarray(HE_TMA)

        if (CK_TMA.dtype == "object") or (HE_TMA.dtype == "object"):
            print("TMA was 'corrupt', either HE or CK")
            continue

        print("\n---Counter:")
        print(CK_TMA.shape, CK_TMA.dtype)
        print(HE_TMA.shape, HE_TMA.dtype)

        shapes_CK_TMA = CK_TMA.shape
        shapes_HE_TMA = HE_TMA.shape

        height, width, _ = CK_TMA.shape  # need when finding TMA in mask slide

        longest_height = max([shapes_CK_TMA[0], shapes_HE_TMA[0]])
        longest_width = max([shapes_CK_TMA[1], shapes_HE_TMA[1]])

        CK_TMA_padded = np.zeros((longest_height, longest_width, 3), dtype="uint8")
        HE_TMA_padded = np.ones((longest_height, longest_width, 3), dtype="uint8")*255
        print("CK padded shape", CK_TMA_padded.shape)
        print("CK unique", np.unique(CK_TMA_padded))
        print("HE padded shape", HE_TMA_padded.shape)
        print("HE unique", np.unique(HE_TMA_padded))

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

        position_CK_y = height_mask - position_CK_y - height

        # get corresponding TMA core in the annotated image as in the CK:
        patch = access.getPatchAsImage(int(level), int(position_CK_x), int(position_CK_y), int(width), int(height),
                                       False)
        patch = np.asarray(patch)

        print(patch.shape)
        #patch = patch[..., 0]  # used to do this, and probably still should
        patch = patch[..., 0:3]
        print(patch.shape)

        patch = np.flip(patch, axis=0)  # since annotation is flipped

        mask_TMA_padded = np.zeros((longest_height, longest_width, 3), dtype="uint8")

        mask_TMA_padded[:patch.shape[0], :patch.shape[1]] = patch

        mask_padded_shifted = ndi.shift(mask_TMA_padded, shifts, order=0, mode="constant", cval=0, prefilter=False)

        # Pad TMAs:
        x = HE_TMA_padded[:CK_TMA.shape[0], :CK_TMA.shape[1]]
        y = tma_padded_shifted[:CK_TMA.shape[0], :CK_TMA.shape[1]]
        mask = mask_padded_shifted[:patch.shape[0], :patch.shape[1]]

        # Visualize TMAs:
        if plot_flag:
            f, axes = plt.subplots(2, 2, figsize=(30, 30))  # Figure of TMAs
            #axes[0, 0].imshow(x) HE
            axes[0, 0].imshow(y)
            axes[0, 1].imshow(x)
            axes[0, 1].imshow(y, alpha=0.5)
            axes[1, 0].imshow(mask[..., 0], cmap="gray") #(patch[..., 0], cmap="gray")
            axes[1, 1].imshow(y) #(mask_TMA[..., 0], cmap="gray")
            axes[1, 1].imshow(mask[..., 0], alpha=0.5, cmap="gray") #(y, alpha=0.5)
            plt.show()

        tma_idx += 1
        #exit()

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

print()
print(shifts)


# END NEW

exit()
