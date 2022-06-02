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

#  Import IHC and HE image:
importerIHC = fast.WholeSlideImageImporter.create(
    '/data/Maren_P1/TMA/TP02  pan CK AE1-AE3_01.vsi')
importerHE = fast.WholeSlideImageImporter.create(
    '/data/Maren_P1/TMA/H2 TP02 HE helsnittscan.vsi')

level = 4
TMA_pairs = []

# get HE TMA
extractor = fast.TissueMicroArrayExtractor.create(level=level).connect(importerHE)
HE_TMAs = []
for TMA in fast.DataStream(extractor):
    HE_TMAs.append(TMA)

# Get IHC TMA
extractor = fast.TissueMicroArrayExtractor.create(level=level).connect(importerIHC)
IHC_TMAs = []
for TMA in fast.DataStream(extractor):
    IHC_TMAs.append(TMA)

print("IHC_TMAs length", len(IHC_TMAs))
print("HE_TMAs length", len(HE_TMAs))

HE_counter = 0
IHC_counter = 0
shifts = []

#
dataset_path = './datasets/TMASegmentation020622/'
os.makedirs(dataset_path, exist_ok=True)  # creates if not exists already
cnt = 1

while True:   # Use this, HE_counter < 4 just for testing
#while HE_counter < 4:
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

        IHC_TMA_padded = np.zeros((1200, 1200, 3), dtype=IHC_TMA.dtype).copy()
        HE_TMA_padded = IHC_TMA_padded.copy()

        IHC_TMA_padded[:IHC_TMA.shape[0], :IHC_TMA.shape[1]] = IHC_TMA
        HE_TMA_padded[:HE_TMA.shape[0], :HE_TMA.shape[1]] = HE_TMA

        detected_shift = phase_cross_correlation(HE_TMA_padded, IHC_TMA_padded)  # detect shift between IHC and HE

        shifts.append(detected_shift)

        tmp = detected_shift[0]
        tmp[2] = 0

        tma_padded_shifted = ndi.shift(IHC_TMA_padded, tmp)

        # Threshold TMA (IHC)
        tma_padded_shifted = (tma_padded_shifted[..., 2] > 127.5).astype(np.uint8)

        print(HE_TMA_padded.shape, tma_padded_shifted.shape)

        x = HE_TMA_padded[:IHC_TMA.shape[0], :IHC_TMA.shape[1]]
        y = tma_padded_shifted[:IHC_TMA.shape[0], :IHC_TMA.shape[1]]

        print(x.shape, y.shape)
        print()

        # Threshold TMA (IHC)
        tma_padded_shifted = (tma_padded_shifted[..., 2] > 127.5).astype(np.uint8)


        # resize both to fixed size ex: (512, 512), image bilinear, gt nearest
        x = cv2.resize(x, (512, 512), interpolation=cv2.INTER_LINEAR)
        y = cv2.resize(y, (512, 512), interpolation=cv2.INTER_NEAREST)

        # One-hot TMA (IHC) binary, 01
        y = np.stack([1-y, y], axis=-1)

        # add stupid first dim
        #x = np.expand_dims(x, axis=0)
        #y = np.expand_dims(y, axis=0)

        print(x.shape)
        print(y.shape)

        # save TMAs on disk
        with h5py.File(dataset_path + str(cnt) + ".h5", "w") as f:
            f.create_dataset(name="input", data=x)
            f.create_dataset(name="output", data=y)

        cnt += 1

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




