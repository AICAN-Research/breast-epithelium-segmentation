import fast
import numpy as np
from skimage.registration import phase_cross_correlation
from scipy import ndimage as ndi
import cv2

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
while True:   # Use this, HE_counter < 4 just for testing
#while HE_counter < 4:
    if HE_counter == len(HE_TMAs) or IHC_counter == len(IHC_TMAs):
        break
    HE_TMA = HE_TMAs[HE_counter]  # HE TMA at place HE_counter in HE_TMAs. HE_TMA and IHC_TMA are Image objects
    IHC_TMA = IHC_TMAs[IHC_counter]  # IHC TMA at place IHC_counter in IHC_TMAs

    print("IHC_TMAs length", len(IHC_TMAs))
    print("HE_TMAs length", len(HE_TMAs))

    position_HE = HE_TMA.getTransform().getTranslation()  # position of HE TMA at position HE_counter. just zero? Should it be center of TMA?
    print("position HE", position_HE)
    #print(position_HE[0])
    #print(position_HE[1])
    print("HE_TMA shape",np.asarray(HE_TMA).shape)

    print('----')
    position_IHC = IHC_TMA.getTransform().getTranslation()  # position of IHC TMA at position IHC_counter. just zero, why?
    print("position IHC", position_IHC)
    print("IHC TMA shape",np.asarray(IHC_TMA).shape)

    position_HE_x = position_HE[0]
    position_HE_y = position_HE[1]
    position_HE_z = position_HE[2]

    position_IHC_x = position_IHC[0]
    position_IHC_y = position_IHC[1]
    position_IHC_z = position_IHC[2]

    dist_x = position_HE_x - position_IHC_x
    dist_y = position_HE_y - position_IHC_y

    dist_limit = 500 # distance shift between HE and IHC TMA allowed
    print("position_IHC_x", position_IHC_x)
    print("position_HE_x", position_HE_x)
    print("dist_x", dist_x)
    print("dist_y", dist_y)

    if np.abs(dist_x) < dist_limit and np.abs(dist_y) < dist_limit:  # if positions are close we have a pair
        TMA_pairs.append((HE_TMA, IHC_TMA))
        HE_counter += 1
        IHC_counter += 1
        #f, axes = plt.subplots(1, 2)  # Figure of the two corresponding TMAs
        #axes[0].imshow(np.asarray(HE_TMA))
        #axes[1].imshow(np.asarray(IHC_TMA))
        #plt.show()

        #fig, ax = plt.subplots(1, 1)  # Figure of the two TMAs on top of each other
        #ax.imshow(HE_TMA)
        #ax.imshow(IHC_TMA, alpha=0.5)  # Add opacity
        #plt.show()  # Show the two images on top of each other

        IHC_TMA = np.asarray(IHC_TMA)
        HE_TMA = np.asarray(HE_TMA)

        IHC_TMA_padded = np.zeros((1200, 1200, 3), dtype=IHC_TMA.dtype).copy()
        HE_TMA_padded = IHC_TMA_padded.copy()

        IHC_TMA_padded[:IHC_TMA.shape[0], :IHC_TMA.shape[1]] = IHC_TMA
        HE_TMA_padded[:HE_TMA.shape[0], :HE_TMA.shape[1]] = HE_TMA

        detected_shift = phase_cross_correlation(HE_TMA_padded, IHC_TMA_padded)  # detect shift between IHC and HE
        print("detected shift",detected_shift)

        shifts.append(detected_shift)

        tmp = detected_shift[0]  # x,y,z coordinate
        tmp[2] = 0  # set z coordinate to zero

        tma_padded_shifted = ndi.shift(IHC_TMA, tmp)  # shift IHC tma in the x,y direction given by the shift

        x = HE_TMA_padded[:IHC_TMA.shape[0], :IHC_TMA.shape[1]]  # HE back to original shape
        y = tma_padded_shifted[:IHC_TMA.shape[0], :IHC_TMA.shape[1]]  # IHC back to original shape

        #fig, ax = plt.subplots(1, 1)  # Figure of the two TMAs on top of each other
        #ax.imshow(x)
        #ax.imshow(y.astype("uint8"), alpha=0.5)  # Add opacity
        #plt.show()  # Show the two images on top of each other

        print("y shape", y.shape)
        y_thresh = (y[..., 2] > 127.5).astype(np.uint8)  # Threshold TMA (IHC)
        print("y thresh shape", y_thresh.shape)
        print("y thresh min max", np.amin(y_thresh), np.amax(y_thresh))

        #f, axes = plt.subplots(1, 2)  # Figure of IHC with and without thresholding
        #axes[0].imshow(y.astype("uint8"))  # IHC without thresholding
        #axes[1].imshow(y_thresh, cmap="gray")  # IHC with thresholding
        #plt.show()

        y_hsv = cv2.cvtColor(y, cv2.COLOR_RGB2HSV)  # rgb to hsv color space
        print("y hsv shape", y_hsv.shape)
        y_hsv_1 = y_hsv[:, :, 1]  # hue, saturation, value
        y_hsv_0 = y_hsv[:, :, 0]  # hue, saturation, value
        y_hsv_2 = y_hsv[:, :, 2]  # hue, saturation, value
        print("y hsv 1 min max",np.amin(y_hsv_1), np.amax(y_hsv_1))
        print("y hsv 0 min max", np.amin(y_hsv_0), np.amax(y_hsv_0))
        print("y hsv 2 min max", np.amin(y_hsv_2), np.amax(y_hsv_2))
        print("y hsv channel shape", y_hsv_1.shape)
        print("y hsv channel 0 shape", y_hsv_0.shape)
        print("y hsv channel 2 shape", y_hsv_2.shape)
        y_hsv_temp = (y_hsv > 60).astype('uint8')  # threshold
        print(np.amin(y_hsv_temp), np.amax(y_hsv_temp))

        print("y hsv temp shape",y_hsv_temp.shape)

        #f, axes = plt.subplots(1, 2)  # Figure of IHC with and without threshold
        #axes[0].imshow(y.astype("uint8"))
        #axes[1].imshow(y_hsv_temp, cmap="gray")
        #plt.show()

        exit()


        # resize both to fixed size ex: (512, 512), image bilinear, gt nearest
        x = cv2.resize(x, (512, 512), interpolation=cv2.INTER_LINEAR)

        y = cv2.resize(y, (512, 512), interpolation=cv2.INTER_NEAREST)

        # One-hot TMA (IHC) binary, 01
        y = np.stack([1 - y, y], axis=-1)  # shape (512,512,3) -> shape (512,512,3,
        # 2)

        print(x.shape)
        print(y.shape)
        print(x.shape)

        input("Press ENTER To go next:")
        #exit()

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

    print("length HE TMAs: " + str(len(HE_TMAs)))
    print("length IHC TMAs: " + str(len(IHC_TMAs)))
    print("HE counter: " + str(HE_counter))
    print("IHC counter: " + str(IHC_counter))

print(len(TMA_pairs))
print()
print(shifts)




