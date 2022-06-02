import fast
import matplotlib.pyplot as plt
import utils
import numpy as np
from skimage.registration import phase_cross_correlation
from scipy import ndimage as ndi
import cv2
import scipy.ndimage.morphology as morph
from skimage.transform import rescale

#  Import IHC and HE image:
importerIHC = fast.WholeSlideImageImporter.create(
    '/data/Maren_P1/TMA/TP02  pan CK AE1-AE3_01.vsi')
importerHE = fast.WholeSlideImageImporter.create(
    '/data/Maren_P1/TMA/H2 TP02 HE helsnittscan.vsi')

level = 4

"""
# GAmmelt mitt, fungerer å kjøre:
extractor = fast.TissueMicroArrayExtractor.create(level=level).connect(importer)
extractor_HE = fast.TissueMicroArrayExtractor.create(level=level).connect(importer_HE) # Default threshold 0.85, want to lower it? Talk to Erik

#fast.ImageRenderer_create()  # Add opacity to one image when showing two images on top of each other

for tma, tma_HE in fast.DataStream(extractor, extractor_HE):
    print('***')
    tma = np.asarray(tma)
    tma_HE = np.asarray(tma_HE)

    #renderer = fast.ImageRenderer.create(opacity=1.0).connect(tma)  # These three lines does not work per now
    #renderer_HE = fast.ImageRenderer.create(opacity=0.2).connect(tma_HE)
    #window = fast.SimpleWindow2D.create().connect([renderer, renderer_HE]).run()

    # downsample images to assist with feature selection
    #tma = rescale(tma, 0.2, preserve_range=True, multichannel=True).astype("uint8")
    #tma_HE = rescale(tma_HE, 0.2, preserve_range=True, multichannel=True).astype("uint8")

    print(tma.shape, tma.dtype, np.amin(tma), np.amax(tma))
    print(tma_HE.shape, tma_HE.dtype, np.amin(tma_HE), np.amax(tma_HE))

    #tma_gray = cv2.cvtColor(tma, cv2.COLOR_BGR2HSV)[..., 0]
    #tma_HE_gray = cv2.cvtColor(tma_HE, cv2.COLOR_BGR2HSV)[..., 0]

    #tma_gray = tma[..., 0]
    #tma_HE_gray = tma_HE[..., 2]

    print("\n IT UPDATE ")

    # Registration of images (determine correlation):
    a = np.amax((tma.shape[0],tma_HE.shape[0]))
    b = np.amax((tma.shape[1], tma_HE.shape[1]))

    tma_padded = np.zeros((1200, 1200, 3), dtype=tma.dtype).copy()
    tma_HE_padded = tma_padded.copy()

    tma_padded[:tma.shape[0], :tma.shape[1]] = tma
    tma_HE_padded[:tma_HE.shape[0], :tma_HE.shape[1]] = tma_HE

    detected_shift = phase_cross_correlation(tma_HE_padded, tma_padded)
    print(detected_shift)

    tma_padded_shifted = ndi.shift(tma_padded, detected_shift[0])

    fig, ax = plt.subplots(1, 1)  # Figure of the two TMAs on top of each other
    ax.imshow(tma_HE_padded)
    ax.imshow(tma_padded_shifted, alpha=0.5)  # Add opacity
    plt.show()  # Show the two images on top of each other

    f, axes = plt.subplots(2,2)  # (1, 3) if i want to also show registrered image
    axes[0,0].imshow(tma) #axes[0].imshow(tma_gray, cmap="gray")
    axes[0,1].imshow(tma_HE) #axes[1].imshow(tma_HE_gray, cmap="gray")
    #im1Reg, h = utils.alignImages(tma, tma_HE)
    #print(im1Reg.shape)
    #axes[2].imshow(im1Reg, cmap="gray")
    axes[1, 0].imshow(tma_HE)  # not registrered
    axes[1, 0].imshow(tma, alpha = 0.5)  # not registrered
    axes[1,1].imshow(tma_HE_padded)  # registrered
    axes[1,1].imshow(tma_padded_shifted, alpha=0.5)  # registrered
    plt.show()
"""

# FRA ERIK:
# Hent HE TMA
extractor = fast.TissueMicroArrayExtractor.create(level=level).connect(importerHE)

HE_TMAs = []
for TMA in fast.DataStream(extractor):
    HE_TMAs.append(TMA)


TMA_pairs = []
# Hent IHC TMA
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
    # TMAs often approximately 1000*1000 in shape

    print("IHC_TMAs length", len(IHC_TMAs))
    print("HE_TMAs length", len(HE_TMAs))
    #f, axes = plt.subplots(1,2)
    #axes[0].imshow(np.asarray(HE_TMA))
    #axes[1].imshow(np.asarray(IHC_TMA))
    #plt.show()

    position_HE = HE_TMA.getTransform().getTranslation()  # position of HE TMA at position HE_counter. just zero? Should it be center of TMA?
    print("position HE", position_HE)
    #print(position_HE[0])
    #print(position_HE[1])
    print("HE_TMA shape",np.asarray(HE_TMA).shape)

    print('----')
    position_IHC = IHC_TMA.getTransform().getTranslation()  # position of IHC TMA at position IHC_counter. just zero, why?
    print("position IHC", position_IHC)
    print("IHC TMA shape",np.asarray(IHC_TMA).shape)

    #HE_counter += 1
    #print("HE counter: " + str(HE_counter))
    #IHC_counter += 1

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
        f, axes = plt.subplots(1, 2)  # Figure of the two corresponding TMAs
        axes[0].imshow(np.asarray(HE_TMA))
        axes[1].imshow(np.asarray(IHC_TMA))
        plt.show()

        fig, ax = plt.subplots(1, 1)  # Figure of the two TMAs on top of each other
        ax.imshow(HE_TMA)
        ax.imshow(IHC_TMA, alpha=0.5)  # Add opacity
        plt.show()  # Show the two images on top of each other

        IHC_TMA = np.asarray(IHC_TMA)
        HE_TMA = np.asarray(HE_TMA)

        IHC_TMA_padded = np.zeros((1200, 1200, 3), dtype=IHC_TMA.dtype).copy()
        HE_TMA_padded = IHC_TMA_padded.copy()

        IHC_TMA_padded[:IHC_TMA.shape[0], :IHC_TMA.shape[1]] = IHC_TMA
        HE_TMA_padded[:HE_TMA.shape[0], :HE_TMA.shape[1]] = HE_TMA

        detected_shift = phase_cross_correlation(HE_TMA_padded, IHC_TMA_padded)  # detect shift between IHC and HE
        print("detected shift",detected_shift)

        shifts.append(detected_shift)

        tmp = detected_shift[0]
        tmp[2] = 0

        tma_padded_shifted = ndi.shift(IHC_TMA, tmp)

        fig, ax = plt.subplots(1, 1)  # Figure of the two TMAs on top of each other
        ax.imshow(HE_TMA)
        ax.imshow(tma_padded_shifted.astype("uint8"), alpha=0.5)  # Add opacity
        plt.show()  # Show the two images on top of each other

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
    """
    # Sjekk posisjonene, har en av de forbigått den andre?
    if #if positions are close, we have a pair:
        TMA_pairs.append((HE_TMA, IHC_TMA))
        HE_counter += 1
        IHC_counter += 1
    elif position_HE has gone past position_IHC:
        IHC_counter += 1 # Skip current IHC TMA
    elif position_IHC has gone past position_HE:
        HE_counter += 1 # Skip current HE TMA
    else:
        raise ValueError("Not possible..")
    """
# Resultat vil så ligge i lista TMA_pairs
print(len(TMA_pairs))

print()
print(shifts)


'''
# Want only one segmentation, otherwise the patches will not match for HE and IHC, that is why tissueSegmentation_HE is commented out
tissueSegmentation = fast.TissueSegmentation.create(threshold=60).connect(importer)
'''

'''
imageRenderer = fast.ImagePyramidRenderer.create().connect(importer)  # give image to renderer
segRenderer = fast.SegmentationRenderer.create().connect(extractor)
pointRenderer = fast.VertexRenderer.create().connect(extractor,1)
#bbRenderer = fast.BoundingBoxRenderer.create(20).connect(extractor,2)
fast.SimpleWindow2D.create().connect(imageRenderer,segRenderer).run()#,bbRenderer).run()
exit()
'''
'''
for tma, tma_HE in zip(fast.DataStream(extractor), fast.DataStream(extractor_HE)):
    tma = np.asarray(tma)
    print(tma.shape)
    tma_HE = np.asarray(tma_HE)
    print(tma_HE.shape)
    print()
    plt.imshow(tma)
    plt.show()

'''



