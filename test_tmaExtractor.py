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
importer = fast.WholeSlideImageImporter.create(
    '/data/Maren_P1/TMA/TP02  pan CK AE1-AE3_01_plane_0_cm_lzw_jpeg_Q_85.tif')
importer_HE = fast.WholeSlideImageImporter.create(
    '/data/Maren_P1/TMA/H2 TP02 HE helsnittscan_plane_0_cm_lzw_jpeg_Q_85.tif')

level = 3
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

    print("\n IT UPDATED ")



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



