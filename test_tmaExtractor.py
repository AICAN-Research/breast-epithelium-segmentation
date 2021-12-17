import fast
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.ndimage.morphology as morph

#  Import IHC and HE image:
importer = fast.WholeSlideImageImporter.create(
    '/data/Maren_P1/TMA/TP02  pan CK AE1-AE3_01_plane_0_cm_lzw_jpeg_Q_85.tif')
importer_HE = fast.WholeSlideImageImporter.create(
    '/data/Maren_P1/TMA/H2 TP02 HE helsnittscan_plane_0_cm_lzw_jpeg_Q_85.tif')

extractor = fast.TissueMicroArrayExtractor.create(level=0).connect(importer)
extractor_HE = fast.TissueMicroArrayExtractor.create(level=0).connect(importer_HE)

# Want only one segmentation, otherwise the patches will not match for HE and IHC, that is why tissueSegmentation_HE is commented out
tissueSegmentation = fast.TissueSegmentation.create(threshold=60).connect(importer)
# tissueSegmentation_HE = fast.TissueSegmentation.create().connect(importer_HE)

'''
imageRenderer = fast.ImagePyramidRenderer.create().connect(importer)  # give image to renderer
segRenderer = fast.SegmentationRenderer.create().connect(extractor)
pointRenderer = fast.VertexRenderer.create().connect(extractor,1)
#bbRenderer = fast.BoundingBoxRenderer.create(20).connect(extractor,2)
fast.SimpleWindow2D.create().connect(imageRenderer,segRenderer).run()#,bbRenderer).run()
exit()
'''

for tma, tma_HE in zip(fast.DataStream(extractor), fast.DataStream(extractor_HE)):
    tma = np.asarray(tma)
    print(tma.shape)
    tma_HE = np.asarray(tma_HE)
    print(tma_HE.shape)
    print()
    plt.imshow(tma)
    plt.show()





