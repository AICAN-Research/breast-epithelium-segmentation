/**
*Script for making thresholded mask in images, thresholding dab channel. Using the pixel classifier dab_seg
*/
import groovy.time.*

def start = new Date()
setImageType('BRIGHTFIELD_H_DAB');
setColorDeconvolutionStains('{"Name" : "H-DAB default", "Stain 1" : "Hematoxylin", "Values 1" : "0.65111 0.70119 0.29049", "Stain 2
" : "DAB", "Values 2" : "0.26917 0.56824 0.77759", "Background" : " 255 255 255"}');
resetSelection();
createAnnotationsFromPixelClassifier("dab_seg2", 25.0, 150.0)

def end = new Date()

TimeDuration duration = TimeCategory.minus(end, start)
println duration