 /**
 * Script to export annotations as tiled, pyramidal images.
 * By defaults saves image in the OME-TIF format with extension "*.ome.tif".
 * Use downsample 1 to get the full image pyramid, but to remove the largest planes, choose downsample > 1.
 * Each annotation label is assigned an integer in the uint8 image.
 * Change label_name (and value!) to export a different annotation; remember that order matters!
 *
 * Code was based on the implementation by Pete Bankhead:
 * https://qupath.readthedocs.io/en/stable/docs/advanced/exporting_annotations.html#full-labeled-image
 *
 * Support for pyramidization was added by Pete Bankhead using the following fix:
 * https://forum.image.sc/t/exporting-full-labelled-images-of-arbitrary-large-wsis/66708/11?u=andreped
 */

import groovy.time.*

def start = new Date()

// --- SET THESE PARAMETERS ---
def label_name = "Tumor"
def format = "geojson"
// ----------------------------

// define output path (relative to project)
//def outputDir = buildFilePath(PROJECT_BASE_DIR, "export_geojson_291122_test") //031122b")
def outputDir = "/data/Maren_P1/data/annotations_converted/blue_channel_tumor_only/"
def imageData = getCurrentImageData()
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def path = buildFilePath(outputDir, name + "-labels." + format)

// create folder to store output
mkdirs(outputDir)

// convert annotations to GeoJSON and save on disk
def annotations = getAnnotationObjects().findAll{it.getPathClass() == getPathClass(label_name)}
exportObjectsToGeoJson(annotations, path)

def end = new Date()
TimeDuration duration = TimeCategory.minus(end, start)
println duration