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
import qupath.lib.images.writers.ome.OMEPyramidWriter
import groovy.time.*

// --- SET THESE PARAMETERS ---
double downsample = 1
def label1 = "Benign epithelium"
def label2 = "In situ changes"
def format = "ome.tif"
double ds_factor = 2
// ---------------------------
def start = new Date()
// define output path (relative to project)
def outputDir = "path"
def imageData = getCurrentImageData()
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def path = buildFilePath(outputDir, name + "-labels." + format)

// create an ImageServer where the pixels are derived from annotations
def labelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, ColorTools.WHITE)  // background label, usually 0 or 255
    .downsample(downsample)                // server resolution; should match resolution which tiles are exported
    .addLabel(label1, 1)               // output labels, which labels to export
    .addLabel(label2, 2)             // to add more than one label, add one more addLabel(), change name and value
    .multichannelOutput(false)             // if true, each label refers to a channel of a multichannel binary image(requires for multiclass prob.)
    .build()

new OMEPyramidWriter.Builder(labelServer)
    .downsamples(1, 2, 4, 8, 16)
    .tileSize(512)
    .losslessCompression()
    .build()
    .writePyramid(path)

// reclaim memory - relevant for running this within a RunForProject
Thread.sleep(100);
javafx.application.Platform.runLater {
    getCurrentViewer().getImageRegionStore().cache.clear();
}
Thread.sleep(100);

def end = new Date()
TimeDuration duration = TimeCategory.minus(end, start)
println duration