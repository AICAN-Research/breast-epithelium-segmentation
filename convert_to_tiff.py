"""
Script to convert annotations to geoJSON, then tif, then pyramidal tiff
Get shape of CK image
"""
import os
import fast
import subprocess as sp
from tqdm import tqdm

fast.Reporter.setGlobalReportMethod(fast.Reporter.COUT)


def get_size(image_path, level):
    importerCK = fast.WholeSlideImageImporter.create(image_path)
    image = importerCK.runAndGetOutputData()

    height = image.getLevelHeight(level)
    width = image.getLevelWidth(level)

    # spacing = image.getSpacing()
    spacing = None  # @TODO: Spacing [1, 1, 1] not found in FAST using cellSens VSI format
    return height, width, spacing


# Parameters:
level = 0  # image pyramid level to get size from

CK_paths = "/data/Maren_P1/epithelium/CK/"
geojson_path = "/home/maren/workspace/qupath-ck-seg/export_geojson_201022/"
output_path = "/home/maren/workspace/qupath-ck-seg/pyramidal_tiff/"
tmp_path = "/home/maren/workspace/qupath-ck-seg/tmp_results/"

wsis = []
for file in os.listdir(CK_paths):
    if (".vsi" in file) and ("Overview" not in file):
        wsis.append(CK_paths + file)

wsis = wsis[::-1]

for pathCK in tqdm(wsis, "WSI:"):
    # get ID
    id_ = pathCK.split("/")[-1].split(".")[0]

    print(pathCK)

    # get image metadata info
    height, width, spacing = get_size(pathCK, level=level)
    print(height, width, spacing)

    # subprocesses to run terminal commands from python script
    file1 = geojson_path + id_ + ".vsi - EFI 40x-labels.geojson"  # path to geoJSON file
    file2 = tmp_path + id_ + "_nonpyramidal.tif"  # path to tif file (not pyramid)
    file3 = output_path + id_ + ".tiff"  # path to tiff file (pyramid)

    # convert from geoJSON to tiled geoTIFF (TIFF)
    sp.check_call(["gdal_rasterize", "-burn", "1", "-ts", str(width), str(height), "-ot",
                   "Byte", file1, file2, "-co", "COMPRESS=LZW"])

    # convert from TIFF to pyramidal TIFF
    sp.check_call(["vips", "tiffsave", file2, file3, "--bigtiff", "--tile", "--pyramid",
                   "--compression=lzw"])  # do I need to change quality of compression? "--Q=" + str(Q2)?

