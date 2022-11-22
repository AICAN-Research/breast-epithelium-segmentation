"""
Script to convert annotations to geoJSON, then tif, then pyramidal tiff
Get shape of CK image
"""
import json
import os
import fast
import subprocess as sp
from tqdm import tqdm
import shutil
import numpy as np

fast.Reporter.setGlobalReportMethod(fast.Reporter.COUT)


def get_size(image_path, level):
    importerCK = fast.WholeSlideImageImporter.create(image_path)
    image = importerCK.runAndGetOutputData()

    # get all metadata
    print(image.getMetadata())
    print(np.asarray(image.getMetadata()))
    print(dict(image.getMetadata()))

    # exit()

    # get WSI dimensions
    height = image.getLevelHeight(level)
    width = image.getLevelWidth(level)

    # spacing = image.getSpacing()
    spacing = None  # @TODO: Spacing [1, 1, 1] not found in FAST using cellSens VSI format
    return height, width, spacing


# Parameters:
level = 0  # image pyramid level to get size from

CK_paths = "/data/Maren_P1/epithelium/CK/"
geojson_path = "/home/maren/workspace/qupath-ck-seg/export_geojson_031122b/"
tmp_path = "/home/maren/workspace/qupath-ck-seg/tmp_results/"
output_path = "/home/maren/workspace/qupath-ck-seg/pyramidal_tiff/"

# delete folders
shutil.rmtree(tmp_path)
shutil.rmtree(output_path)

os.makedirs(tmp_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

wsis = []
for file in os.listdir(CK_paths):
    if (".vsi" in file) and ("Overview" not in file):
        wsis.append(CK_paths + file)

# wsis = wsis[:-1]

for pathCK in tqdm(wsis, "WSI:"):
    print(pathCK)
    if not pathCK.endswith("4.vsi"):
        continue
    # get ID
    id_ = pathCK.split("/")[-1].split(".")[0]

    height =  # height from qupath/bioformats
    width = # width from qupath/bioformats

    # subprocesses to run terminal commands from python script
    file1 = geojson_path + id_ + ".vsi - EFI 40x-labels.geojson"  # path to geoJSON file
    file2 = tmp_path + id_ + "_nonpyramidal.tif"  # path to tif file (not pyramid)
    file3 = output_path + id_ + ".tiff"  # path to tiff file (pyramid)

    # convert from geoJSON to tiled geoTIFF (TIFF)
    sp.check_call(["gdal_rasterize", "-burn", "1", "-ts", str(width), str(height), "-te", "0", "0", str(width), str(height), "-ot",
                   "Byte", file1, file2, "-co", "COMPRESS=LZW"])

    # convert from TIFF to pyramidal TIFF
    sp.check_call(["vips", "tiffsave", file2, file3, "--bigtiff", "--tile", "--pyramid",
                   "--compression=lzw"])  # do I need to change quality of compression? "--Q=" + str(Q2)?
