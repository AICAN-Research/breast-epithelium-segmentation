"""
Script to convert annotations from geoJSON to geoTIFF, then pyramidal TIFF
Get shape of CK image
"""
import os
import fast
import subprocess as sp
from tqdm import tqdm
import shutil
import javabridge as jb
import bioformats as bf

fast.Reporter.setGlobalReportMethod(fast.Reporter.COUT)

# start java VM
jb.start_vm(class_path=bf.JARS)

# mute BF
log_config = os.path.join(os.path.split(__file__)[0], "resources", "log4j.properties")

# Parameters:
level = 0  # image pyramid level to get size from

CK_paths = "/data/Maren_P1/epithelium/CK/"
geojson_path = "/home/maren/workspace/qupath-ck-seg/export_geojson_291122/"
tmp_path = "/home/maren/workspace/qupath-ck-seg/tmp_results/"
output_path = "/home/maren/workspace/qupath-ck-seg/pyramidal_tiff/"

# delete folders
shutil.rmtree(tmp_path)
#shutil.rmtree(output_path)  # usually keep this, but had one in there already that I wanted to keep

os.makedirs(tmp_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

wsis = []
for file in os.listdir(CK_paths):
    if (".vsi" in file) and ("Overview" not in file):
        wsis.append(CK_paths + file)

# wsis = wsis[:-1]

for pathCK in tqdm(wsis, "WSI:"):
    print(pathCK)
    if not pathCK.endswith("3.vsi"):
        continue

    # get ID
    id_ = pathCK.split("/")[-1].split(".")[0]

    ImageReader = bf.formatreader.make_image_reader_class()

    # with ImageReader() as reader:
    reader = ImageReader()
    reader.setId(pathCK)
    reader.setSeries(0)
    height = reader.getSizeY()
    width = reader.getSizeX()
    reader.close()

    # subprocesses to run terminal commands from python script
    file1 = geojson_path + id_ + ".vsi - EFI 40x-labels.geojson"  # path to geoJSON file
    file2 = tmp_path + id_ + "_nonpyramidal.tif"  # path to tif file (not pyramid)
    file3 = output_path + id_ + ".tiff"  # path to tiff file (pyramid)

    # convert from geoJSON to tiled geoTIFF (TIFF)
    sp.check_call(
        ["gdal_rasterize", "-burn", "1", "-ts", str(width), str(height), "-te", "0", "0", str(width), str(height),
         "-ot", "Byte", file1, file2, "-co", "COMPRESS=LZW"])

    # convert from TIFF to pyramidal TIFF
    sp.check_call(["vips", "tiffsave", file2, file3, "--bigtiff", "--tile", "--pyramid",
                   "--compression=lzw"])  # do I need to change quality of compression? "--Q=" + str(Q2)?

# kill java VM
jb.kill_vm()
