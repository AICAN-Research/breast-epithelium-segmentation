"""
Script to convert annotations from geoJSON to geoTIFF, then pyramidal TIFF
Get shape of CK image
Requires numpy version above 1.22.0 (upgrade numpy before running this script)
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

#CK_paths = "/data/Maren_P1/epithelium/CK/"
#geojson_path = "/home/maren/workspace/qupath-ck-seg/export_geojson_291122/"
#tmp_path = "/home/maren/workspace/qupath-ck-seg/tmp_results/"
#output_path = "/home/maren/workspace/qupath-ck-seg/pyramidal_tiff/"

# Need to run this script once for every folder with CK paths
cohorts_path = "/data/Maren_P1/data/TMA/cohorts/"
geojson_path = "/data/Maren_P1/data/annotations_converted/blue_channel_tumor_only/"
tmp_path = "/data/Maren_P1/data/annotations_converted/blue_channel_tumor_only/blue_channel_temp/"
output_path = "/data/Maren_P1/data/annotations_converted/blue_channel_tiff/"

# delete folders
if os.path.exists(tmp_path):
    shutil.rmtree(tmp_path)
#shutil.rmtree(output_path)  # do not remove if images one wants to keep

os.makedirs(tmp_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

wsis = []
for cohort in os.listdir(cohorts_path):
    curr_cohort_path = cohorts_path + cohort + "/"
    if cohort in ["HPA"]:
        for file in os.listdir(curr_cohort_path):
            full_path = curr_cohort_path + file
            # get CK only
            if (".vsi" in full_path) and ("Overview" not in full_path) and ("_CK_" in full_path):
                wsis.append(full_path)
    else:
        continue

print(wsis)

for pathCK in tqdm(wsis, "WSI:"):
    print(pathCK)

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
         "-ot", "Byte", file1, file2, "-co", "COMPRESS=LZW"], stderr=sp.DEVNULL, stdout=sp.DEVNULL)

    # convert from TIFF to pyramidal TIFF
    sp.check_call(["vips", "tiffsave", file2, file3, "--bigtiff", "--tile", "--pyramid",
                   "--compression=lzw"], stderr=sp.DEVNULL, stdout=sp.DEVNULL)  # do I need to change quality of compression? "--Q=" + str(Q2)?

# when finished, delete temporary dir
if os.path.exists(tmp_path):
    shutil.rmtree(tmp_path)

# kill java VM
jb.kill_vm()
