# Code based on vsi2tif by André Pedersen

import subprocess as sp
import os


# choose image to convert
loc = "/data/Maren_P1/epithelium/CK/ECD_EFI_CK_BC_4.vsi"

# output path
out_path = "/home/maren/workspace/qupath-ck-seg/vsi_to_tif/"

# path to bfconvert command line tool
bfconvert_path = "/home/maren/bftools/bfconvert"

# path to showinf command line tool
showinf_path = "/home/maren/bftools/showinf"

os.makedirs(out_path, exist_ok=True)  # does not make folder if exist

# current image
image = loc.split("/")[-1].split(".")[0]

# params
plane = 0  # to convert all (cut-off, all images planes larger and equal to this value are considered from the image)
tz = 512  # tile size

# compression methods
comp1 = "LZW"  # "JPEG-2000" #JPEG #"LZW"
comp2 = "jpeg"  # "jpeg" #"lzw", "jpeg", "deflate" (zip), "none" # <----- Best to use zip-conversion here(?)
Q2 = 85  # default: 75 (quality of compression)

# output paths (mid-step and final converted and compressed output)
file1 = out_path + image + ".tif"
# file2 = out_path + image + "_fixed.tif"

# get metadata
sp.check_call(["sh", showinf_path, "-nopix", loc])

# vsi -> btf
#sp.check_call(["sh", bfconvert_path, "-tilex", str(tz), "-tiley", str(tz), "-nogroup", "-no-upgrade",
#               "-overwrite", "-bigtiff", "-series", str(plane), loc, file1])
sp.check_call(["sh", bfconvert_path, "-tilex", str(tz), "-tiley", str(tz), "-nogroup", "-no-upgrade",
               "-overwrite", "-bigtiff", "-pyramid-resolutions", "6", "-compression", "JPEG-2000", loc, file1])
# -compression, comp1 <- choose no compression instead in this mid-step

# btf -> tif
#sp.check_call(["vips", "tiffsave", file1, file2, "--bigtiff", "--tile", "--pyramid",
#               "--compression=" + comp2, "--Q=" + str(Q2)])

# delete btf file
os.remove(file1)
