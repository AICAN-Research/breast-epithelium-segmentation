# Script to convert annotations to geoJSON, then tif, then pyramidal tiff
# Get shape of CK image
import os
import fast

"""
temp = []
temp.append("2.hei")
temp.append("5.hei")
temp.append("1.hei")
print(temp)

temp2 = temp.copy()
for element in temp2:
    element.replace(element)

"""

exit()

# Parameters:
level = 0  # image pyramid level to get size from

path = ''  # path to images to convert
pathCK = 'data/Maren_P1/FIND PATH'  # path to corresponding CK images
image_names = os.listdir(path)

def get_size(image_path):
    importerCK = fast.WholeSlideImageImporter.create(image_path)
    image = importerCK.runAndGetOutputData()
    #access = image.getAccess(fast.ACCESS_READ)  # do I need this, what does it do?
    #numpy_image = np.asarray(image)  # should I do this instead, and then use shape[0] and [1]? instead of getLevelHeight()...?

    height = image.getLevelHeight(level)
    width = image.getLevelWidth(level)
    return height, width


get_size(pathCK)
exit()
heights = []
widths = []
for image in image_names:
    image_path = pathCK + '/image'
    height, width = get_size(image_path)  # get shape of CK image
    heights.append(height)
    widths.append(width)

# Need to make shell script that does this:
geoJSON_folder = 'export_geojson_DATE'  #add date at DATE
geoJSON_list = os.listdir(geoJSON_folder)
tif_folder = 'geojson2tif_results'

# create names for new tif files:
tif_list = geoJSON_list.copy()
for tif in tif_list:
    tif.replace()

#gdal_rasterize -burn 1 -ts height width -ot Byte geoJSON_folder/geoJSON_list[nbr]



# Find out: script with terminal commands? run script in scripts?
# kan gjøre dette i python med subprosess:
sp.check_call(["vips", "tiffsave", out1, out2, "--bigtiff", "--tile", "--pyramid",\
               "--compression=" + comp2, "--Q=" + str(Q2)])
# First convert to geoJSON (this is already done)
# Then convert to tif (need to know image shape to do this correctly)
# Then convert to tiff


