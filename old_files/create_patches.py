import numpy as np

# Create patches from extracted TMAs (or should I do this earlier)?

# Instead, try to make function to divide before convert to hdf5 in create_data.py
# Remove resize in create_data.py

# Look at size for patches
# Padding
# Think about removing patches with too little tissue
# Make fool-proof, if patch size larger than image/gt, return image/gt

def patches(image, gt, size): #size = patch size
    image_patches = []
    gt_patches = []

    row = image.shape[0]
    col = image.shape[1]


    # patches without overlap, make them with overlap also (50 percent?)
    for i in range(np.floor(row/size).astype("uint8")):
        for j in range(np.floor(col/size).astype("uint8")):
            #print(image[i*size:(i+1)*size, j*size:(j+1)*size])
            image_patch = image[i * size:(i + 1) * size, j * size:(j + 1) * size]
            gt_patch = gt[i * size:(i + 1) * size, j * size:(j + 1) * size]

            image_patches.append(image_patch)
            gt_patches.append(gt_patch)
            print(image_patch)
            print("----")


    return image_patches, gt_patches

# Fix edges
# Put into create_data, not necessary to make list then.
def patches_overlap(image, gt, size): #size = patch size
    image_patches = []
    gt_patches = []

    row = image.shape[0]
    col = image.shape[1]

    r = np.floor(row / size).astype("uint8")*2
    c = np.floor(col / size).astype("uint8")*2
    print("r", r)
    print("c", c)
    step = np.floor(size/2).astype("uint8")

    # patches without overlap, make them with overlap also (50 percent?)
    for i in range(r):
        for j in range(c):
            #print(image[i*size:(i+1)*size, j*size:(j+1)*size])
            image_patch = image[i * step:(i * step) + size, j * step:(j * step) + size]
            gt_patch = gt[i * step:(i * step) + size, j * step:(j * step) + size]

            image_patches.append(image_patch)
            gt_patches.append(gt_patch)
            print(image_patch)
            print("----")


    return image_patches, gt_patches

test = np.zeros((8,8))
test[1,2] = 4
test[2,3] = 2
test[1,1] = 9
test[7,7] = 3
print(test)
patches_overlap(test, test, 4)