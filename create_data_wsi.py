"""
Script to create patches from whole slide images
Annotations from QuPath of which areas to avoid (from he image)
Put patches in train/val/test set
Which wsis into which set is determined previously
"""
from datetime import datetime, date
import h5py
import numpy as np
import fast
import cv2
from skimage.registration import phase_cross_correlation
from scipy import ndimage as ndi
from skimage.exposure import equalize_hist


def create_dataset(he_path, ck_path, annot_path, dataset_path, level, patch_size, ds_factor):
    importer_he = fast.WholeSlideImageImporter.create(
        he_path)  # path to CK image
    importer_ck = fast.WholeSlideImageImporter.create(
        ck_path)  # path to CK image
    importer_annot = fast.TIFFImagePyramidImporter.create(
        annot_path)  # path to annotated image or areas to keep

    extractor_he = fast.ImagePyramidLevelExtractor.create(level=level).connect(importer_he)
    extractor_ck = fast.ImagePyramidLevelExtractor.create(level=level).connect(importer_ck)

    he_image = extractor_he.runAndGetOutputData()
    ck_image = extractor_ck.runAndGetOutputData()

    he_image = np.asarray(he_image)
    ck_image = np.asarray(ck_image)

    # get shape of he and ck images
    he_height, he_width, _ = he_image.shape
    ck_height, ck_width, _ = ck_image.shape

    longest_height = max([he_height, ck_height])
    longest_width = max([he_width, ck_width])

    # pad smallest image
    ck_image_padded = np.ones((longest_height, longest_width, 3), dtype="uint8") * 255
    he_image_padded = np.ones((longest_height, longest_width, 3), dtype="uint8") * 255

    ck_image_padded[:ck_image.shape[0], :ck_image.shape[1]] = ck_image
    he_image_padded[:he_image.shape[0], :he_image.shape[1]] = he_image

    # downsample before registration
    curr_shape = ck_image_padded.shape[:2]
    ck_image_padded_ds = cv2.resize(ck_image_padded, np.round(np.array(curr_shape) / ds_factor).astype("int32"),
                                    interpolation=cv2.INTER_NEAREST)
    he_image_padded_ds = cv2.resize(he_image_padded, np.round(np.array(curr_shape) / ds_factor).astype("int32"),
                                    interpolation=cv2.INTER_NEAREST)

    # detect shift between ck and he, histogram equalization for better shift in image with few
    # distinct landmarks
    ck_image_padded_ds_histeq = equalize_hist(ck_image_padded_ds)
    shifts, reg_error, phase_diff = phase_cross_correlation(he_image_padded_ds, ck_image_padded_ds_histeq,
                                                            return_error=True)

    shifts[2] = 0  # set z-axis to zero (should be from beginning)

    # scale shifts back and apply to original resolution
    shifts = (np.round(ds_factor * shifts)).astype("int32")
    ck_image_padded_shifted = ndi.shift(ck_image_padded, shifts, order=0, mode="constant", cval=255, prefilter=False)


    # create patches

    # if patch includes areas in annotated image -> skip


if __name__ == "__main__":
    level = 2
    patch_size = 1024
    ds_factor = 4

    data_split_path = ""  # split train/val/test
    he_ck_path = ""  # path to he and ck slides
    annot_path = ""  # area to extract patches from, from he image to avoid shifting

    curr_date = "".join(date.today().strftime("%d/%m").split("/")) + date.today().strftime("%Y")[2:]
    curr_time = "".join(str(datetime.now()).split(" ")[1].split(".")[0].split(":"))
    dataset_path = "./datasets/" + curr_date + "_" + curr_time + \
                   "_level_" + str(level) + \
                   "_psize_" + str(patch_size) + \
                   "_ds_" + str(ds_factor) + "/"

    # define datasets (train/val/test) - always uses predefined dataset
    with h5py.File(data_split_path, "r") as f:
        train_set = np.array(f['train']).astype(str)
        val_set = np.array(f['val']).astype(str)
        test_set = np.array(f['test']).astype(str)

    # get elements in each dataset
    N_train = len(list(train_set))  # should be one wsi?
    N_val = len(list(val_set))  # should be one wsi?

    file_set = train_set, val_set
    set_names = ['ds_train', 'ds_val']

    print("file set", file_set)
    print("length file set", len(file_set))

    # go through files in train/val/test -> create_dataset()