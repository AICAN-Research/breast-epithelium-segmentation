"""
Script to create patches from whole slide images
"""
from datetime import datetime, date
import h5py
import numpy as np
import fast

def create_dataset(he_path, ck_path, annot_path):
    importer_he = fast.WholeSlideImageImporter.create(
        he_path)  # path to CK image
    importer_ck = fast.WholeSlideImageImporter.create(
        ck_path)  # path to CK image
    importer_annot = fast.TIFFImagePyramidImporter.create(
        annot_path)  # path to annotated image or areas to avoid

    annot = importer_annot.runAndGetOutputData()




if __name__ == "__main__":
    level = 2
    patch_size = 1024
    downsample_factor = 4

    data_split_path = ""  # split train/val/test
    he_ck_path = ""  # path to he and ck slides
    annot_path = ""  # area to not extract patches from

    curr_date = "".join(date.today().strftime("%d/%m").split("/")) + date.today().strftime("%Y")[2:]
    curr_time = "".join(str(datetime.now()).split(" ")[1].split(".")[0].split(":"))
    dataset_path = "./datasets/" + curr_date + "_" + curr_time + \
                   "_level_" + str(level) + \
                   "_psize_" + str(patch_size) + \
                   "_ds_" + str(downsample_factor) + "/"

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