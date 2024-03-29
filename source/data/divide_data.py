"""
Divide data into train/validation and test set. The test set should not be used during development of the model.
This script should only be run to divide once.
Use fraction of wsis to begin with, then these have to be correctly placed when using all
Wait with cross-validation to later
"""
from datetime import datetime, date
import os
import numpy as np
import h5py


mask_path = '/data/Maren_P1/data/annotations_converted/blue_channel_tiff/'

curr_date = "".join(date.today().strftime("%d/%m").split("/")) + date.today().strftime("%Y")[2:]
curr_time = "".join(str(datetime.now()).split(" ")[1].split(".")[0].split(":"))
data_split_path = './data_splits/' + curr_date + "_" + curr_time + "/"


files = np.array(os.listdir(mask_path))
nbr_files = len(files)

N = nbr_files  # remaining files after manually put some in train set
N_train = 16
N_val = 4

# shuffle
np.random.shuffle(files)

# files in train, validation and test set
train_set = files[:N_train]
val_set = files[N_train:(N_train + N_val)]
test_set = files[(N_train + N_val):]

# make folder if not exist
os.makedirs(data_split_path, exist_ok=True)

# convert to hdf5
with h5py.File((data_split_path + "dataset_split.h5"), 'w') as f:
    f.create_dataset("test", data=np.array(test_set).astype('S400'), compression="gzip", compression_opts=4)
    f.create_dataset("val", data=np.array(val_set).astype('S400'), compression="gzip", compression_opts=4)
    f.create_dataset("train", data=np.array(train_set).astype('S400'), compression="gzip", compression_opts=4)
