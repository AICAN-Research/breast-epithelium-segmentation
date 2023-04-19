"""
Script to divide whole slide images that are divided into eight squares into a training and a validation set.
6 squares are placed in the training set and 2 in the validation set.
Should only be run once
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, date
import h5py

curr_date = "".join(date.today().strftime("%d/%m").split("/")) + date.today().strftime("%Y")[2:]
curr_time = "".join(str(datetime.now()).split(" ")[1].split(".")[0].split(":"))
wsi_split_path = './wsi_splits/' + curr_date + "_" + curr_time + "/"

N_train = 6
N = np.arange(8)  # array from 0 (included) to 7 (included)
np.random.shuffle(N)  # shuffles numbers from 0 to 7

# numbers in train and validation set
train_set = N[:N_train]
val_set = N[N_train:]

os.makedirs(wsi_split_path, exist_ok=True)

# convert to hdf5
with h5py.File((wsi_split_path + "dataset_split.h5"), 'w') as f:
    f.create_dataset("val", data=np.array(val_set).astype('S400'), compression="gzip", compression_opts=4)
    f.create_dataset("train", data=np.array(train_set).astype('S400'), compression="gzip", compression_opts=4)