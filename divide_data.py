# Divide data into train/validation and test set. The test set should not be used during development of the model.
# This script should only be run to divide once.
# Use fraction of wsis to begin with, then these have to be correctly placed when using all
# Wait with cross-validation to later
from datetime import datetime, date
import os
import numpy as np
import pandas as pd


mask_path = '/data/Maren_P1/data/annotations_converted/blue_channel_tiff/'

curr_date = "".join(date.today().strftime("%d/%m").split("/")) + date.today().strftime("%Y")[2:]
curr_time = "".join(str(datetime.now()).split(" ")[1].split(".")[0].split(":"))
data_split_path = './data_splits/' + curr_date + "_" + curr_time + "/"

files = os.listdir(mask_path)
nbr_files = len(files)
train_set = []

N = nbr_files  # remaining files after manually put some in train set
N_train = 6
N_val = 2
N_test = 2

# ECD and HUNT2_2 needs to be placed in the training set
for file in files[:]:  # copy to avoid skipping element in list after removed elements
    if ("ECD" in file) or ("HUNT2_EFI_CK_BC_2" in file):
        train_set.append(file)
        files.remove(file)

N_train = N_train - len(train_set)

nbr_files = len(files)

# shuffle
order = np.arange(nbr_files)
np.random.shuffle(order)
shuffled_files = [files[x] for x in order]

# files in train, validation and test set
train_set = train_set + shuffled_files[:N_train]
val_set = shuffled_files[N_train:(N_train + N_val)]
test_set = shuffled_files[(N_train + N_val):]

# convert to DataFrame
labels = {'train': [train_set], 'val': [val_set], 'test': [test_set]}
df = pd.DataFrame(data=labels, columns=['train', 'val', 'test'])

# create folder if not exists
os.makedirs(data_split_path, exist_ok=True)
exit()
# save df as csv:
df.to_csv(data_split_path)


