# Divide data into train/validation and test set. The test set should not be used during development of the model.
# This script should only be run to divide once.
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

N = nbr_files
N_train = int(N * 0.75)
N_val = int(np.floor(N * 0.15))
N_test = int(N - N_train - N_val)

# shuffle
order = np.arange(nbr_files)
np.random.shuffle(order)
shuffled_files = [files[x] for x in order]

# files in train, validation and test set
train_set = shuffled_files[:N_train]
val_set = shuffled_files[N_train:(N_train + N_val)]
test_set = shuffled_files[(N_train + N_val):]

# convert to DataFrame
labels = {'train': [train_set], 'val': [val_set], 'test': [test_set]}
df = pd.DataFrame(data=labels, columns=['train', 'val', 'test'])

# create folder if not exists
os.makedirs(data_split_path, exist_ok=True)

# save df as csv:
df.to_csv(data_split_path + 'data_split.csv')


