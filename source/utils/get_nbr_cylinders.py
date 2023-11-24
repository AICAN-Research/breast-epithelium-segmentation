"""
Script to get nbr of TMA cylinders in a dataset
One needs to look at both train and test and all three class sets (invasive, benign, in situ) to find
the highest tma-nbr for each slide. Zero is included, thus 1 added
"""
import os
import numpy as np
import pandas as pd


def get_nbr_slides(path_):
    wsi_ids = []
    for path in os.listdir(path_):
        splits = path.split("_")
        wsi_idx = splits[1]
        if wsi_idx not in wsi_ids:
            wsi_ids.append(wsi_idx)
    return wsi_ids


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    path = '/path/to/dataset/'
    tma_ids = []
    wsi_ids = get_nbr_slides(path)
    df = pd.DataFrame(np.array(np.zeros((1, len(wsi_ids)), dtype=int)), columns=wsi_ids)

    for path in os.listdir(path):
        splits = path.split("_")
        wsi_idx = splits[1]
        tma_idx = splits[2]
        highest = df[wsi_idx]
        if highest[0] <= int(tma_idx):
            df[wsi_idx] = int(tma_idx) + 1  # one added as zero is included
    print(df)