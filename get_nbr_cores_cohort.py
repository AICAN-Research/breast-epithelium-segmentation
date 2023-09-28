"""
Script for calculating the number of tma cylidners in each cohort
"""

import os
import pandas as pd
import numpy as np

level = 1  # level 1 = TMA-cylinder level, level 2 = patches

if level == 1:
    ##
    stata_path = '/path-to-stata-dataset/'
    all_paths = '/path-to-dataset/'
    df = pd.read_stata(stata_path, convert_categoricals=False)

    nbr_cylinders_H0 = 0
    nbr_cylinders_H2 = 0
    nbr_cylinders_ECD = 0
    nbr_cylinders_HUS = 0

    # find number of wsis in dataset
    nbr_wsis = 0
    for path in os.listdir(all_paths):
        wsi_nbr = path.split('_')[1]
        wsi_nbr = int(wsi_nbr)
        if wsi_nbr > nbr_wsis:
            nbr_wsis = wsi_nbr

    nbr_wsis += 1  # since one wsi has wsi_idx 0
    tma_idxs = np.zeros(nbr_wsis).astype("uint8")  # list to include largest nbr of cylidners in each wsi
    cohort_idxs = np.zeros(nbr_wsis).astype("uint8")  # list of which cohort each wsi corresponds to

    for path in os.listdir(all_paths):
        cohort = path.split('_')[3]
        tma_idx = path.split('_')[2]
        wsi_idx = path.split('_')[1]

        tma_idx = int(tma_idx)
        wsi_idx = int(wsi_idx)

        if tma_idx > tma_idxs[wsi_idx]:
            tma_idxs[wsi_idx] = tma_idx

        if cohort == 'HUNT0':
            cohort_idxs[wsi_idx] = 1
        elif cohort == 'HUNT2':
            cohort_idxs[wsi_idx] = 2
        elif cohort == 'ECD':
            cohort_idxs[wsi_idx] = 3
        elif cohort == 'HUS':
            cohort_idxs[wsi_idx] = 4

    print(tma_idxs)  # need to add 1 to each as 0 is also a tma_idx
    print()
    print(cohort_idxs)


if level == 2:
    ds_ = 'ds_val'  # ds_train or ds_val
    path = '/path-to-dataset/' + ds_ + '/'

    ##
    stata_path = '/path-to-stata-dataset'
    df = pd.read_stata(stata_path, convert_categoricals=False)

    set = os.listdir(path)  # benign, in situ or invasive

    nbr_cylinders_H0 = 0
    nbr_cylinders_H2 = 0
    nbr_cylinders_ECD = 0
    nbr_cylinders_HUS = 0

    all_paths = []
    for set_ in set:
        cylinders_paths = path + str(set_) + '/'
        cylinders_paths = os.listdir(cylinders_paths)
        for path_ in cylinders_paths:
            all_paths.append(path_)

    # find number of wsis in dataset
    nbr_wsis = 0
    for path in all_paths:
        wsi_nbr = path.split('_')[1]
        wsi_nbr = int(wsi_nbr)
        if wsi_nbr > nbr_wsis:
            nbr_wsis = wsi_nbr

    nbr_wsis += 1  # since one wsi has wsi_idx 0
    tma_idxs = np.zeros(nbr_wsis).astype("uint8")  # list to include largest nbr of cylidners in each wsi
    cohort_idxs = np.zeros(nbr_wsis).astype("uint8")  # list of which cohort each wsi corresponds to

    for path in all_paths:
        split = path.split('.')[0]
        triplet_nbr = split.split('_')[7]
        id_ = split.split('_')[6]
        cohort = split.split('_')[4]
        tma_idx = path.split('_')[2]
        wsi_idx = path.split('_')[1]

        tma_idx = int(tma_idx)
        wsi_idx = int(wsi_idx)

        if tma_idx > tma_idxs[wsi_idx]:
            tma_idxs[wsi_idx] = tma_idx

        if cohort == 'HUNT0':
            cohort_idxs[wsi_idx] = 1
        elif cohort == 'HUNT2':
            cohort_idxs[wsi_idx] = 2
        elif cohort == 'ECD':
            cohort_idxs[wsi_idx] = 3
        elif cohort == 'HUS':
            cohort_idxs[wsi_idx] = 4

    print(tma_idxs)  # need to add 1 to each as 0 is also a tma_idx
    print()
    print(cohort_idxs)
