"""
Script for finding nbr of patients in a TMA-dataset (cylinders at image plane 1 or patches at image plane 2)
Will not work when multiple triplets/dublets from same patient
The files must be saved according to "create_data_tma.py" for patches from TMAs and to "create_tma_pairs.py" for whole
TMA cylinders
"""
import os

level = 2  # level 1 = TMA-cylinder level, level 2 = patches

if level == 1:
    ds_set = 1  # 1 = test set 1, 2 = test set 2
    path = '/path/to/dataset'

    cylinders_paths = os.listdir(path)

    nbr_patients = 0
    checked_paths = []

    for path in cylinders_paths:
        if ds_set == 1:
            split = path.split('.')[0]
            triplet_nbr = split.split('_')[5]
            id_ = split.split('_')[4]
            cohort = split.split('_')[3]
        else:
            split = path.split('.')[0]
            triplet_nbr = split.split('_')[7]
            id_ = split.split('_')[4]
            cohort = split.split('_')[3]

        N = len(checked_paths)
        include = True
        for checked_path in checked_paths:
            checked_path_id = checked_path.split('_')[4]
            checked_path_triplet = checked_path.split('_')[5]
            checked_path_triplet = checked_path_triplet.split('.')[0]
            if cohort in checked_path and id_ == checked_path_id and triplet_nbr == checked_path_triplet:
                include = False
                continue
        if include and cohort == 'HPA':  # if include and cohort == 'cohort name' if want to check for specific cohort
            checked_paths.append(path)
    print("nbr of patients in dataset is: ", len(checked_paths))

# if patches:
if level == 2:
    path = '/path/to/dataset'

    set = os.listdir(path)  # benign, in situ or invasive

    nbr_patients = 0
    checked_paths = []

    all_paths = []
    for set_ in set:
        cylinders_paths = path + str(set_) + '/'
        cylinders_paths = os.listdir(cylinders_paths)
        for path_ in cylinders_paths:
            all_paths.append(path_)

    print(len(all_paths))

    for path in all_paths:
        split = path.split('.')[0]
        triplet_nbr = split.split('_')[7]
        id_ = split.split('_')[6]
        cohort = split.split('_')[4]

        N = len(checked_paths)
        include = True
        for checked_path in checked_paths:
            checked_path_id = checked_path.split('_')[6]
            checked_path_triplet = checked_path.split('_')[7]
            checked_path_triplet = checked_path_triplet.split('.')[0]
            if cohort in checked_path and id_ == checked_path_id and triplet_nbr == checked_path_triplet:
                include = False
                continue
        if include and cohort == 'HUS':  # if include and cohort == 'cohort name' if want to check for specific cohort:
            checked_paths.append(path)

    print("nbr of patients in dataset is: ", len(checked_paths))
