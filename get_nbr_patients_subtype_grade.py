"""
Script for extracting the number of patients from a dataset divided into subtype and grade
"""
import os
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt

level = 2  # level 1 = TMA-cylinder level, level 2 = patches

if level == 1:
    d_set = 3  # 1, 2 or 3

    ##
    if d_set == 1:  # test set 1
        path = '/path-to-dataset/'
        stata_path = '/path-to-stata-dataset/'
    elif d_set == 2:  # test set 2
        path = '/path-to-dataset/'
        stata_path = '/path-to-stata-dataset/'
    elif d_set == 3:  # validation set at level 1
        path = '/path-to-dataset/'
        stata_path = '/path-to-stata-dataset/'
    df = pd.read_stata(stata_path, convert_categoricals=False)

    cylinders_paths = os.listdir(path)

    nbr_patients = 0
    checked_paths = []
    id_participants_1 = []
    id_participants_2 = []
    for path in cylinders_paths:
        split = path.split('.')[0]
        if d_set == 1 or d_set == 3:
            triplet_nbr = split.split('_')[5]
            id_ = split.split('_')[4]
        else:
            triplet_nbr = split.split('_')[7]
            id_ = split.split('_')[6]
        cohort = split.split('_')[3]

        N = len(checked_paths)
        include = True
        for checked_path in checked_paths:
            if d_set == 1 or d_set == 3:
                checked_path_id = checked_path.split('_')[4]
                checked_path_triplet = checked_path.split('_')[5]
            else:
                checked_path_id = checked_path.split('_')[6]
                checked_path_triplet = checked_path.split('_')[7]
            checked_path_triplet = checked_path_triplet.split('.')[0]
            # skip cylinders where patient is already included due to same triplet number
            if cohort in checked_path and id_ == checked_path_id and triplet_nbr == checked_path_triplet:
                include = False
                continue
        if include:
            checked_paths.append(path)

    # find nbr of subtype and grade on patient level
    grade_1_H0 = 0
    grade_1_H2 = 0
    grade_1_ECD = 0
    grade_1_HUS = 0
    grade_1_HPA = 0

    grade_2_H0 = 0
    grade_2_H2 = 0
    grade_2_ECD = 0
    grade_2_HUS = 0
    grade_2_HPA = 0

    grade_3_H0 = 0
    grade_3_H2 = 0
    grade_3_ECD = 0
    grade_3_HUS = 0
    grade_3_HPA = 0

    subtype_1_H0 = 0
    subtype_1_H2 = 0
    subtype_1_ECD = 0
    subtype_1_HUS = 0
    subtype_1_HPA = 0

    subtype_2_H0 = 0
    subtype_2_H2 = 0
    subtype_2_ECD = 0
    subtype_2_HUS = 0
    subtype_2_HPA = 0

    subtype_8_H0 = 0
    subtype_8_H2 = 0
    subtype_8_ECD = 0
    subtype_8_HUS = 0
    subtype_8_HPA = 0

    for file in checked_paths:
        file_front = file.split(".")[0]
        splits = file_front.split("_")
        if d_set == 1 or d_set == 3:
            cohort = splits[3]
            id_ = splits[4]
            case = splits[5]
        else:
            id_ = splits[4][1:]
            case = splits[7]

        # get matching case in stata file as TMA cylinder
        if d_set == 1 or d_set == 3:
            filtered_data = df.loc[
                (df["Maren_P1"] == 1)
                & (df[str(cohort)] == 1)
                & (df["slide"] == int(id_))
                & (df["case"] == int(case))
                ]
        else:
            filtered_data = df.loc[
                (df["slide_" + str(id_)] == 1)
                & (df["case_" + str(id_)] == int(case))
                ]

        # needs to skip cylinders without information (excluded in STATA-file):
        if len(filtered_data) == 0:
            print("not included in grade/subtype, included in images: ", file)
            continue

        # some patients have more than one case, those have to be skipped:
        if d_set == 1 or d_set == 3:
            id_deltaker = filtered_data["ID_deltaker"]
            if cohort == "HUS":
                id_deltaker = int(id_deltaker)
                if id_deltaker in id_participants_1:
                    print("paticipant already included with different triplet nbr: ", id_deltaker)
                    print(file)
                    continue
                else:
                    id_participants_1.append(id_deltaker)
        else:
            id_deltaker = filtered_data["ID_deltaker"]
            id_deltaker = int(id_deltaker)
            if id_deltaker in id_participants_2:
                print("paticipant already included with different triplet nbr: ", id_deltaker)
                print(file)
                continue
            else:
                id_participants_2.append(id_deltaker)

        # get histological subtype and grade for case
        type_ = int(filtered_data["type_six"])  # 1, 2, 3, 4, 5, 8, still need eight in dice_types
        if 8 > type_ > 2:  # combine type 3, 4, 5 into type 8
            type_ = 8
        grade_ = int(filtered_data["GRAD"])

        # skip grade of types which are not 1-3 (grade) or 1, 2, 8 (type)
        if grade_ < 1 or grade_ > 3 or type_ < 1 or type_ > 8:
            continue

        if cohort == 'HUNT0':
            if type_ == 1:
                subtype_1_H0 += 1
            elif type_ == 2:
                subtype_2_H0 += 1
            elif type_ == 8:
                subtype_8_H0 += 1
            if grade_ == 1:
                grade_1_H0 += 1
            elif grade_ == 2:
                grade_2_H0 += 1
            elif grade_ == 3:
                grade_3_H0 += 1
        elif cohort == 'HUNT2':
            if type_ == 1:
                subtype_1_H2 += 1
            elif type_ == 2:
                subtype_2_H2 += 1
            elif type_ == 8:
                subtype_8_H2 += 1
            if grade_ == 1:
                grade_1_H2 += 1
            elif grade_ == 2:
                grade_2_H2 += 1
            elif grade_ == 3:
                grade_3_H2 += 1
        elif cohort == 'ECD':
            if type_ == 1:
                subtype_1_ECD += 1
            elif type_ == 2:
                subtype_2_ECD += 1
            elif type_ == 8:
                subtype_8_ECD += 1
            if grade_ == 1:
                grade_1_ECD += 1
            elif grade_ == 2:
                grade_2_ECD += 1
            elif grade_ == 3:
                grade_3_ECD += 1
        elif cohort == 'HUS':
            if type_ == 1:
                subtype_1_HUS += 1
            elif type_ == 2:
                subtype_2_HUS += 1
            elif type_ == 8:
                subtype_8_HUS += 1
            if grade_ == 1:
                grade_1_HUS += 1
            elif grade_ == 2:
                grade_2_HUS += 1
            elif grade_ == 3:
                grade_3_HUS += 1
        elif cohort == 'HPA':
            if type_ == 1:
                subtype_1_HPA += 1
            elif type_ == 2:
                subtype_2_HPA += 1
            elif type_ == 8:
                subtype_8_HPA += 1
            if grade_ == 1:
                grade_1_HPA += 1
            elif grade_ == 2:
                grade_2_HPA += 1
            elif grade_ == 3:
                grade_3_HPA += 1

    print('H0')
    print("type 1: ", subtype_1_H0)
    print("type 2: ", subtype_2_H0)
    print("type 8: ", subtype_8_H0)
    print("grade 1: ", grade_1_H0)
    print("grade 2: ", grade_2_H0)
    print("grade 3: ", grade_3_H0)
    print()
    print('H2')
    print("type 1: ", subtype_1_H2)
    print("type 2: ", subtype_2_H2)
    print("type 8: ", subtype_8_H2)
    print("grade 1: ", grade_1_H2)
    print("grade 2: ", grade_2_H2)
    print("grade 3: ", grade_3_H2)
    print()
    print('ECD')
    print("type 1: ", subtype_1_ECD)
    print("type 2: ", subtype_2_ECD)
    print("type 8: ", subtype_8_ECD)
    print("grade 1: ", grade_1_ECD)
    print("grade 2: ", grade_2_ECD)
    print("grade 3: ", grade_3_ECD)
    print()
    print('HUS')
    print("type 1: ", subtype_1_HUS)
    print("type 2: ", subtype_2_HUS)
    print("type 8: ", subtype_8_HUS)
    print("grade 1: ", grade_1_HUS)
    print("grade 2: ", grade_2_HUS)
    print("grade 3: ", grade_3_HUS)
    print()
    print('HPA')
    print("type 1: ", subtype_1_HPA)
    print("type 2: ", subtype_2_HPA)
    print("type 8: ", subtype_8_HPA)
    print("grade 1: ", grade_1_HPA)
    print("grade 2: ", grade_2_HPA)
    print("grade 3: ", grade_3_HPA)

if level == 2:
    ds_ = 'ds_val'  # ds_train or ds_val
    path = 'path-to-dataset' + ds_ + '/'

    ##
    stata_path = 'path-to-stata-dataset'
    df = pd.read_stata(stata_path, convert_categoricals=False)

    set = os.listdir(path)  # benign, in situ or invasive

    nbr_patients = 0
    checked_paths = []
    id_participants_1 = []

    all_paths = []
    for set_ in set:
        cylinders_paths = path + str(set_) + '/'
        cylinders_paths = os.listdir(cylinders_paths)
        for path_ in cylinders_paths:
            all_paths.append(path_)

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
            # skip cylinders where patient is already included due to same triplet number
            if cohort in checked_path and id_ == checked_path_id and triplet_nbr == checked_path_triplet:
                include = False
                continue
        if include:
            checked_paths.append(path)

    # find nbr of subtype and grade on patient level
    grade_1_H0 = 0
    grade_1_H2 = 0
    grade_1_ECD = 0
    grade_1_HUS = 0

    grade_2_H0 = 0
    grade_2_H2 = 0
    grade_2_ECD = 0
    grade_2_HUS = 0

    grade_3_H0 = 0
    grade_3_H2 = 0
    grade_3_ECD = 0
    grade_3_HUS = 0

    subtype_1_H0 = 0
    subtype_1_H2 = 0
    subtype_1_ECD = 0
    subtype_1_HUS = 0

    subtype_2_H0 = 0
    subtype_2_H2 = 0
    subtype_2_ECD = 0
    subtype_2_HUS = 0

    subtype_8_H0 = 0
    subtype_8_H2 = 0
    subtype_8_ECD = 0
    subtype_8_HUS = 0

    for file in checked_paths:
        file_front = file.split(".")[0]
        splits = file_front.split("_")
        cohort = splits[4]
        id_ = splits[6]
        case = splits[7]

        # get matching case in stata file as TMA cylinder
        filtered_data = df.loc[
            (df["Maren_P1"] == 1)
            & (df[str(cohort)] == 1)
            & (df["slide"] == int(id_))
            & (df["case"] == int(case))
            ]

        # needs to skip cylinders without information (excluded in STATA-file):
        if len(filtered_data) == 0:
            print("not included in grade/subtype, included in images: ", file)
            continue

        # skip participatns with multiple triplets
        id_deltaker = filtered_data["ID_deltaker"]
        if cohort == "HUS":
            id_deltaker = int(id_deltaker)
            if id_deltaker in id_participants_1:
                print("paticipant already included with different triplet nbr: ", id_deltaker)
                print(file)
                continue
            else:
                id_participants_1.append(id_deltaker)

        # get histological subtype and grade for case
        type_ = int(filtered_data["type_six"])  # 1, 2, 3, 4, 5, 8, still need eight in dice_types
        if 8 > type_ > 2:  # combine type 3, 4, 5 into type 8
            type_ = 8
        grade_ = int(filtered_data["GRAD"])

        # skip grade of types which are not 1-3 (grade) or 1, 2, 8 (type)
        if grade_ < 1 or grade_ > 3 or type_ < 1 or type_ > 8:
            continue

        if cohort == 'HUNT0':
            if type_ == 1:
                subtype_1_H0 += 1
            elif type_ == 2:
                subtype_2_H0 += 1
            elif type_ == 8:
                subtype_8_H0 += 1
            if grade_ == 1:
                grade_1_H0 += 1
            elif grade_ == 2:
                grade_2_H0 += 1
            elif grade_ == 3:
                grade_3_H0 += 1
        elif cohort == 'HUNT2':
            if type_ == 1:
                subtype_1_H2 += 1
            elif type_ == 2:
                subtype_2_H2 += 1
            elif type_ == 8:
                subtype_8_H2 += 1
            if grade_ == 1:
                grade_1_H2 += 1
            elif grade_ == 2:
                grade_2_H2 += 1
            elif grade_ == 3:
                grade_3_H2 += 1
        elif cohort == 'ECD':
            if type_ == 1:
                subtype_1_ECD += 1
            elif type_ == 2:
                subtype_2_ECD += 1
            elif type_ == 8:
                subtype_8_ECD += 1
            if grade_ == 1:
                grade_1_ECD += 1
            elif grade_ == 2:
                grade_2_ECD += 1
            elif grade_ == 3:
                grade_3_ECD += 1
        elif cohort == 'HUS':
            if type_ == 1:
                subtype_1_HUS += 1
            elif type_ == 2:
                subtype_2_HUS += 1
            elif type_ == 8:
                subtype_8_HUS += 1
            if grade_ == 1:
                grade_1_HUS += 1
            elif grade_ == 2:
                grade_2_HUS += 1
            elif grade_ == 3:
                grade_3_HUS += 1

    print('H0')
    print("type 1: ", subtype_1_H0)
    print("type 2: ", subtype_2_H0)
    print("type 8: ", subtype_8_H0)
    print("grade 1: ", grade_1_H0)
    print("grade 2: ", grade_2_H0)
    print("grade 3: ", grade_3_H0)
    print()
    print('H2')
    print("type 1: ", subtype_1_H2)
    print("type 2: ", subtype_2_H2)
    print("type 8: ", subtype_8_H2)
    print("grade 1: ", grade_1_H2)
    print("grade 2: ", grade_2_H2)
    print("grade 3: ", grade_3_H2)
    print()
    print('ECD')
    print("type 1: ", subtype_1_ECD)
    print("type 2: ", subtype_2_ECD)
    print("type 8: ", subtype_8_ECD)
    print("grade 1: ", grade_1_ECD)
    print("grade 2: ", grade_2_ECD)
    print("grade 3: ", grade_3_ECD)
    print()
    print('HUS')
    print("type 1: ", subtype_1_HUS)
    print("type 2: ", subtype_2_HUS)
    print("type 8: ", subtype_8_HUS)
    print("grade 1: ", grade_1_HUS)
    print("grade 2: ", grade_2_HUS)
    print("grade 3: ", grade_3_HUS)

