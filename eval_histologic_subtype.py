"""
Script for evaluating segmentations (invasive, benign, in situ) on case level and for each histologic subtype
and histologic grade
"""
import os
import numpy as np


def eval_histologic_subtype():


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # stata path
    stata_path = ''

    # dataset path
    dataset_path = ''

    # model path
    model_path = ''

    dataset_path_ = os.listdir(dataset_path)
    paths_ = np.array([dataset_path + x for x in dataset_path_]).astype("U400")
    for file in os.listdir(paths_):


