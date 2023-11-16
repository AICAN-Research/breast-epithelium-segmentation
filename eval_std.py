""""
Script for reading Dice scores from .csv files and calculating mean and std again (with ddof=1 for np.std())
"""
import pandas as pd
import numpy as np
path = '/path/to/file/.csv'
df = pd.read_csv(path)
df_filtered = df.iloc[11, :]  # find correct metric and class to evaluate (ex 11 which corresponds to ex in situ pos target)
df_filtered = np.asarray(df_filtered)
df_filtered = df_filtered[1:]  # remove column name from ndarray
df_new = []
for elem in df_filtered:
    if elem >= 0:
        df_new.append(elem)
df_new = np.asarray(df_new)
print(np.std(df_new, ddof=1))
