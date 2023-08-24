""""
Script for reading Dice scores from .csv files and calculating mean and std again (with ddof=1 for np.std())
"""
import pandas as pd
import numpy as np

df = pd.read_csv('/filepath/results.csv')
print(df)
df_filtered = df.iloc[0, :]  # find correct metric and class to evaluate
print(df_filtered)
df_filtered = np.asarray(df_filtered)
df_filtered = df_filtered[1:]  # remove column name from ndarray

print(np.mean(df_filtered))
print(np.std(df_filtered))
print(np.std(df_filtered, ddof=1))
