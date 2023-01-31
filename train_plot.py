"""
Plot training/validation loss
Remember to change history path
"""

import pandas as pd
import matplotlib.pyplot as plt

epochs = 360  # match number of epochs in train script

history_path = './output/history/history_091222_090900_unet_bs_16.csv'  # path to history file

data = pd.read_csv(history_path)

print(data.shape)

loss = data['loss']
val_loss = data['val_loss']

epochs = range(1, epochs + 1)
plt.plot(epochs, loss, '-', label='Training loss')
plt.plot(epochs, val_loss, '-', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
