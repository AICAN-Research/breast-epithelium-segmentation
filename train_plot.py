"""
Plot training/validation loss
Remember to change history path
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


history_path = './output/history/history_210223_152958_unet_bs_32.csv'  # path to history file

data = pd.read_csv(history_path)

print(data.shape)

epochs = data.shape[0]

loss = data['loss']
val_loss = data['val_loss']

benign = data['benign']
inSitu = data['insitu']
invasive = data['invasive']

val_benign = data['val_benign']
val_inSitu = data['val_insitu']
val_invasive = data['val_invasive']

epochs = range(1, epochs + 1)
plt.plot(epochs, benign, '-', label='Train benign loss')
plt.plot(epochs, inSitu, '-', label='Train insitu loss')
plt.plot(epochs, invasive, '-', label='Train invasive loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()  # Show the two images on top of each other

plt.plot(epochs, val_benign, '-', label='Val benign loss')
plt.plot(epochs, val_inSitu, '-', label='Val insitu loss')
plt.plot(epochs, val_invasive, '-', label='Val invasive loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()  # Show the two images on top of each other

plt.plot(epochs, loss, '-', label='Training loss')
plt.plot(epochs, val_loss, '-', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# get the lowest train, validation loss
print(np.amin(loss))
print(np.amin(val_loss))

