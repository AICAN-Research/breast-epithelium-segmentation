"""
Plot training/validation loss
Remember to change history path
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


history_path = './output/history/history_260423_093216_agunet_bs_8.csv'  # path to history file

data = pd.read_csv(history_path)

print(data.shape)

net = "agunet"
nbr = 54
epochs = data.shape[0]

if net == "agunet":
    loss = data['conv2d_' + str(nbr) +'_loss']
    val_loss = data['val_conv2d_' + str(nbr) + '_loss']

    benign = data['conv2d_' + str(nbr) +'_benign']
    inSitu = data['conv2d_' + str(nbr) +'_insitu']
    invasive = data['conv2d_' + str(nbr) +'_invasive']

    val_benign = data['val_conv2d_' + str(nbr) +'_benign']
    val_inSitu = data['val_conv2d_' + str(nbr) +'_insitu']
    val_invasive = data['val_conv2d_' + str(nbr) +'_invasive']

    # train loss
    epochs = range(1, epochs + 1)
    plt.plot(epochs, benign, '-', label='Train benign loss')
    plt.plot(epochs, inSitu, '-', label='Train insitu loss')
    plt.plot(epochs, invasive, '-', label='Train invasive loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # validation loss
    plt.plot(epochs, val_benign, '-', label='Val benign loss')
    plt.plot(epochs, val_inSitu, '-', label='Val insitu loss')
    plt.plot(epochs, val_invasive, '-', label='Val invasive loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # train and validation loss
    plt.plot(epochs, loss, '-', label='Training loss')
    plt.plot(epochs, val_loss, '-', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

elif net == "unet":

    loss = data['loss']  # [0:50]
    val_loss = data['val_loss']  # [0:50]

    benign = data['benign']  # [0:50]
    inSitu = data['insitu']  # [0:50]
    invasive = data['invasive']  # [0:50]

    val_benign = data['val_benign']  # [0:50]
    val_inSitu = data['val_insitu']  # [0:50]
    val_invasive = data['val_invasive']  # [0:50]

    # train loss
    epochs = range(1, epochs + 1)
    plt.plot(epochs, benign, '-', label='Train benign loss')
    plt.plot(epochs, inSitu, '-', label='Train insitu loss')
    plt.plot(epochs, invasive, '-', label='Train invasive loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # validation loss
    plt.plot(epochs, val_benign, '-', label='Val benign loss')
    plt.plot(epochs, val_inSitu, '-', label='Val insitu loss')
    plt.plot(epochs, val_invasive, '-', label='Val invasive loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # train and validation loss
    plt.plot(epochs, loss, '-', label='Training loss')
    plt.plot(epochs, val_loss, '-', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# get the lowest train, validation loss
print(np.amin(loss))
print(np.amin(val_loss))

