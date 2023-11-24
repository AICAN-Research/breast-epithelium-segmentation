"""
Plot training/validation loss
Remember to change history path
Old script
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

history_path = './csv'
history_path_2 = './.csv'

data = pd.read_csv(history_path)
data_2 = pd.read_csv(history_path_2)

print(data.shape)

net = "agunet"
nbr = 54
epochs = data.shape[0]
epochs2 = data_2.shape[0]

if net == "agunet":
    loss = data['conv2d_' + str(nbr) +'_loss']
    loss_2 = data_2['conv2d_' + str(nbr) + '_loss']
    val_loss = data['val_conv2d_' + str(nbr) + '_loss']
    val_loss_2 = data_2['val_conv2d_' + str(nbr) + '_loss']


    benign = data['conv2d_' + str(nbr) +'_benign']
    inSitu = data['conv2d_' + str(nbr) +'_insitu']
    invasive = data['conv2d_' + str(nbr) +'_invasive']

    benign_2 = data_2['conv2d_' + str(nbr) + '_benign']
    inSitu_2 = data_2['conv2d_' + str(nbr) + '_insitu']
    invasive_2 = data_2['conv2d_' + str(nbr) + '_invasive']

    val_benign = data['val_conv2d_' + str(nbr) +'_benign']
    val_inSitu = data['val_conv2d_' + str(nbr) +'_insitu']
    val_invasive = data['val_conv2d_' + str(nbr) +'_invasive']

    val_benign_2 = data_2['val_conv2d_' + str(nbr) + '_benign']
    val_inSitu_2 = data_2['val_conv2d_' + str(nbr) + '_insitu']
    val_invasive_2 = data_2['val_conv2d_' + str(nbr) + '_invasive']

    # train loss
    epochs_1 = range(1, epochs + 1)
    epochs_2 = range(1, epochs2 + 1)

    #plt.plot(epochs, benign, '-', label='Train benign loss')
    plt.plot(epochs_1, inSitu, '-', label='Train insitu loss aug + drop')
    plt.plot(epochs_1, val_inSitu, '-', label='Val insitu loss aug + drop')
    plt.plot(epochs_2, inSitu_2, '-', label='Train insitu loss aug')
    plt.plot(epochs_2, val_inSitu_2, '-', label='Val insitu loss aug')
    #plt.plot(epochs_1, invasive, '-', label='Train invasive loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # validation loss
    plt.plot(epochs_1, benign, '-', label='Train benign loss aug + drop')
    plt.plot(epochs_1, val_benign, '-', label='Val benign loss aug + drop')
    plt.plot(epochs_2, benign_2, '-', label='Train benign loss aug')
    plt.plot(epochs_2, val_benign_2, '-', label='Val benign loss aug')
    #plt.plot(epochs, val_inSitu, '-', label='Val insitu loss')
    #plt.plot(epochs, val_invasive, '-', label='Val invasive loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # validation loss
    plt.plot(epochs_1, invasive, '-', label='Train invasive loss aug + drop')
    plt.plot(epochs_1, val_invasive, '-', label='Val invasive loss aug + drop')
    plt.plot(epochs_2, invasive_2, '-', label='Train invasive loss aug')
    plt.plot(epochs_2, val_invasive_2, '-', label='Val invasive loss aug')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

    # train and validation loss
    plt.plot(epochs_1, loss, '-', label='Training loss aug + drop')
    plt.plot(epochs_1, val_loss, '-', label='Validation loss aug + drop')
    plt.plot(epochs_2, loss_2, '-', label='Training loss aug')
    plt.plot(epochs_2, val_loss_2, '-', label='Validation loss aug')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
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

else: 
    raise ValueError(f"Supported networks are 'agunet' or 'unet', but {net} was chosen.")


# get the lowest train, validation loss
print(np.amin(loss))
print(np.amin(val_loss))
print("loss drop: ", np.amin(loss_2))
print("val loss drop: ", np.amin(val_loss_2))
