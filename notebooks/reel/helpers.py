import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def metrics_viewer(model):
    acu = model.history['binary_accuracy']
    val_acu = model.history['val_binary_accuracy']
    loss = model.history['loss']
    val_loss = model.history['val_loss']
    epochsticks = range(len(acu))
    plt.figure(figsize = (15, 10))
    ax1 = plt.subplot(311)
    ax1.plot(epochsticks, acu, 'b', label = 'Training Accuracy')
    ax1.plot(epochsticks, val_acu, 'g', label = 'Validation Accuracy')
    ax1.set_title('Training and validation mse')
    ax1.grid()
    ax1.legend()
    ax2 = plt.subplot(312, sharex = ax1)
    ax2.plot(epochsticks, loss, 'b', label = 'Training loss')
    ax2.plot(epochsticks, val_loss, 'g', label = 'Validation loss')
    ax2.set_title('Training and validation loss')
    ax2.grid()
    ax2.legend()
    return plt

def color_label_maker(array, num_colors):
    multi = round(len(array)/num_colors)
    rest = len(array)-multi*num_colors
    colors = [1]*multi+[1]*rest
    for i in range(num_colors-1):
        colors.extend([i+2]*multi)
    return colors