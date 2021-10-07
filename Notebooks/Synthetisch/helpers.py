import numpy as np
import random
import pandas as pd
import os
import getpass
import subprocess
import matplotlib.pyplot as plt
import tensorflow.keras as keras

def find_drive_path():
    username = getpass.getuser() # Finding the username of the user
    bashCommand1 = "lsblk -o NAME -nl"
    process1 = subprocess.Popen(bashCommand1.split(), stdout=subprocess.PIPE)
    output1, error1 = process1.communicate()
    array_disks = output1.decode('ascii').split("\n")
    if "sda1" in array_disks:
        path = "/dev/sda1"
    elif "sda" in array_disks:
        path = "/dev.sda"
    bashCommand2 = f"findmnt -A {path}" # to find the path of the place where mounted
    process2 = subprocess.Popen(bashCommand2.split(), stdout=subprocess.PIPE)
    output2, error2 = process2.communicate()
    drive_path = output2.decode('ascii').split('\n')[1].split(' ')[0] # Using the output
    return drive_path

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
    
def truncate(n, decimals=0):
    """
    Function to truncate a large float into the rounded (floor) value
    ex: 0.112233 -> 0.1
    """
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

def allele_freq_1(p_min, p_max,N_SNPS):
    """
    Create first quadratic function to change allele frequency
    and obtain an array of allele frequencies small to big
    [p_min, 0.1, 0.2, ... , 0.7, p_max]
    """
    i = p_min
    array = []
    while i < p_max:
        array.append(truncate(i,1))
        i += ((p_max-p_min)/N_SNPS)
    return array

def allele_freq_2(p_min, p_max,N_SNPS):
    """
    Create first quadratic function to change allele frequency
    and obtain an array of allele frequencies big to small
    [p_max, 0.7, 0.6, ... , 0.1, p_min]
    """
    i = p_min
    array = []
    while i < p_max:
        array.append(truncate(i,1))
        i += (((p_max-p_min)**2/N_SNPS))
        if len (array) == N_SNPS:
            i = p_max
    array = array[::-1]
    return array

def allele_matrix(frequency_array, N_SNPS):
    """
    Create for each snp an array of alleles respecting the frequency distribution from the frequency array
    snp1: [0, 0, 0, 0, 0, 1, 1, 1, 2, 2]
    ...
    snpn: [2, 2, 2, 2, 1, 1, 1, 1, 0, 0]
    """
    allele_matrix = []
    for i in range(N_SNPS):
        p = frequency_array[i]
        q = 1-p
        p2 = int(truncate((p ** 2),1)*10)
        pq = int(truncate((2*p*q),1)*10)
        q2 = int(truncate((q ** 2),1)*10)
        if p2 == 0:
            p2 = 10 - pq - q2
        elif q2 == 0:
            q2 = 10 - pq - p2
        allele_matrix.append(np.concatenate([[0] * p2, [1] * pq, [2]* q2],axis = 0).tolist())
    return allele_matrix

def create_syn_data_monogeen(N_HUMANS, allele_matrix, population):
    """
    Create synthetic data from Number of individuals wanted and respecting the allele frequency matrix
    For this synthetic dataset the individuals are cases if they have genotype = 2 as the third snp
    (index = 2)
    """
    data_dict = [] 
    for i in range(N_HUMANS):
        geno = []
        for snp in range(len(allele_matrix)):
            geno.append(random.choice(allele_matrix[snp]))
        if geno[2] == 2:
            data_dict.append({"IID":f"{i}{population}",
                              'Pheno':1,
                              'Geno':geno,
                              'State': f"{population}sick"
                         })
        else:
            data_dict.append({"IID":f"{i}{population}",
                              'Pheno':0,
                              'Geno':geno,
                              'State': f"{population}healthy"
                         })
    return data_dict

def snp_counter(df):
    """
    Grouping snp's together by their genotype between all the individuals
    """
    df_matrix = []
    for human in range(len(df)):
        snp = []
        for i in range(len(df["Geno"][0])):
            snp.append(
                {'id': i,
                 'val': 1,
                 'genotype': df['Geno'][human][i],
                 'id2': f"{i}_{df['Geno'][human][i]}"
                }
            )
        df_matrix.append(snp)
    array = []
    for i in  range(len(df_matrix)):
        for b in range(len(df_matrix[0])):
            array.append(df_matrix[i][b])
    df_sorted = pd.DataFrame(array)
    df_grouped = df_sorted[['val','id2','id', 'genotype']].groupby('id2').sum()
    df_grouped['id'] = df_grouped['id']/df_grouped['val']
    df_grouped['genotype'] = df_grouped['genotype']/df_grouped['val']
    return df_grouped.sort_values(by=['id'])

def p_finder(genos,nr_splits):
    p_matrix = []
    for i in range(len(genos[0])):
        geno = genos[:,i] - 1
        major = np.where(geno == -1, 0, geno)
        minor = np.where(geno == 1, 0, geno)
        x_axis = np.array(range(0, len(major), 1))
        arrays = np.split(geno, nr_splits)
        array_sums = []
        for array in arrays:
            array_sums.append(sum(array)/len(array))
        p = list(map(lambda x:(x+1)/2, array_sums))
        p_matrix.append(p)
    return p_matrix
def polifit_model(x_reduced, p_matrix, x_axis):
    q_pred_mat = []
    try:
        polifit_model = keras.models.load_model("../../Data/MyPoliFitModel")
    except:
        polifit_model = keras.Sequential()
        polifit_model.add(keras.layers.Dense(units = 1, activation = 'linear', input_shape=[1]))
        polifit_model.add(keras.layers.Dense(units = 64, activation = 'elu'))
        polifit_model.add(keras.layers.Dense(units = 64, activation = 'elu'))
        polifit_model.add(keras.layers.Dense(units = 1, activation = 'linear'))
        polifit_model.compile(loss='mse', optimizer="adam")
        polifit_model.fit( x_reduced, np.array(p_matrix[0]), epochs=1000, verbose=0)
        polifit_model.save("../../Data/MyPoliFitModel")
    for p in p_matrix:
        polifit_model.fit( x_reduced, np.array(p), epochs=100, verbose=0)
        y_predicted = polifit_model.predict(x_axis)
        q_pred_mat.append(y_predicted)
    return q_pred_mat