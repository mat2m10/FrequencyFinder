import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from scipy.fft import fft, fftfreq, fftshift

def mirror_geno(geno):
    mirror = np.ones(shape=(geno.shape[0],geno.shape[1]))*2 - geno
    return mirror

def geno_to_allele_converter(pandas_geno):
    """
    Function to convert genotype data looking like this [0, 2, 1, 2, ... 0]
    0 = homozygous minor allele, 
    1 = heterozygous, 
    2 = homozygous major allele
    into allele data looking like this [[0,2], [2,0], [1,1], [2,0] ... [0,2]]:
    [0,2] = homozygous minor allele,
    [1,1] = heterozygous,
    [2,0] = homozygous major allele
    """
    geno1 = np.array(pandas_geno.tolist())
    geno2 = mirror_geno(geno1)
    allele = np.stack((geno1, geno2), axis=2).astype(int)
    return allele

def splitter(numpy_array, ratio):
    """
    Function to split the data into a train and test set of multiple features
    """
    splitted = []
    for element in numpy_array:
        [train, test] = train_test_split(element, test_size=ratio, random_state=3)
        splitted.append(train)
        splitted.append(test)
    return splitted

def tensor_converter(array):
    """
    Transform multiple features in tensors for easier computation
    """
    tensors = []
    for element in array:
        tensors.append(tf.constant(element, dtype = tf.float32))
    return tensors

def frequency_geno(panda_geno):
    """
    Convert the genotype in a the fourier transform
    """
    Ys = np.array(panda_geno.tolist())
    Ys = mirror_geno(Ys)
    fourier = []
    for y in Ys:
        N = len(y)
        yf = fft(y)
        yf = 1/N * np.abs(yf)
        mid = len(yf)/2
        yf = yf[int(mid):]
        yf = yf/max(yf)
        fourier.append(yf.tolist())
    return np.array(fourier)

def allele_sorting_based_on_max_one_and_two_individual(panda_geno):
    """
    One individuals will have a maximum of major alleles [2,2,2]
    We will sort his genotype and use the same sorting mapping on other individuals
    """
    genos = np.array(panda_geno.tolist())
    geno = mirror_geno(genos)
    df = pd.DataFrame(genos)
    index_max = df.sum(axis = 1).idxmax()
    print(index_max)
    x = genos[index_max]
    ind = np.unravel_index(np.argsort(x, axis = None),x.shape)
    sorted_array = []
    for geno in genos:
        sorted_array.append(geno[ind].tolist())
    return [np.array(sorted_array), ind]
def only_minor(panda_geno):
    """
    We put all the major alleles and the heterozygous allele to 0
    and the minor alleles to 1
    """
    genos = np.array(panda_geno.tolist())
    genos = mirror_geno(genos)
    genos = np.where(genos == 1., 0, genos)/2
    return genos

    