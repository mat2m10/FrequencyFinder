import numpy as np
import random
import pandas as pd

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