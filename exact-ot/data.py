import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.spatial import distance

def generate_lipton(scale=True, num_pts=1000, seed=0):
    '''
    Synthetic data used by Lipton et al. in arXiv:1711.07076
    '''
    np.random.seed(seed)
    work_exp_m = np.random.poisson(31, size=num_pts) - np.random.normal(20, 0.2, size=num_pts)
    work_exp_f = np.random.poisson(25, size=num_pts) - np.random.normal(20, 0.2, size=num_pts)
    
    np.random.seed(seed+1)
    hair_len_m = 35 * np.random.beta(2, 7, size=num_pts)
    hair_len_f = 35 * np.random.beta(2, 2, size=num_pts)
    
    np.random.seed(seed+2)
    ym = np.random.uniform(size=num_pts) < 1 / (1 + np.exp(25.5 - 2.5*work_exp_m))
    yf = np.random.uniform(size=num_pts) < 1 / (1 + np.exp(25.5 - 2.5*work_exp_f))
    
    if scale: #scale the input attributes to zero mean and unit variance
        work_exp = np.concatenate((work_exp_m, work_exp_f))
        work_exp = preprocessing.scale(work_exp)
        work_exp_m = work_exp[:num_pts]
        work_exp_f = work_exp[num_pts:]
        hair_len = np.concatenate((hair_len_m, hair_len_f))
        hair_len = preprocessing.scale(hair_len)
        hair_len_m = hair_len[:num_pts]
        hair_len_f = hair_len[num_pts:]
    
    #combine the input attributes to create the input matrix
    Xm = np.stack((work_exp_m, hair_len_m), axis=1)
    Xf = np.stack((work_exp_f, hair_len_f), axis=1)
    columns = ['work_exp', 'hair_len']
    
    return Xm, Xf, ym, yf, columns

def convert_age_ssl(age_str):
    '''
    Converts the strings representing age in the Chicago SSL dataset to
    integers.
    '''
    try:
        age_int = int(age_str[0:2])
    except ValueError:
        if age_str == 'less than 20':
            age_int = 10
        else:
            raise ValueError(age_str)
    return age_int

def process_ssl_gender():
    agecolname = 'PREDICTOR RAT AGE AT LATEST ARREST'
    
    df = pd.read_csv('chicago-ssl-clean.csv')
    
    #convert age to integers
    df[agecolname] = df[agecolname].map(convert_age_ssl)
    
    #scale the input attributes
    arr = df.values
    arr[:,1:9] = preprocessing.scale(arr[:,1:9])
    
    #separate data by gender and then remove the race and gender columns
    arr_m = arr[np.where(arr[:,9] == 'M')][:,:9].astype(np.float64)
    arr_f = arr[np.where(arr[:,9] == 'F')][:,:9].astype(np.float64)
    
    #split into input and response
    ym, Xm = np.split(arr_m, [1], axis=1)
    yf, Xf = np.split(arr_f, [1], axis=1)
    ym = np.squeeze(ym)
    yf = np.squeeze(yf)
    
    #get names of input columns
    columns = list(df.columns[1:9])
    
    return Xm, Xf, ym, yf, columns

def process_ssl_race():
    agecolname = 'PREDICTOR RAT AGE AT LATEST ARREST'
    
    df = pd.read_csv('chicago-ssl-clean.csv')
    
    #convert age to integers
    df[agecolname] = df[agecolname].map(convert_age_ssl)
    
    #scale the input attributes
    arr = df.values
    arr[:,1:9] = preprocessing.scale(arr[:,1:9])
    
    #separate data by race and then remove the race and gender columns
    arr_w = arr[np.where(arr[:,10] == 'WHI')][:,:9].astype(np.float64)
    arr_b = arr[np.where(arr[:,10] == 'BLK')][:,:9].astype(np.float64)
    
    #split into input and response
    yw, Xw = np.split(arr_w, [1], axis=1)
    yb, Xb = np.split(arr_b, [1], axis=1)
    yw = np.squeeze(yw)
    yb = np.squeeze(yb)
    
    #get names of input columns
    columns = list(df.columns[1:9])
    
    return Xw, Xb, yw, yb, columns
    
def get_distance_ssl(row1, row2):
    dist = distance.cityblock(row1, row2)
    return dist

def get_all_distances_ssl(X1, X2):
    '''
    The output `dists` is a 2-D array such that `dists[i,j]` is the distance
    between `Xm[i]` and `Xf[j]`.
    '''
    dists = distance.cdist(X1, X2, metric='cityblock')
    return dists
