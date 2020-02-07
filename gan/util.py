from __future__ import print_function, division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.utils.extmath import cartesian

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop

np.set_printoptions(suppress=True)

# Data generation
def generate_random(mean1, cov1, mean2, cov2, num_pts=1000, seed=0):
    np.random.seed(seed)
    
    X1 = np.random.multivariate_normal(mean1, cov1, num_pts)
    X2 = np.random.multivariate_normal(mean2, cov2, num_pts)
    
    return X1, X2

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
    
    #np.random.seed(seed+2)
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

def process_1d():
    df = pd.read_csv('1d-10000.csv')
    arr = df.values
    
    #separate the data by the value of the protected attribute
    arr1 = arr[np.where(arr[:,2] == 1)]
    arr2 = arr[np.where(arr[:,2] == 2)]
    
    #split into input and response
    X1, y1, _ = np.split(arr1, [1, 2], axis=1)
    X2, y2, _ = np.split(arr2, [1, 2], axis=1)
    y1 = np.squeeze(y1)
    y2 = np.squeeze(y2)
    
    columns = ['arrests']
    
    return X1, X2, y1, y2, columns

# Data processing
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

def permute(X1, X2, y1, y2):
    np.random.seed(0)
    perm1 = np.arange(len(X1))
    np.random.shuffle(perm1) #permutation array for X1, y1
    np.random.seed(0)
    perm2 = np.arange(len(X2))
    np.random.shuffle(perm2) #permutation array for X2, y2

    X1 = X1[perm1]
    y1 = y1[perm1]
    X2 = X2[perm2]
    y2 = y2[perm2]
    return X1, X2, y1, y2

# Loss functions
def wasserstein_loss(y_true, y_pred):
    return -K.mean(y_true * y_pred)

def squared_l1_loss(x_inp, x_out):
    return K.mean(K.square(K.sum(K.abs(x_inp - x_out), axis=1)))

def squared_l2_loss(x_inp, x_out):
    return K.mean(K.sum(K.square(x_inp - x_out), axis=1))

# Visualization
def set_label_lim(xlabel, ylabel, xlim, ylim):
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

def plot(X1, X2, G, name1, name2, num_pts_plot=100, xlabel=None, ylabel=None, xlim=None, ylim=None, xindex=None, yindex=None):
    if xindex is None:
        xindex = 0
    if yindex is None:
        yindex = 1
    
    real1 = X1[:num_pts_plot]
    plt.scatter(real1[:,xindex], real1[:,yindex], color='b')
    real2 = X2[:num_pts_plot]
    plt.scatter(real2[:,xindex], real2[:,yindex], color='g')
    fake2 = G.predict(X1[:num_pts_plot])
    plt.scatter(fake2[:,xindex], fake2[:,yindex], color='r')
    
    set_label_lim(xlabel, ylabel, xlim, ylim)
    
    plt.legend(['Real {}'.format(name1), 'Real {}'.format(name2), 'GAN {}'.format(name2)])
    plt.show()

def plot_changes(X1, G, name1, name2, num_pts_plot=100, xlabel=None, ylabel=None, xlim=None, ylim=None, xindex=None, yindex=None):
    if xindex is None:
        xindex = 0
    if yindex is None:
        yindex = 1
    
    real1 = X1[:num_pts_plot]
    plt.scatter(real1[:,xindex], real1[:,yindex], color='b')
    fake2 = G.predict(X1[:num_pts_plot])
    plt.scatter(fake2[:,xindex], fake2[:,yindex], color='r')
    
    for i in range(num_pts_plot):
        plt.arrow(real1[i,xindex], real1[i,yindex], fake2[i,xindex]-real1[i,xindex], fake2[i,yindex]-real1[i,yindex], color='lightgray')
    
    set_label_lim(xlabel, ylabel, xlim, ylim)
    
    plt.legend(['Real {}'.format(name1), 'GAN {}'.format(name2)])
    plt.show()

def plot_critic(D, xlim, ylim, xlabel=None, ylabel=None, num_samples=51):
    xmin, xmax = xlim
    ymin, ymax = ylim
    xs = np.linspace(xmin, xmax, num_samples)
    ys = np.linspace(ymin, ymax, num_samples)
    
    pts = cartesian((xs, ys))
    zs = np.squeeze(D.predict(pts))
    
    xlen = len(xs)
    ylen = len(ys)
    zs = zs.reshape((xlen, ylen))
    
    plt.contourf(xs, ys, zs, levels=100)
    
    set_label_lim(xlabel, ylabel, xlim, ylim)
    plt.show()

def plot_marginals(X1, X2, G, name1, name2, columns, bins=20):
    for i, colname in enumerate(columns):
        plt.hist((X1[:,i], X2[:,i], G.predict(X1)[:,i]), bins=bins, density=True)
        plt.legend(['Real {}'.format(name1), 'Real {}'.format(name2), 'GAN {}'.format(name2)])
        plt.title(columns[i])
        plt.show()

def plot_dists(X1, flipset_pn, flipset_np, name1, columns, bins=40):
    for i in range(len(columns)):
        bin_edges = np.linspace(X1[:,i].min(), X1[:,i].max(), bins+1)
        
        sns.distplot(X1[:,i], bins=bin_edges)
        sns.distplot(flipset_pn[:,i], bins=bin_edges)
        sns.distplot(flipset_np[:,i], bins=bin_edges)
        sns.set(style='white')
        
        plt.legend(['All {}'.format(name1), 'Adv. {}'.format(name1), 'Disadv. {}'.format(name1)], loc='upper right')
        plt.xlabel(columns[i])
        plt.ylabel('Density')
        plt.show()

# GAN
def generator_small(data_dim):
    model = Sequential()
    
    model.add(Dense(128, input_dim=data_dim))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(data_dim))
    
    return model

def critic_small(data_dim):
    model = Sequential()
    
    model.add(Dense(128, input_dim=data_dim))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(1))
    
    return model

def generator_big(data_dim):
    model = Sequential()
    
    model.add(Dense(512, input_dim=data_dim))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(data_dim))
    
    return model

def critic_big(data_dim):
    model = Sequential()
    
    model.add(Dense(512, input_dim=data_dim))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(1))
    
    return model

def create_gan_small(data_dim, trans_loss_func=squared_l2_loss, trans_loss_wt=1e-3):
    '''Create a small Wasserstein GAN'''
    # Compile the critic (small)
    D = critic_small(data_dim)
    D.compile(loss=wasserstein_loss, optimizer=RMSprop(lr=5e-5))
    
    # Compile the generator (small)
    G = generator_small(data_dim)
    inp = Input(shape=(data_dim,))
    combined = Model(inputs=inp, outputs=[D(G(inp)), G(inp)])
    D.trainable = False #for the combined D(G(inp)) model, D is not trainable
    combined.compile(loss=[wasserstein_loss, trans_loss_func], loss_weights=[1, trans_loss_wt], optimizer=RMSprop(lr=5e-5))
    
    return D, G, combined

def create_gan_big(data_dim, trans_loss_func=squared_l1_loss, trans_loss_wt=1e-3):
    '''Create a big Wasserstein GAN'''
    # Compile the critic (big)
    D = critic_big(data_dim)
    D.compile(loss=wasserstein_loss, optimizer=RMSprop(lr=5e-5))
    
    # Compile the generator (big)
    G = generator_big(data_dim)
    inp = Input(shape=(data_dim,))
    combined = Model(inputs=inp, outputs=[D(G(inp)), G(inp)])
    D.trainable = False #for the combined D(G(inp)) model, D is not trainable
    combined.compile(loss=[wasserstein_loss, trans_loss_func], loss_weights=[1, trans_loss_wt], optimizer=RMSprop(lr=5e-5))
    
    return D, G, combined

def train(D, G, combined, X1, X2, name1, name2, epochs=20000, n_critic=5, critic_clip_value=0.01, batch_size=64, plot_progress=True, num_pts_plot=200, xlabel=None, ylabel=None, xlim=None, ylim=None, xindex=None, yindex=None):
    '''Train a Wasserstein GAN'''
    np.random.seed(0)
    
    ones = np.ones((batch_size, 1))
    
    for epoch in range(epochs+1):
        # Train the critic
        for _ in range(n_critic):
            idx = np.random.randint(0, X2.shape[0], batch_size)
            real = X2[idx]
    
            idx = np.random.randint(0, X1.shape[0], batch_size)
            fake = G.predict(X1[idx])
    
            d_loss_real = D.train_on_batch(real, ones)
            d_loss_fake = D.train_on_batch(fake, -ones)
            d_loss = (d_loss_real + d_loss_fake) / 2
    
            # Clip critic weights
            for l in D.layers:
                weights = l.get_weights()
                weights = [np.clip(w, -critic_clip_value, critic_clip_value) for w in weights]
                l.set_weights(weights)
        
        # Train the generator
        idx = np.random.randint(0, X1.shape[0], batch_size)
        inp = X1[idx]
        g_loss = combined.train_on_batch(inp, [ones, inp])
    
        # Print and plot the progress
        if plot_progress and epoch % 1000 == 0:
            print("Epoch %d [D loss: %f] [G loss: %f]" % (epoch, d_loss, g_loss[0]))
            if xlabel is None and ylabel is None and X1.shape[1] == 1: #if there is only one feature
                plot_marginals(X1, X2, G, name1, name2, [''], bins=np.arange(21) - 0.5)
            else:
                plot(X1, X2, G, name1, name2, num_pts_plot=num_pts_plot, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, xindex=xindex, yindex=yindex)
