import numpy as np
import matplotlib.pyplot as plt

def set_label_lim(xlabel, ylabel, xlim, ylim):
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

def plot_pairs(X1, X2, forward, reverse_colors=False, xlabel=None, ylabel=None, xlim=None, ylim=None, num_pairs_plot=100):
    '''
    Makes a scatter plot of the rows in `X1` and `X2`, with the i-th row in
    `X1` connected by a line to the i-th row in `X2`.
    
    The inputs are
    X1: nx2 numpy array, each row of which represents a member of Group 1
    X2: nx2 numpy array, each row of which represents a member of Group 2
    forward: 1-D numpy array of size n that represents a mapping from `X1` to
             `X2`. `forward[i] == j` if and only if `X1[i]` maps to `X2[j]`.
             This is the same format as the outputs of `get_index_arrays` in
             `util.py`.
    
    The resulting scatter plot uses blue dots for `X1` and red dots for `X2`,
    unless `reverse_colors` is True. In that case, the colors are swapped.
    '''
    assert X1.shape[0] == X2.shape[0] >= num_pairs_plot
    assert X1.shape[1] == X2.shape[1] == 2
    
    if not reverse_colors:
        color1 = 'blue'
        color2 = 'red'
    else:
        color1 = 'red'
        color2 = 'blue'
    
    X2_map = X2[forward]
    
    plt.scatter(X1[:num_pairs_plot,0], X1[:num_pairs_plot,1], color=color1)
    plt.scatter(X2_map[:num_pairs_plot,0], X2_map[:num_pairs_plot,1], color=color2)
    for i in range(num_pairs_plot):
        plt.arrow(X1[i,0], X1[i,1], X2_map[i,0]-X1[i,0], X2_map[i,1]-X1[i,1], color='lightgray')
    set_label_lim(xlabel, ylabel, xlim, ylim)
    plt.show()

def plot_flips(X1, X2, forward, func, reverse_colors=False, xlabel=None, ylabel=None, xlim=None, ylim=None):
    '''
    Same as `visualize`, but only shows the rows whose `func` values flip
    after the mapping. More formally, shows the rows in `X1` such that
    `func(row) == True` but whose counterpart in `X2` satisfies
    `func(row) == False`
    '''
    assert X1.shape[0] == X2.shape[0] == forward.shape[0]
    assert X1.shape[1] == X2.shape[1] == 2
    
    X1_true = []
    X2_false = []
    
    num_pts = X1.shape[0]
    for i in range(num_pts):
        j = forward[i]
        if func(X1[i]) and not func(X2[j]):
            X1_true.append(X1[i])
            X2_false.append(X2[j])
    X1_true = np.array(X1_true).reshape((-1, 2))
    X2_false = np.array(X2_false).reshape((-1, 2))
    
    if not reverse_colors:
        color1 = 'blue'
        color2 = 'red'
    else:
        color1 = 'red'
        color2 = 'blue'
    
    plt.scatter(X1_true[:,0], X1_true[:,1], color=color1)
    plt.scatter(X2_false[:,0], X2_false[:,1], color=color2)
    for i in range(len(X1_true)):
        plt.arrow(X1_true[i,0], X1_true[i,1], X2_false[i,0]-X1_true[i,0], X2_false[i,1]-X1_true[i,1], color='lightgray')
    set_label_lim(xlabel, ylabel, xlim, ylim)
    plt.show()
