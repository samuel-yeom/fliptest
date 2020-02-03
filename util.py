import numpy as np

def get_index_arrays(forward_dict, reverse_dict):
    '''
    Given the outputs of the `optimize.optimize` function on `X1` and `X2`,
    creates a simpler form of these outputs. `X1` and `X2` must have the same
    number of rows.
    
    The outputs `forward` and `reverse` are 1-D numpy arrays. If `X1[i]`
    maps to `X2[j]`, we have `forward[i] = j` and `reverse[j] = i`.
    '''
    assert len(forward_dict) == len(reverse_dict)
    num_pts = len(forward_dict)
    
    min_weight = min([max(forward_dict[i].values()) for i in forward_dict])
    assert min_weight > 0.5 #ensures a one-to-one mapping
    
    #initialize the arrays to -1
    forward = -np.ones(num_pts, dtype=np.int64)
    reverse = -np.ones(num_pts, dtype=np.int64)
    
    #fill the arrays with the correct indices
    for i in forward_dict:
        forward[i] = max(forward_dict[i].keys(), key=lambda j: forward_dict[i][j])
    for j in reverse_dict:
        reverse[j] = max(reverse_dict[j].keys(), key=lambda i: reverse_dict[j][i])
    
    return forward, reverse

def get_mean_dist(X1, X2, forward):
    '''
    Compute the mean L1 distance between rows in `X1` and `X2` that map to
    each other.
    '''
    return np.mean(np.sum(np.abs(X1 - X2[forward]), axis=1))
