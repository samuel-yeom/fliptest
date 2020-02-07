import numpy as np
from gurobipy import Model, quicksum

def optimize(X1, X2, dists, counts1=None, counts2=None, decimals=6, verbose=True):
    '''
    Finds the optimal transport mapping between the people in Groups 1 and 2
    based on the attributes that are inputs to the machine learning model.

    The inputs are
    X1:      2-D numpy array. Each row of `X1` represents the input attributes
             of some member(s) in Group 1.
    X2:      2-D numpy array. Each row of `X2` represents the input attributes
             of some member(s) in Group 2.
    dists:   2-D numpy array. `dists[i,j]` is the distance between `X1[i]` and
             `X2[j]`.
    counts1: 1-D numpy array. `count1[i]` is the number of people with the
             input attributes given in `X1[i]`. If None, it is assumed that
             every row corresponds to exactly one person.
    counts2: Same as `count1` except with Group 2.
    
    The outputs are
    forward: Dict of dicts that represents the optimal mapping from Group 1 to
             Group 2. The keys of the outer dict are indices `i` into `X1`.
             The values are dicts whose keys are indices `j` into `X2`.
             `j` is included in the inner dictionary if and only if a nonzero
             weight is transported from `X1[i]` to `X2[j]` in the optimal
             mapping. The values of the inner dicts represent the fraction of
             `X1[1]` that is transported to `X2[j]`.
    reverse: Same as `forward`, except from Group 2 to Group 1.
    '''
    num_rows1 = X1.shape[0] #number of distinct rows in Group 1
    num_rows2 = X2.shape[0] #number of distinct rows in Group 2
    if counts1 is None:
        counts1 = np.ones(num_rows1, dtype=np.int64)
    if counts2 is None:
        counts2 = np.ones(num_rows2, dtype=np.int64)
    num_ppl1 = np.sum(counts1) #number of people in Group 1
    num_ppl2 = np.sum(counts2) #number of people in Group 2
    
    cost = np.square(dists) #cost matrix
    
    m = Model()
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.LogFile = '' #do not write log to file
    
    x = m.addVars(num_rows1, num_rows2, lb=0)
    m.addConstrs(x.sum(i, '*') == counts1[i]/num_ppl1 for i in range(num_rows1))
    m.addConstrs(x.sum('*', j) == counts2[j]/num_ppl2 for j in range(num_rows2))
    m.setObjective(quicksum(cost[i,j] * x[i,j] for i in range(num_rows1) for j in range(num_rows2)))
    m.optimize()
    
    result = np.array(m.X).reshape((num_rows1, num_rows2))
    
    #organize the result into dicts
    r12 = np.round(result * num_ppl1, decimals=decimals)
    forward = {}
    for i in range(num_rows1):
        forward[i] = {j: r12[i,j] for j in range(num_rows2) if r12[i,j] != 0}
    r21 = np.round(result.T * num_ppl2, decimals=decimals)
    reverse = {}
    for j in range(num_rows2):
        reverse[j] = {i: r21[j,i] for i in range(num_rows1) if r21[j,i] != 0}
    
    return forward, reverse
