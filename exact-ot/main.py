from __future__ import print_function

import numpy as np
from scipy.spatial import distance
import sklearn
import argparse

import data
import optimize_gurobi as optimize
import util
import visualize

def run_ssl(num_pts=1000, seed=0):
    Xw, Xb, yw, yb, columns = data.process_ssl_race()
    
    #shuffle the rows at use only `num_pts` of the rows from each race
    Xw, yw = sklearn.utils.shuffle(Xw, yw, random_state=seed)
    Xb, yb = sklearn.utils.shuffle(Xb, yb, random_state=seed)
    Xw = Xw[:num_pts]
    Xb = Xb[:num_pts]
    yw = yw[:num_pts]
    yb = yb[:num_pts]
    
    dists = data.get_all_distances_ssl(Xw, Xb)
    
    print('Solving for the exact optimal transport mapping...')
    forward, reverse = optimize.optimize(Xw, Xb, dists)
    forward, reverse = util.get_index_arrays(forward, reverse)
    
    print('Mean L1 distance:', util.get_mean_dist(Xw, Xb, forward))
    
    return Xw, Xb, yw, yb, columns, forward, reverse

def run_lipton(num_pts=1000, plots=False):
    Xm, Xf, ym, yf, columns = data.generate_lipton(num_pts=num_pts)
    dists = distance.cdist(Xm, Xf, metric='cityblock')
    
    print('Solving for the exact optimal transport mapping...')
    forward, reverse = optimize.optimize(Xm, Xf, dists)
    forward, reverse = util.get_index_arrays(forward, reverse)
    
    print('Mean L1 distance:', util.get_mean_dist(Xm, Xf, forward))
    
    if plots:
        visualize.plot_pairs(Xm, Xf, forward, xlabel='work_exp', ylabel='hair_len', xlim=(-3,3), ylim=(-3,3), num_pairs_plot=100)
        
        func = lambda row: np.dot(row, [1.2956, 1.0862]) > 0.8668 #output of a "fair" model
        print('Hired women and their unhired male counterparts')
        visualize.plot_flips(Xf, Xm, reverse, func, reverse_colors=True, xlabel='work_exp', ylabel='hair_len', xlim=(-3,3), ylim=(-3,3))
        print('Hired men and their unhired female counterparts')
        visualize.plot_flips(Xm, Xf, forward, func, xlabel='work_exp', ylabel='hair_len', xlim=(-3,3), ylim=(-3,3))
    
    return Xm, Xf, ym, yf, columns, forward, reverse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['ssl', 'lipton'])
    args = parser.parse_args()
    
    if args.dataset == 'ssl':
        Xw, Xb, yw, yb, columns, forward, reverse = run_ssl()
    
    if args.dataset == 'lipton':
        Xm, Xf, ym, yf, columns, forward, reverse = run_lipton(plots=True)
