Code for the paper "FlipTest: Fairness Testing via Optimal Transport". Currently, the code supports exact optimal transport mappings for the Lipton synthetic hiring dataset (Lipton et al., 2018) and the Strategic Subject List (City of Chicago, 2017), as well as GAN approximations for the Lipton hiring dataset.

## Links to the paper
* [Conference version](https://dl.acm.org/doi/abs/10.1145/3351095.3372845)
* [arXiv version](https://arxiv.org/abs/1906.09218)

## How to run the code
You will need an installation of Python 3 with some commonly used data analysis packages, such as `numpy`, `scipy`, `sklearn`, `pandas`, and `matplotlib`.

### Exact optimal transport
Requires [Gurobi](https://www.gurobi.com/). This is proprietary software, but free licenses are available for academic users.

Go to the `exact-ot/` directory and run `python main.py lipton` or `python main.py ssl` to find the exact optimal transport mapping on the Lipton hiring dataset or the Strategic Subject List, respectively. Alternatively, you can import `main.py` and run
```
X1, X2, y1, y2, columns, forward, reverse = run_lipton() #or run_ssl()
```
to load the mapping into the current namespace. `X1` and `X2` are 2-D numpy arrays of the input features, `y1` and `y2` are 1-D numpy arrays containing the response, and `columns` is a list of the feature names. `forward` and `reverse` are defined by the following relation: if `X1[i]` maps to `X2[j]` under the optimal transport mapping, then `forward[i] = j` and `reverse[j] = i`.

### GAN approximation
In the `gan/` directory, there are Jupyter notebooks containing the results of the GAN experiments on the Lipton hiring dataset. Due to GPU nondeterminism, these results are slightly different from those reported in the paper. The notebooks can be rerun with TensorFlow 2.0 if desired.

## References
(Lipton et al., 2018) Zachary Lipton, Julian McAuley, and Alexandra Chouldechova. Does mitigating ML's impact disparity require treatment disparity? Neural Information Processing Systems, 2018.

(City of Chicago, 2017) City of Chicago. Strategic Subject List. https://data.cityofchicago.org/Public-Safety/Strategic-Subject-List/4aki-r3np, 2017.
