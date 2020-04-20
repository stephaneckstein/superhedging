import numpy as np
from LP_Programs import *

T = 2
d = 2

if d == 2:
    n = 18
if d == 3:
    n = 4
if d == 4:
    n = 2

# LP Sanity Check:
for run_ind in range(100):
    np.random.seed(run_ind)
    c = np.random.random_sample([d, d])

    print('Number of parameters in the LP: ' + str((2 * n + 1) ** d * (4 * n + 1) ** d))

    n_true = 10 ** 8
    x_samp = np.random.random_sample(n_true) * 4 - 2
    y_samp = np.zeros([n_true, d])
    for i in range(d):
        y_samp[:, i] = x_samp
    true_val = 0
    for i in range(d):
        for j in range(d):
            true_val += c[i, j] * np.mean(y_samp[:, i] * y_samp[:, j])
    print('Approximate true value by sampling: ' + str(true_val))

    def f_obj(x):
        out = 0
        for i in range(d):
            for j in range(d):
                out += c[i, j] * x[T - 1, i] * x[T - 1, j]
        return out


    # marginals spread:
    w = 1 / (2 * n) * np.ones(2 * n + 1)
    w[0] = 1 / (4 * n)
    w[-1] = 1 / (4 * n)
    margs_0 = [np.linspace(-1, 1, 2 * n + 1), w]
    w2 = 1 / (4 * n) * np.ones(4 * n + 1)
    w2[0] = 1 / (8 * n)
    w2[-1] = 1 / (8 * n)
    margs_1 = [np.linspace(-2, 2, 4 * n + 1), w2]

    margs = [[margs_0] * d, [margs_1] * d]
    val, opti = lp_hom_mmot(margs, f_obj, minmax='max', mart=1, hom=0, mmot_eps=0)
    print('Optimal Value by LP: ' + str(val))