from LP_Programs import *

T_MAX = 9
T = 2
d = 1
K = 1


def f_forward(x):
    return np.maximum(x[-1, 0] - K * x[-2, 0], 0)


def f_asian(x):
    return np.maximum(np.sum(x)/T - K, 0)


margs = []
n_vars = 1
for t in range(T_MAX-T, T_MAX):
    n_vars *= t+1
    margs.append([[np.linspace(100-t, 100+t, t+1), np.ones(t+1) * 1/(t+1)]])
print(n_vars)
ov, opti = lp_hom_mmot(margs, f_forward, minmax='min', hom=1, mart=1)
ov2, opti2 = lp_hom_mmot(margs, f_forward, minmax='max', hom=1, mart=1)
