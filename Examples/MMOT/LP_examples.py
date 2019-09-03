from LP_Programs import *

# LP MMOT test:
T = 2
d = 2
p = 2  # power of spread option
K = 0  # strike of basket option
n = 5  # number of discretization points as described in the paper
MART = 0  # if 1, includes martingale constraint. if 0, it doesn't.

P_TYPE = 'spread'  # 'basket', 'spread'
MINMAX = 'min'  # 'min', 'max'

if P_TYPE == 'spread':
    def f(x):
        return np.abs(x[T - 1, 0] - x[T - 1, 1]) ** p

    w = 1/(2*n) * np.ones(2*n+1)
    w[0] = 1/(4*n)
    w[-1] = 1/(4*n)
    margs_00 = [np.linspace(-1, 1, 2*n+1), w]
    margs_01 = [np.linspace(-1, 1, 2*n+1), w]
    w2 = 1/(4*n) * np.ones(4*n+1)
    w2[0] = 1/(8*n)
    w2[-1] = 1/(8*n)
    margs_11 = [np.linspace(-2, 2, 4*n+1), w2]
    w3 = 1/(6*n) * np.ones(6*n+1)
    w3[0] = 1/(12*n)
    w3[-1] = 1/(12*n)
    margs_10 = [np.linspace(-3, 3, 6*n+1), w3]
else:
    def f(x):
        return np.maximum(x[T - 1, 0] + x[T - 1, 1] - K, 0)

    w = 1 / (2 * n) * np.ones(2 * n + 1)
    w[0] = 1 / (4 * n)
    w[-1] = 1 / (4 * n)
    margs_01 = [np.linspace(-1, 1, 2 * n + 1), w]
    w2 = 1 / (4 * n) * np.ones(4 * n + 1)
    w2[0] = 1 / (8 * n)
    w2[-1] = 1 / (8 * n)
    margs_00 = [np.linspace(-2, 2, 4 * n + 1), w2]
    w3 = 1 / (6 * n) * np.ones(6 * n + 1)
    w3[0] = 1 / (12 * n)
    w3[-1] = 1 / (12 * n)
    margs_11 = [np.linspace(-3, 3, 6 * n + 1), w3]
    margs_10 = [np.linspace(-3, 3, 6 * n + 1), w3]


margs = [[margs_00, margs_01], [margs_10, margs_11]]
val, opti = lp_hom_mmot(margs, f, minmax=MINMAX, mart=MART, mmot_eps=0)
print('Optimal value = ' + str(val))
points_1 = margs_11[0]
points_2 = margs_10[0]
opti_2 = np.sum(opti, axis=(0, 1))
visualize_LP_sol(points_1, points_2, opti_2)
