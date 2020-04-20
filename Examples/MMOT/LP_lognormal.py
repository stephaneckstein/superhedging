import numpy as np
import tensorflow as tf
from V1 import Hedging
from scipy.stats import lognorm


d = 2
S0 = 1

ftype = 0  # 0 for spread option, 1 for basket option
fpar = 2  # power of spread option, respectively strike for basket option.
T = 2  # number of time points included in optimization.
MINMAX = 1  # -1 for minimization. +1 for maximization.

if ftype == 0:
    p_spread = fpar
if ftype == 1:
    K_basket = fpar
if T == 2:
    sig_list = [[0.1, 0.1], [0.2, 0.2]]
if T == 3:
    sig_list = [[0.1, 0.1], [0.11, 0.19], [0.2, 0.2]]
GAMMA = 5000 * T * d


if MINMAX == -1:
    LPMINMAX = 'min'
else:
    LPMINMAX = 'max'


if ftype == 0:
    def f_obj(s):
        return MINMAX * tf.pow(tf.abs(s[:, T - 1, 0] - s[:, T - 1, 1]), p_spread)

    def f_obj_np(s):
        return np.abs(s[T-1, 0] - s[T-1, 1]) ** p_spread

if ftype == 1:
    def f_obj(s):
        return MINMAX * tf.nn.relu(s[:, T - 1, 0] + s[:, T - 1, 1] - K_basket)

    def f_obj_np(s):
        return np.maximum(0, s[T - 1, 0] + s[T - 1, 1] - K_basket)


def gen_fun(batch_size):
    while 1:
        data = np.zeros([batch_size, T, d])
        for i in range(T):
            for j in range(d):
                data[:, i, j] = S0 * np.exp(sig_list[i][j] * np.random.normal(size=[batch_size]) - (sig_list[i][j] ** 2)/2)
        yield data


def return_lambda(i, j):
    return lambda x: S0 * np.exp(-sig_list[i][j] ** 2 / 2) * lognorm.ppf(x, s=sig_list[i][j], scale=1)


quantile_funs = [[return_lambda(i, j) for j in range(d)] for i in range(T)]

session_conf = tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=2)
training_spec = {'batch_size_mu_0': 2 ** 14, 'batch_size_theta': 2 ** 15, 'n_train': 40000, 'n_fine': 15000,
                 'n_report': 5000}
prob_hedge = Hedging(theta=gen_fun, mu_0=gen_fun, f=f_obj, prob_type='MMOT', t=T, d=d, gamma=GAMMA, training_spec=training_spec,
                     hidden=((1, 64), (0, 64)), layers=((1, 4), (0, 5)), tf_conf=session_conf, f_np=f_obj_np, quantile_funs=quantile_funs)
print('Problem type: ' + prob_hedge.prob_type)
val, opti, a_list, w_list = prob_hedge.solve_LP(dis_each=int(np.round((2.5 * 10 ** 6) ** (1/(T*d)))), n_samples=10 ** 8,
                                                quant_arr_calc=1, minmax=LPMINMAX,
                                                eps=10 ** -16, mmot_eps=0, given_ev=S0 * np.ones([d]), method='a')

print('LP value: ' + str(val))