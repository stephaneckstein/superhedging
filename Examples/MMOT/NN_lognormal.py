import numpy as np
import tensorflow as tf
from V1 import Hedging


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

if ftype == 0:
    def f_obj(s):
        return MINMAX * tf.pow(tf.abs(s[:, T - 1, 0] - s[:, T - 1, 1]), p_spread)

if ftype == 1:
    def f_obj(s):
        return MINMAX * tf.nn.relu(s[:, T - 1, 0] + s[:, T - 1, 1] - K_basket)

# marginals
def gen_fun(batch_size):
    while 1:
        data = np.zeros([batch_size, T, d])
        for i in range(T):
            for j in range(d):
                data[:, i, j] = S0 * np.exp(sig_list[i][j] * np.random.normal(size=[batch_size]) - (sig_list[i][j] ** 2)/2)
        yield data


session_conf = tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=2)  # to obtain roughly the computation times reported in the paper
training_spec = {'batch_size_mu_0': 2 ** (8 + T * d), 'batch_size_theta': 2 ** (9 + T * d),
                 'n_train': 40000, 'n_fine': 15000, 'n_report': 5000}
prob_hedge = Hedging(theta=gen_fun, mu_0=gen_fun, f=f_obj, prob_type='MMOT', t=T, d=d, gamma=GAMMA,
                     training_spec=training_spec,
                     hidden=((1, 64), (0, 64)), layers=((1, 4), (0, 5)), tf_conf=session_conf)
print('Problem type: ' + prob_hedge.prob_type)
prob_hedge.build_graph()
prob_hedge.train_model()