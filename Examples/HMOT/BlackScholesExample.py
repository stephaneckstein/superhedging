import numpy as np
import tensorflow as tf
from V1 import Hedging


T = 3
d = 1
f_type = 4
MINMAX = 1
S0 = 1
T_list = [1, 2, 4]
sig = 0.25
GAMMA = 10000


# function type:
def f_obj(s):
    return MINMAX * tf.nn.relu(s[:, 2, 0] - s[:, 1, 0]/2 - s[:, 0, 0]/2)


# marginals
def gen_fun(batch_size):
    while 1:
        data = np.zeros([batch_size, T, d])
        for i in range(T):
            data[:, i, 0] = S0 * np.exp(sig * np.sqrt(T_list[i]) * np.random.normal(size=[batch_size]) - (sig ** 2) *
                                                                                                          T_list[i]/2)
        yield data


# densities d theta^{s, t} / d mu_s for homogeneous trading
def densities(x, s, t, d, l):
    dens_s = 1 / (np.sqrt(2 * np.pi * T_list[s]) * x * sig) * np.exp(
        -(np.log(x) - np.log(S0) + 0.5 * (sig ** 2) * T_list[s]) ** 2 /
        (2 * T_list[s] * sig ** 2))
    dens_t = 1 / (np.sqrt(2 * np.pi * T_list[t]) * x * sig) * np.exp(
        -(np.log(x) - np.log(S0) + 0.5 * (sig ** 2) * T_list[t]) ** 2 /
        (2 * T_list[t] * sig ** 2))
    if l == 0:
        return np.minimum(dens_s, dens_t) / dens_s
    else:
        return np.minimum(dens_s, dens_t) / dens_t


training_spec = {'batch_size_mu_0': 2 ** 13, 'batch_size_theta': 2 ** 13, 'n_train': 60000, 'n_fine': 90000,
                 'n_report': 1000, 'beta1': 0.9, 'beta2': 0.995, 'dec_interval': 250}
prob_hedge = Hedging(theta=gen_fun, mu_0=gen_fun, f=f_obj, prob_type='Markov_new', t=T, d=d, gamma=GAMMA,
              training_spec=training_spec, add_id='', hidden=((1, 64), (0, 128)), layers=((1, 5), (0, 5)),
                 densities=densities)
print('Problem type: ' + prob_hedge.prob_type)
prob_hedge.build_graph()
prob_hedge.train_model()