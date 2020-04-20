import numpy as np
import tensorflow as tf
from V1 import Hedging

T = 2
d = 2
GAMMA = 2500 * T * d
MINMAX = 1  # MINMAX = 1 makes sense for the sanity check as Tongseok described
print('T = ' + str(T))
print('d = ' + str(d))
print('Minmax = ' + str(MINMAX))
CUSTOM_ID = 'TEST'

# Marginals: the j-th marginal at time i (i=0, ..., T-1) is U([-margs[i][j], margs[i][j]])
margs = [[1] * d, [2] * d]
print('Marginal specification:')
print(margs)

for run_ind in range(100):
    tf.reset_default_graph()
    print('__________________')
    print('Current Sample Run: ' + str(run_ind))
    # cost function
    np.random.seed(run_ind)
    c = np.random.random_sample([d, d])

    # true value
    n_true = 10 ** 8
    x_samp = np.random.random_sample(n_true) * margs[1][1] * 2 - margs[1][1]
    y_samp = np.zeros([n_true, d])
    for i in range(d):
        y_samp[:, i] = x_samp
    true_val = 0
    for i in range(d):
        for j in range(d):
            true_val += c[i, j] * np.mean(y_samp[:, i] * y_samp[:, j])
    print('Approximate true value by sampling: ' + str(true_val))


    def f_obj(s):
        out = 0
        for i in range(d):
            for j in range(d):
                out += c[i, j] * s[:, T - 1, i] * s[:, T - 1, j]
        return MINMAX * out


    def gen_prod(batch_size):
        while 1:
            yield (2 * np.random.random_sample([batch_size, T, d]) - 1) * margs


    session_conf = tf.ConfigProto(intra_op_parallelism_threads=2,
                                  inter_op_parallelism_threads=2)  # to obtain similar computation times as in the paper
    training_spec = {'batch_size_mu_0': 2 ** (5 + 2 * d), 'batch_size_theta': 2 ** (7 + 2 * d), 'n_train': 40000,
                     'n_fine': 15000,
                     'n_report': 5000}
    print('Training spec:')
    print(training_spec)
    print('Gamma = ' + str(GAMMA))
    prob_hedge = Hedging(theta=gen_prod, mu_0=gen_prod, f=f_obj, prob_type='MMOT', t=T, d=d, gamma=GAMMA,
                         training_spec=training_spec, add_id=CUSTOM_ID, tf_conf=session_conf, hidden=((1, 32), (0, 32)),
                         layers=((1, 4), (0, 4)))
    print('Problem type: ' + prob_hedge.prob_type)
    prob_hedge.build_graph()
    prob_hedge.train_model()

    fin_val_c = np.mean(prob_hedge.value_list[-10000:])
    fin_val_p1 = np.mean(prob_hedge.primal_1_value_list[-10000:])
    fin_val_p2 = np.mean(prob_hedge.primal_2_value_list[-10000:])

    print('Final value: ' + str(fin_val_p2))

    # save value if desired.
