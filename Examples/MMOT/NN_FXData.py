import numpy as np
import tensorflow as tf
from NNStructures import ff_net


T = 3
Tenors = [1.0, 1.4986301369863013, 2.0]
DIM = 2  # fix at the moment
ADD_INFO = 0  # if 0, no trading in third currency pair. If 1, trading in third currency pair.
MINMAX = -1  # if -1, minimization. if +1, maximization.
MART = 0  # if 0, no dynamic trading. if 1, dyanamic trading.
GAMMA = 12000


BATCH_SIZE = 2 ** (7 + 2 * T)
F_TYPE = 'Spread'  # either 'Spread' or 'Asian'.
K_ASIAN = 1.32  # 1.32 is GBPUSD rate for t = 0 (28.01.2019)
K_SPREAD = 0
N = 25000*T
N_FINE = 25000*T
C_BALA = 1.146  # 1.146 is the GBPEUR rate for t = 0 (28.01.2019)
# EURUSD is 1.14 for t = 0 (28.01.2019)
Assets = ['GBPUSD', 'EURUSD']
A_add = 'EURGBP'  # should correspond to options on the fraction Assets[1]/Assets[0]

P12 = [[np.loadtxt('Data/ex1_price'+Assets[i]+'_'+str(Tenors[t])) for i in range(DIM)] for t in range(T)]
K12 = [[np.loadtxt('Data/ex1_strike'+Assets[i]+'_'+str(Tenors[t])) for i in range(DIM)] for t in range(T)]
print(K12)
Size12 = [[len(P12[t][i]) for i in range(DIM)] for t in range(T)]
Max_Size = max(max(Size12[t]) for t in range(T))

P_add = [np.loadtxt('Data/ex1_price'+A_add+'_'+str(Tenors[t])) for t in range(T)]
K_add = [np.loadtxt('Data/ex1_strike'+A_add+'_'+str(Tenors[t])) for t in range(T)]
Size_add = [len(P_add[t]) for t in range(T)]
Max_Size_add = max(Size_add)

I_ref = 0.25  # hedging inequality is tested in the intervals [lowest_strike-I_ref, highest_strike+I_ref]
lowest_strike = [[np.min(K12[t][i]) for i in range(DIM)] for t in range(T)]
highest_strike = [[np.max(K12[t][i]) for i in range(DIM)] for t in range(T)]


def theta(batch_size):
    while 1:
        data = np.random.random_sample([batch_size, T, DIM])
        for t in range(T):
            for i in range(DIM):
                data[:, t, i] = lowest_strike[t][i] - I_ref + data[:, t, i] * (highest_strike[t][i]-lowest_strike[t][
                    i]+2*I_ref)
        yield data


if F_TYPE == 'Asian':
    def f(u):
        return MINMAX * tf.nn.relu(tf.reduce_mean(u[:, :, 0], axis=1)-K_ASIAN)
elif F_TYPE == 'Spread':
    def f(u):
        return MINMAX * tf.nn.relu(tf.reduce_mean(u[:, :, 0], axis=1) - C_BALA * tf.reduce_mean(u[:, :, 1], axis=1) - K_SPREAD)
else:
    def f(u):
        return MINMAX * tf.nn.relu(u[:, T-1, 1] + u[:, T-1, 0] - 1.1 - 1.3)

X = tf.placeholder(shape=[None, T, DIM], dtype=tf.float32)
Stat_hedge = tf.get_variable('stat', shape=[Max_Size, T, DIM], initializer=tf.random_normal_initializer(),
                             dtype=tf.float32)
Add_hedge = tf.get_variable('add', shape=[Max_Size_add, T], initializer=tf.random_normal_initializer(),
                             dtype=tf.float32)

Stat_price = 0
Stat_term = 0
for t in range(T):
    for i in range(DIM):
        for s in range(Size12[t][i]):
            Stat_price += Stat_hedge[s, t, i] * P12[t][i][s]
            Stat_term += Stat_hedge[s, t, i] * tf.nn.relu(X[:, t, i] - K12[t][i][s])

Add_price = 0
Add_term = 0
if ADD_INFO > 0:
    for t in range(T):
        for s in range(Size_add[t]):
            Add_price += Add_hedge[s, t] * P_add[t][s]
            Add_term += Add_hedge[s, t] * tf.nn.relu(X[:, t, 1]/X[:, t, 0] - K_add[t][s])

Mart_term = 0
if MART > 0:
    for t in range(T-1):
        for i in range(DIM):
            input_1 = X[:, 0:t, 0]
            for i in range(1, DIM):
                input_1 = tf.concat([input_1, X[:, 0:t, i]], axis=1)
            Mart_term += tf.reduce_sum(ff_net(input_1, 'dyn_'+str(t)+'_'+str(i), feats=((1, 'identity'),), n_layers=4,
                                              hidden_dim=32, input_dim=1, output_dim=1, activation='ReLu'),
                                       axis=1) * (X[:, t+1, i] - X[:, t, i])

Fterm = f(X)

constant = tf.Variable(0, dtype=tf.float32)
total_hedge_term = Stat_term + Add_term + Mart_term + constant
total_price = Stat_price + Add_price + constant
diff = Fterm - total_hedge_term
density_optimizer = 2 * GAMMA * tf.nn.relu(diff)
running_primal_2 = tf.reduce_mean(density_optimizer * Fterm)
obj = total_price + GAMMA * tf.reduce_mean(tf.square(tf.nn.relu(diff)))
step_decrease = tf.Variable(0, trainable=False)
rate = tf.train.exponential_decay(0.01, step_decrease, 200, 0.98, staircase=True)
opt = tf.train.AdamOptimizer(learning_rate=rate, beta1=0.9, beta2=0.995).minimize(obj)

vals = []
pr_vals = []
theta_samp = theta(BATCH_SIZE)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=2)
with tf.Session(config=session_conf) as sess:
    sess.run(tf.global_variables_initializer())
    for t in range(1, N+N_FINE+1):
        samp = next(theta_samp)
        (rp, v, _) = sess.run([running_primal_2, obj, opt], feed_dict={X: samp, step_decrease: max(0, t-N)})
        vals.append(v)
        pr_vals.append(rp)
        if t % 5000 == 0:
            print(t)
            print('Dual value: ' + str(np.mean(vals[-5000:])))
            print('Primal value: ' + str(np.mean(pr_vals[-5000:])))  # the primal value is (a lot) more accurate.