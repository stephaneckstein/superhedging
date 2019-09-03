import numpy as np
import tensorflow as tf
from V1 import Hedging

T = 2
d = 2
f_type = 'basket'
p_spread = 2
K_basket = 0
MINMAX = 1
print('T = ' + str(T))
print('d = ' + str(d))
print('Objective type : ' + f_type)
print('Exponent of spread option type : ' + str(p_spread))
print('Strike of basket option type : ' + str(K_basket))
print('Minmax = ' + str(MINMAX))

CUSTOM_ID = str(T)+'_'+f_type+'_'+str(p_spread)+'_'+str(K_basket)+'_'+str(MINMAX)  # id for saving
SAVEIT = 0  # 1 if one wants to save trained model. Make sure 'SavedModels' directory exists in current directory

# Marginals: the j-th marginal at time i (i=0, ..., T-1) is U([-margs[i][j], margs[i][j]])

if f_type == 'spread':
    if T == 1:
        margs = np.array([[3, 2]])
    elif T == 2:
        margs = np.array([[1, 1], [3, 2]])
    elif T == 3:
        margs = np.array([[2.5, 1.6], [3, 2]])
    elif T == 4:
        margs = np.array([[1, 1], [1.6, 1.5], [2.5, 1.6], [3, 2]])

if f_type == 'basket':
    if T == 1:
        margs = np.array([[3, 3]])
    elif T == 2:
        margs = np.array([[1, 2], [3, 3]])
    elif T == 3:
        margs = np.array([[1, 2], [1.75, 2.1], [3, 3]])
    elif T == 4:
        margs = np.array([[1, 2], [1.75, 2.1], [2, 2.5], [3, 3]])

print('Marginal specification:')
print(margs)


def gen_prod(batch_size):
    while 1:
        yield (2 * np.random.random_sample([batch_size, T, d])-1) * margs


if f_type == 'spread':
    def f_obj(s):
        return MINMAX * tf.pow(tf.abs(s[:, T-1, 0] - s[:, T-1, 1]), p_spread)
elif f_type == 'basket':
    def f_obj(s):
        return MINMAX * tf.nn.relu(s[:, T-1, 0] + s[:, T-1, 1] - K_basket)
else:
    print('ERROR: Cost function type not implemented')
    exit()


# Current training spec is for T=2. For T=4, batch size has to be around 2**14 to work stably.
training_spec = {'batch_size_mu_0': 2 ** 9, 'batch_size_theta': 2 ** 11, 'n_train': 40000, 'n_fine': 20000,
                 'n_report': 1000}
print('Training spec:')
print(training_spec)
GAMMA = 2500 * T
print('Gamma = ' + str(GAMMA))

prob_hedge = Hedging(theta=gen_prod, mu_0=gen_prod, f=f_obj, prob_type='MMOT', t=T, d=d, gamma=GAMMA,
              training_spec=training_spec, add_id=CUSTOM_ID, saveit=SAVEIT)
print('Problem type: ' + prob_hedge.prob_type)
prob_hedge.build_graph()
prob_hedge.train_model()