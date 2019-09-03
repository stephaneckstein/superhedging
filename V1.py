import tensorflow as tf
from NNStructures import ff_net, penal, penal_der
import time
from ProbDist import *
import numpy as np
from datetime import datetime


class Hedging:
    def __init__(self, t=2, d=1, f=None, f_np=None, mu_0=None, theta=None, prob_type='MMOT', penal_type='L2',
                 gamma=500, updating=0, batch_size=2**10, hidden=((1, 32), (0, 32)), layers=((1, 4), (0, 5)),
                 features=((1, 'identity'),), add_instruments=None, training_spec=None, id=None, add_id=None,
                 densities=None, quantile_funs=None, misc_parameters=None, gradient_cap=None, activation='ReLu',
                 saveit=0):
        """
        Very work in progress set up to (simply) calculate various (super-)hedging problems via both linear programming
        or neural networks.
        :param t:
        :param d:
        :param f:
        :param f_np:
        :param mu_0:
        :param theta:
        :param prob_type:
        :param penal_type:
        :param gamma:
        :param updating:
        :param batch_size:
        :param hidden: (1, 32) means for 1 dimensional functions hidden dimension 32 is used. (0, X) is used for every-
        thing that isn't specified
        :param layers: Same as for hidden
        :param features:
        :param add_instruments:
        :param training_spec: dictionary that can contain entries like "learningrate: 0.001" etc
        :param id:
        :param add_id:
        :param densities:
        :param quantile_funs:
        :param misc_parameters:
        :param gradient_cap: positive value that all gradients get capped by (in norm)
        :param activation:
        :param saveit:
        """

        self.time_steps = t
        self.dimension = d
        self.f = f
        self.f_np = f_np
        self.mu_0 = mu_0
        self.theta = theta
        self.prob_type = prob_type
        self.penal_type = penal_type
        self.gamma = gamma
        self.updating = updating
        self.batch_size = batch_size
        self.hidden = hidden
        self.layers = layers
        self.features = features  # Possible TODO: Make it customizable where features are used
        self.add_instruments = add_instruments
        self.training_spec = training_spec
        self.misc_parameters = misc_parameters
        self.S_mu_0 = None
        self.S_theta = None
        self.objective_fun = None
        self.train_op = None
        self.saver = None
        self.step_decrease = None
        self.value_list = None
        self.samples = None
        self.mu0_hedge = None
        self.total_hedge = None
        self.densities = densities  # TODO: Decide which structure to use here (for markov martingale transport and/or
                                    # importance sampling; At the moment Lebesgue density for each marginal separately,
                                    # i.e. self.densities is a function with input (x, t, d). If all marginals are
                                    # instead defined on (the same) finite grid, the densities are with respect to the
                                    # uniform distribution on that grid.
        self.quantile_funs = quantile_funs  # list of size t of lists of size d with entries being functions
        self.den_var = None
        self.int_ctheta = None
        self.running_primal_1 = None
        self.running_primal_2 = None
        self.grad_lambda = None
        self.grad_theta = None
        self.lambda_variable = None
        self.rho = None
        self.tau_variable = None
        self.gradient_cap = gradient_cap
        self.activation = activation
        self.primal_1_value_list = None
        self.primal_2_value_list = None
        self.vals_met = None
        self.lam_vals = None
        self.saveit = saveit

        if id:
            self.identifier = id
        elif add_id:
            self.identifier = str(prob_type) + '_' + str(t) + '_' + str(d) + datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S") + '_' + add_id
        else:
            self.identifier = str(prob_type) + '_' + str(t) + '_' + str(d) + datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S")
        self.density_optimizer = None
        self.obj_eval = None

    def get_hidden_num(self, dim):
        h0 = 0
        for h in self.hidden:
            if h[0] == dim:
                return h[1]
            elif h[0] == 0:
                h0 = h[1]
        if h0 == 0:
            print('Potential ERROR: hidden dimension not specified and no general entry (0, X) is given')
            return 0
        return h0

    def get_layer_num(self, dim):
        h0 = 0
        for h in self.layers:
            if h[0] == dim:
                return h[1]
            elif h[0] == 0:
                h0 = h[1]
        if h0 == 0:
            print('Potential ERROR: layer number not specified and no general entry (0, X) is given')
            return 0
        return h0

    def get_parameter(self, param):
        if type(self.misc_parameters) is dict:
            if param in self.misc_parameters.keys():
                return self.misc_parameters[param]
        if param == 'rho_value':
            return 1
        if param == 'lam_fun':
            # determines which penalization of the wasserstein distance is used for "wasserstein ball"-type problems
            def phi_star(lam):
                return lam
            return phi_star
        if param == 'dist_metric':
            # determines which cost function to use for wasserstein distance
            def dist_metric(s1, s2):
                return tf.reduce_sum(tf.abs(s1-s2), axis=[1, 2])
            return dist_metric
        if param == 'time_split':
            # determines which time periods belong together, i.e. of which time periods the marginals are specified
            # jointly;
            # The standard case is that all time steps are separately specified.
            # In general the specification has to be a list of lists where the union of the inner lists has to be
            # equal to the list [0, 2, ..., time_steps-1]. E.g. [[0, 1, 2], [3, 4, 5], [6], [7], [8,
            # 9]] if time_steps = 10
            return [[i] for i in range(self.time_steps)]
        if param == 'dimension_split':
            # same as time_split, but for dimension
            return [[i] for i in range(self.dimension)]
        if param == 'avar_minmax':
            # 1 corresponds to max (over measures) and -1 corresponds to min
            return 1
        if param == 'avar_alpha':
            return 0.95

    def get_training_spec(self, param):
        if type(self.training_spec) is dict:
            if param in self.training_spec.keys():
                return self.training_spec[param]
        if param == 'rate' or param == 'learningrate':
            return 0.0001
        if param == 'beta1':
            return 0.99
        if param == 'beta2':
            return 0.995
        if param == 'n_train':
            return 20000 + 2500 * self.time_steps * self.dimension
        if param == 'n_fine':
            return 10000 + 1000 * self.time_steps * self.dimension
        if param == 'dec_interval':
            return 50
        if param == 'dec_rate':
            return 0.98
        if param == 'batch_size':
            return 2 ** 9
        if param == 'batch_size_mu_0':
            return 2 ** 11
        if param == 'batch_size_theta':
            return 2 ** 11
        if param == 'n_report':
            return 1000
        if param == 'markov_trading_cost':
            return 0
        if param == 'martingale_trading_cost':
            return 0
        if param == 'tau_init':
            return 1
        if param == 'n_tau':
            return 1000
        if param == 'n_tau_wait':
            return 5000
        if param == 'rate_tau':
            return 0.0001
        if param == 'lambda_init':
            return 0.5
        if param == 'n_lambda':
            return 250
        if param == 'n_lambda_wait':
            return 2500
        if param == 'rate_lambda':
            return 0.0005
        print('potential ERROR: training parameter not found or specified.')
        return 0

    def set_param(self, **kwargs):
        # TODO: perhaps check whether arguments are allowed
        self.__dict__.update(kwargs)

    def build_graph(self):
        if self.prob_type == 'MMOT':
            self.build_mmot()
        if self.prob_type == 'OT':
            self.build_ot()
        if self.prob_type == 'OT_multi':
            self.build_ot_multi()
        if self.prob_type == 'Markov':
            self.build_markov_mmot()
        if self.prob_type == 'Markov_new':
            self.build_markov_mmot_new()
        if self.prob_type == 'Markov_alone':
            self.build_markov_alone()
        if self.prob_type == 'Wasserstein_Marginals':
            self.build_wasserstein_marginals()
        if self.prob_type == 'Wasserstein_Marginals_AVAR':
            self.build_wasserstein_marginals_avar()

    def build_wasserstein_marginals_avar(self):
        # optimize over Wasserstein distance penalized and marginals specified, as in paper Eckstein, Kupper & Pohl.
        # For all intents and purposes, self.time_steps = 1, as there is no temporal structure in the problem
        # this is specifically to the average value at risk specification, as this leads to an inf-sup problem
        avar_minmax = self.get_parameter('avar_minmax')
        avar_alpha = self.get_parameter('avar_alpha')

        s_mu_0 = tf.placeholder(shape=[None, self.time_steps, self.dimension], dtype=tf.float32)
        s_theta_total = tf.placeholder(shape=[None, self.time_steps, self.dimension * 2], dtype=tf.float32)
        s_theta = s_theta_total[:, :, self.dimension:2*self.dimension]
        s_theta_fix = s_theta_total[:, :, 0:self.dimension]
        tau = tf.get_variable('tau_variable', shape=[1], initializer=tf.constant_initializer(
                self.get_training_spec('tau_init'), dtype=tf.float32))
        rho = tf.placeholder(dtype=tf.float32)  # Input for penalization dual. Radius if Wasserstein balls

        def fun_f(inp_x, inp_tau):
            return inp_tau + (1/(1-avar_alpha)) * tf.nn.relu(tf.reduce_sum(inp_x, axis=[1, 2])-inp_tau)
        self.f = fun_f

        lam_fun = self.get_parameter('lam_fun')
        if lam_fun == 'id':
            lambda_variable = 1/rho
        else:
            lambda_variable = tf.get_variable('lambda_variable', shape=[1], initializer=tf.constant_initializer(
                self.get_training_spec('lambda_init'), dtype=tf.float32))

        metric_d = self.get_parameter('dist_metric')

        s0 = 0
        s1 = 0

        t_list = self.get_parameter('time_split')
        d_list = self.get_parameter('dimension_split')

        for t_inner in t_list:
            for d_inner in d_list:
                t0 = t_inner[0]
                d0 = d_inner[0]
                input_dim = len(t_inner) * len(d_inner)
                input_0 = s_mu_0[:, t0, d0:d0+1]
                input_1 = s_theta[:, t0, d0:d0+1]
                for d0 in d_inner[1:]:
                    input_0 = tf.concat([input_0, s_mu_0[:, t0, d0:d0+1]], axis=1)
                    input_1 = tf.concat([input_1, s_theta[:, t0, d0:d0+1]], axis=1)
                for t0 in t_inner[1:]:
                    for d0 in d_inner[1:]:
                        input_0 = tf.concat([input_0, s_mu_0[:, t0, d0:d0 + 1]], axis=1)
                        input_1 = tf.concat([input_1, s_theta[:, t0, d0:d0 + 1]], axis=1)

                s0 += tf.reduce_sum(ff_net(input_0, 'stat_'+str(t_inner[0])+'_'+str(d_inner[0]), feats=self.features,
                                           n_layers=self.get_layer_num(input_dim), hidden_dim=self.get_hidden_num(
                        input_dim), input_dim=input_dim, output_dim=1, activation=self.activation), axis=1)
                s1 += tf.reduce_sum(ff_net(input_1, 'stat_'+str(t_inner[0])+'_'+str(d_inner[0]), feats=self.features,
                                           n_layers=self.get_layer_num(input_dim), hidden_dim=self.get_hidden_num(
                        input_dim), input_dim=input_dim, output_dim=1, activation=self.activation), axis=1)

        total_input_0 = s_mu_0[:, 0, :]
        total_input_1 = s_theta_fix[:, 0, :]
        for i in range(1, self.time_steps):
            total_input_0 = tf.concat([total_input_0, s_mu_0[:, i, :]], axis=1)
            total_input_1 = tf.concat([total_input_1, s_theta_fix[:, i, :]], axis=1)
        inp_dim = self.time_steps * self.dimension
        s0 += tf.reduce_sum(ff_net(total_input_0, 'g_fun', feats=self.features, n_layers=self.get_layer_num(inp_dim),
                                   hidden_dim=self.get_hidden_num(inp_dim), input_dim=inp_dim, output_dim=1,
                                   activation=self.activation), axis=1)
        s1 += tf.reduce_sum(ff_net(total_input_1, 'g_fun', feats=self.features, n_layers=self.get_layer_num(inp_dim),
                                   hidden_dim=self.get_hidden_num(inp_dim), input_dim=inp_dim, output_dim=1,
                                   activation=self.activation), axis=1)

        s1_pre = s1
        s1 += lambda_variable * metric_d(s_theta_fix, s_theta)
        f_var = self.f(s_theta, tau)

        if lam_fun == 'id':
            lamrho = 0
        else:
            lamrho = rho * lam_fun(lambda_variable)

        s1_pre = s1_pre + lamrho
        ints_pre = s0 + lamrho
        ints = tf.reduce_mean(ints_pre)
        diff = f_var - s1
        penal_term = penal(diff, self.gamma, type=self.penal_type)
        density_optimizer = penal_der(diff, self.gamma, type=self.penal_type)
        ctheta = density_optimizer * metric_d(s_theta_fix, s_theta)
        int_ctheta = tf.reduce_mean(ctheta)
        running_primal_2 = tf.reduce_mean(density_optimizer * f_var)
        criteria_term = ints - tf.reduce_mean(density_optimizer * s1_pre)
        running_primal_1 = criteria_term + running_primal_2

        step_decrease = tf.Variable(0, trainable=False)
        objective_fun = ints + tf.reduce_mean(penal_term) + 10 * tf.nn.relu(-lambda_variable)
        rate = tf.train.exponential_decay(self.get_training_spec('rate'), step_decrease,
                                          self.get_training_spec('dec_interval'), self.get_training_spec('dec_rate'),
                                          staircase=False)
        var_list1 = [v for v in tf.global_variables() if (v.name != 'lambda_variable' and v.name !=
                                                          'lambda_variable:0' and v.name != 'tau_variable' and v.name
                                                          != 'tau_variable:0')]
        var_list2 = [v for v in tf.global_variables() if (v.name == 'lambda_variable' or v.name == 'lambda_variable:0')]
        var_list3 = [v for v in tf.global_variables() if (v.name == 'tau_variable' or v.name == 'tau_variable:0')]

        grads = tf.gradients(objective_fun, var_list1 + var_list2 + var_list3)
        grads1 = grads[:len(var_list1)]
        gradlambda = grads[len(var_list1):len(var_list1) + len(var_list2)]
        gradtheta = grads[len(var_list1) + len(var_list2):len(var_list1) + len(var_list2) + len(var_list3)]

        opt = tf.train.AdamOptimizer(learning_rate=rate, beta1=
                        self.get_training_spec('beta1'), beta2=self.get_training_spec('beta2'))
        if avar_minmax == 1:
            train_op = opt.apply_gradients(zip(grads1, var_list1))
        else:
            train_op = opt.apply_gradients(zip(-grads1, var_list1))

        self.tau_variable = tau
        self.rho = rho
        self.lambda_variable = lambda_variable
        self.grad_lambda = gradlambda
        self.grad_theta = gradtheta
        self.int_ctheta = int_ctheta
        self.running_primal_1 = running_primal_1
        self.running_primal_2 = running_primal_2
        self.mu0_hedge = s0
        self.total_hedge = s1
        self.obj_eval = f_var
        self.density_optimizer = density_optimizer
        self.S_mu_0 = s_mu_0
        self.S_theta = s_theta_total
        self.objective_fun = objective_fun
        self.train_op = train_op
        self.step_decrease = step_decrease
        saver = tf.train.Saver()
        self.saver = saver

        return 0

    def build_wasserstein_marginals(self):
        # optimize over Wasserstein distance penalized and marginals specified, as in paper Eckstein, Kupper & Pohl.
        # For all intents and purposes, self.time_steps = 1, as there is no temporal structure in the problem
        s_mu_0 = tf.placeholder(shape=[None, self.time_steps, self.dimension], dtype=tf.float32)
        s_theta_total = tf.placeholder(shape=[None, self.time_steps, self.dimension * 2], dtype=tf.float32)
        s_theta = s_theta_total[:, :, self.dimension:2*self.dimension]
        s_theta_fix = s_theta_total[:, :, 0:self.dimension]
        rho = tf.placeholder(dtype=tf.float32)  # Input for penalization dual. Radius if Wasserstein balls

        lam_fun = self.get_parameter('lam_fun')
        if lam_fun == 'id':
            lambda_variable = 1/rho
        else:
            lambda_variable = tf.get_variable('lambda_variable', shape=[1], initializer=tf.constant_initializer(
                self.get_training_spec('lambda_init'), dtype=tf.float32))

        metric_d = self.get_parameter('dist_metric')

        s0 = 0
        s1 = 0

        t_list = self.get_parameter('time_split')
        d_list = self.get_parameter('dimension_split')

        for t_inner in t_list:
            for d_inner in d_list:
                t0 = t_inner[0]
                d0 = d_inner[0]
                input_dim = len(t_inner) * len(d_inner)
                input_0 = s_mu_0[:, t0, d0:d0+1]
                input_1 = s_theta[:, t0, d0:d0+1]
                for d0 in d_inner[1:]:
                    input_0 = tf.concat([input_0, s_mu_0[:, t0, d0:d0+1]], axis=1)
                    input_1 = tf.concat([input_1, s_theta[:, t0, d0:d0+1]], axis=1)
                for t0 in t_inner[1:]:
                    for d0 in d_inner[1:]:
                        input_0 = tf.concat([input_0, s_mu_0[:, t0, d0:d0 + 1]], axis=1)
                        input_1 = tf.concat([input_1, s_theta[:, t0, d0:d0 + 1]], axis=1)

                s0 += tf.reduce_sum(ff_net(input_0, 'stat_'+str(t_inner[0])+'_'+str(d_inner[0]), feats=self.features,
                                           n_layers=self.get_layer_num(input_dim), hidden_dim=self.get_hidden_num(
                        input_dim), input_dim=input_dim, output_dim=1, activation=self.activation), axis=1)
                s1 += tf.reduce_sum(ff_net(input_1, 'stat_'+str(t_inner[0])+'_'+str(d_inner[0]), feats=self.features,
                                           n_layers=self.get_layer_num(input_dim), hidden_dim=self.get_hidden_num(
                        input_dim), input_dim=input_dim, output_dim=1, activation=self.activation), axis=1)

        total_input_0 = s_mu_0[:, 0, :]
        total_input_1 = s_theta_fix[:, 0, :]
        for i in range(1, self.time_steps):
            total_input_0 = tf.concat([total_input_0, s_mu_0[:, i, :]], axis=1)
            total_input_1 = tf.concat([total_input_1, s_theta_fix[:, i, :]], axis=1)
        inp_dim = self.time_steps * self.dimension
        s0 += tf.reduce_sum(ff_net(total_input_0, 'g_fun', feats=self.features, n_layers=self.get_layer_num(inp_dim),
                                   hidden_dim=self.get_hidden_num(inp_dim), input_dim=inp_dim, output_dim=1,
                                   activation=self.activation), axis=1)
        s1 += tf.reduce_sum(ff_net(total_input_1, 'g_fun', feats=self.features, n_layers=self.get_layer_num(inp_dim),
                                   hidden_dim=self.get_hidden_num(inp_dim), input_dim=inp_dim, output_dim=1,
                                   activation=self.activation), axis=1)

        s1_pre = s1
        s1 += lambda_variable * metric_d(s_theta_fix, s_theta)
        f_var = self.f(s_theta)

        if lam_fun == 'id':
            lamrho = 0
        else:
            lamrho = rho * lam_fun(lambda_variable)

        s1_pre = s1_pre + lamrho
        ints_pre = s0 + lamrho
        ints = tf.reduce_mean(ints_pre)
        diff = f_var - s1
        penal_term = penal(diff, self.gamma, type=self.penal_type)
        density_optimizer = penal_der(diff, self.gamma, type=self.penal_type)
        ctheta = density_optimizer * metric_d(s_theta_fix, s_theta)
        int_ctheta = tf.reduce_mean(ctheta)
        running_primal_2 = tf.reduce_mean(density_optimizer * f_var)
        criteria_term = ints - tf.reduce_mean(density_optimizer * s1_pre)
        running_primal_1 = criteria_term + running_primal_2

        step_decrease = tf.Variable(0, trainable=False)
        objective_fun = ints + tf.reduce_mean(penal_term) + 0.01 * tf.nn.relu(-lambda_variable)
        rate = tf.train.exponential_decay(self.get_training_spec('rate'), step_decrease,
                                          self.get_training_spec('dec_interval'), self.get_training_spec('dec_rate'),
                                          staircase=False)
        var_list1 = [v for v in tf.global_variables() if (v.name != 'lambda_variable' and v.name != 'lambda_variable:0')]
        var_list2 = [v for v in tf.global_variables() if (v.name == 'lambda_variable' or v.name == 'lambda_variable:0')]

        grads = tf.gradients(objective_fun, var_list1 + var_list2)

        if self.gradient_cap:
            grads = [tf.clip_by_value(grad, -self.gradient_cap, self.gradient_cap) if (grad is not None) else grad for
                     grad in grads]

        grads1 = grads[:len(var_list1)]
        gradlambda = grads[len(var_list1):len(var_list1) + len(var_list2)]

        opt = tf.train.AdamOptimizer(learning_rate=rate, beta1=
                        self.get_training_spec('beta1'), beta2=self.get_training_spec('beta2'))
        train_op = opt.apply_gradients(zip(grads1, var_list1))

        self.rho = rho
        self.lambda_variable = lambda_variable
        self.grad_lambda = gradlambda
        self.int_ctheta = int_ctheta
        self.running_primal_1 = running_primal_1
        self.running_primal_2 = running_primal_2
        self.mu0_hedge = s0
        self.total_hedge = s1
        self.obj_eval = f_var
        self.density_optimizer = density_optimizer
        self.S_mu_0 = s_mu_0
        self.S_theta = s_theta_total
        self.objective_fun = objective_fun
        self.train_op = train_op
        self.step_decrease = step_decrease
        saver = tf.train.Saver()
        self.saver = saver

        return 0

    def build_ot_multi(self):
        s_mu_0 = tf.placeholder(shape=[None, self.time_steps, self.dimension], dtype=tf.float32)
        s_theta = tf.placeholder(shape=[None, self.time_steps, self.dimension], dtype=tf.float32)
        s0 = 0
        s1 = 0
        for t in range(self.time_steps):
            s0 += tf.reduce_sum(ff_net(s_mu_0[:, t, :], 'stat_'+str(t), feats=self.features,
                         n_layers=self.get_layer_num(self.dimension), hidden_dim=self.get_hidden_num(self.dimension),
                                       input_dim=self.dimension, output_dim=1, activation=self.activation), axis=1)
            s1 += tf.reduce_sum(ff_net(s_theta[:, t, :], 'stat_'+str(t), feats=self.features,
                         n_layers=self.get_layer_num(self.dimension), hidden_dim=self.get_hidden_num(self.dimension),
                                       input_dim=self.dimension, output_dim=1, activation=self.activation), axis=1)

        all_ind_too = 1
        if all_ind_too == 1:
            for t in range(self.time_steps):
                for i in range(self.dimension):
                    s0 += tf.reduce_sum(ff_net(s_mu_0[:, t, i:i+1], 'stat_'+str(t)+'_'+str(i), feats=self.features,
                                 n_layers=self.get_layer_num(1), hidden_dim=self.get_hidden_num(1), input_dim=1,
                                 output_dim=1, activation=self.activation), axis=1)
                    s1 += tf.reduce_sum(ff_net(s_theta[:, t, i:i+1], 'stat_'+str(t)+'_'+str(i), feats=self.features,
                                 n_layers=self.get_layer_num(1), hidden_dim=self.get_hidden_num(1), input_dim=1,
                                 output_dim=1, activation=self.activation), axis=1)

        f_var = self.f(s_theta)
        diff = f_var - s1
        ints = tf.reduce_mean(s0)
        density_optimizer = penal_der(diff, self.gamma, type=self.penal_type)
        running_primal_2 = tf.reduce_mean(density_optimizer * f_var)
        criteria_term = ints - tf.reduce_mean(density_optimizer * s1)
        running_primal_1 = criteria_term + running_primal_2

        step_decrease = tf.Variable(0, trainable=False)
        objective_fun = ints + tf.reduce_mean(penal(diff, self.gamma, type=self.penal_type))
        rate = tf.train.exponential_decay(self.get_training_spec('rate'), step_decrease,
                                          self.get_training_spec('dec_interval'), self.get_training_spec('dec_rate'),
                                          staircase=False)

        train_op = tf.train.AdamOptimizer(learning_rate=rate, beta1=
        self.get_training_spec('beta1'), beta2=self.get_training_spec('beta2')).minimize(objective_fun)

        obj_eval = self.f(s_theta)
        self.running_primal_1 = running_primal_1
        self.running_primal_2 = running_primal_2
        self.obj_eval = obj_eval
        self.mu0_hedge = s0
        self.total_hedge = s1
        saver = tf.train.Saver()
        self.saver = saver
        self.density_optimizer = density_optimizer
        self.S_mu_0 = s_mu_0
        self.S_theta = s_theta
        self.objective_fun = objective_fun
        self.train_op = train_op
        self.step_decrease = step_decrease

    def build_ot(self):
        s_mu_0 = tf.placeholder(shape=[None, self.time_steps, self.dimension], dtype=tf.float32)
        s_theta = tf.placeholder(shape=[None, self.time_steps, self.dimension], dtype=tf.float32)
        s0 = 0
        s1 = 0
        for t in range(self.time_steps):
            for i in range(self.dimension):
                s0 += tf.reduce_sum(ff_net(s_mu_0[:, t, i:i+1], 'stat_'+str(t)+'_'+str(i), feats=self.features,
                             n_layers=self.get_layer_num(1), hidden_dim=self.get_hidden_num(1), input_dim=1,
                             output_dim=1, activation=self.activation), axis=1)
                s1 += tf.reduce_sum(ff_net(s_theta[:, t, i:i+1], 'stat_'+str(t)+'_'+str(i), feats=self.features,
                             n_layers=self.get_layer_num(1), hidden_dim=self.get_hidden_num(1), input_dim=1,
                             output_dim=1, activation=self.activation), axis=1)

        # diff = self.f(s_theta) - s1
        # density_optimizer = penal_der(diff, self.gamma, type=self.penal_type)
        # step_decrease = tf.Variable(0, trainable=False)
        # objective_fun = tf.reduce_mean(s0) + tf.reduce_mean(penal(diff, self.gamma, type=self.penal_type))
        # rate = tf.train.exponential_decay(self.get_training_spec('rate'), step_decrease,
        #                                   self.get_training_spec('dec_interval'), self.get_training_spec('dec_rate'),
        #                                   staircase=False)
        #
        # train_op = tf.train.AdamOptimizer(learning_rate=rate, beta1=
        #                 self.get_training_spec('beta1'), beta2=self.get_training_spec('beta2')).minimize(objective_fun)
        # saver = tf.train.Saver()
        # obj_eval = self.f(s_theta)
        f_var = self.f(s_theta)
        diff = f_var - s1
        ints = tf.reduce_mean(s0)
        density_optimizer = penal_der(diff, self.gamma, type=self.penal_type)
        running_primal_2 = tf.reduce_mean(density_optimizer * f_var)
        criteria_term = ints - tf.reduce_mean(density_optimizer * s1)
        running_primal_1 = criteria_term + running_primal_2

        step_decrease = tf.Variable(0, trainable=False)
        objective_fun = ints + tf.reduce_mean(penal(diff, self.gamma, type=self.penal_type))
        rate = tf.train.exponential_decay(self.get_training_spec('rate'), step_decrease,
                                          self.get_training_spec('dec_interval'), self.get_training_spec('dec_rate'),
                                          staircase=False)

        train_op = tf.train.AdamOptimizer(learning_rate=rate, beta1=
        self.get_training_spec('beta1'), beta2=self.get_training_spec('beta2')).minimize(objective_fun)

        obj_eval = self.f(s_theta)
        saver = tf.train.Saver()
        self.running_primal_1 = running_primal_1
        self.running_primal_2 = running_primal_2
        self.obj_eval = obj_eval
        self.mu0_hedge = s0
        self.total_hedge = s1
        self.saver = saver
        self.density_optimizer = density_optimizer
        self.S_mu_0 = s_mu_0
        self.S_theta = s_theta
        self.objective_fun = objective_fun
        self.train_op = train_op
        self.step_decrease = step_decrease

    def build_mmot(self):
        # TODO: Check dimensions matching and make sure that f, mu_0 and theta are supplied
        s_mu_0 = tf.placeholder(shape=[None, self.time_steps, self.dimension], dtype=tf.float32)
        s_theta = tf.placeholder(shape=[None, self.time_steps, self.dimension], dtype=tf.float32)
        s0 = 0
        s1 = 0
        for t in range(self.time_steps):
            for i in range(self.dimension):
                s0 += tf.reduce_sum(ff_net(s_mu_0[:, t, i:i+1], 'stat_'+str(t)+'_'+str(i), feats=self.features,
                             n_layers=self.get_layer_num(1), hidden_dim=self.get_hidden_num(1), input_dim=1,
                             output_dim=1, activation=self.activation), axis=1)
                s1 += tf.reduce_sum(ff_net(s_theta[:, t, i:i+1], 'stat_'+str(t)+'_'+str(i), feats=self.features,
                             n_layers=self.get_layer_num(1), hidden_dim=self.get_hidden_num(1), input_dim=1,
                             output_dim=1, activation=self.activation), axis=1)

        trade_cost_mart = self.get_training_spec('martingale_trading_cost')
        for t1 in range(self.time_steps-1):
            for t2 in range(t1+1, self.time_steps):
                for i in range(self.dimension):
                    strategy = tf.reduce_sum(ff_net(
                        tf.reshape(s_theta[:, 0:(t1+1), :],
                                [tf.shape(s_theta)[0], (t1+1)*self.dimension]), 'dyn_'+str(t1)+'_'+str(t2)+'_'+str(i),
                                 feats=self.features, n_layers=self.get_layer_num((t1+1)*self.dimension), hidden_dim=
                                 self.get_hidden_num((t1+1)*self.dimension),
                                 input_dim=(t1+1)*self.dimension, output_dim=1), axis=1)
                    s1 += (s_theta[:, t2, i] - s_theta[:, t1, i]) * strategy
                    if trade_cost_mart > 0:
                        s1 -= trade_cost_mart * tf.square(strategy)

        f_var = self.f(s_theta)
        diff = f_var - s1
        ints = tf.reduce_mean(s0)
        density_optimizer = penal_der(diff, self.gamma, type=self.penal_type)
        running_primal_2 = tf.reduce_mean(density_optimizer * f_var)
        criteria_term = ints - tf.reduce_mean(density_optimizer * s1)
        running_primal_1 = criteria_term + running_primal_2

        step_decrease = tf.Variable(0, trainable=False)
        objective_fun = ints + tf.reduce_mean(penal(diff, self.gamma, type=self.penal_type))
        rate = tf.train.exponential_decay(self.get_training_spec('rate'), step_decrease,
                                          self.get_training_spec('dec_interval'), self.get_training_spec('dec_rate'),
                                          staircase=False)

        train_op = tf.train.AdamOptimizer(learning_rate=rate, beta1=
        self.get_training_spec('beta1'), beta2=self.get_training_spec('beta2')).minimize(objective_fun)

        obj_eval = self.f(s_theta)
        self.running_primal_1 = running_primal_1
        self.running_primal_2 = running_primal_2
        self.mu0_hedge = s0
        self.total_hedge = s1
        self.obj_eval = obj_eval
        self.density_optimizer = density_optimizer
        self.S_mu_0 = s_mu_0
        self.S_theta = s_theta
        self.objective_fun = objective_fun
        self.train_op = train_op
        self.step_decrease = step_decrease
        saver = tf.train.Saver()
        self.saver = saver

    def markov_gen_theta(self, batch_size):
        point_gen = self.theta(batch_size)
        while 1:
            points = next(point_gen)
            den_vals = np.zeros([batch_size, self.dimension, self.time_steps - 2, self.time_steps - 2])
            for d in range(self.dimension):
                for inc_step in range(1, self.time_steps - 1):
                    for t in range(self.time_steps - 2 * inc_step):
                        den_vals[:, d, inc_step-1, t] = self.densities(points[:, t+inc_step, d], t,
                                                                     d) / self.densities(points[:, t+inc_step, d],
                                                                                         t+inc_step, d)
            yield (points, den_vals)

    def markov_gen_theta_new(self, batch_size):
        point_gen = self.theta(batch_size)
        while 1:
            points = next(point_gen)
            den_vals = np.zeros([batch_size, self.dimension, self.time_steps - 2, self.time_steps - 2, 2])
            for d in range(self.dimension):
                for s in range(self.time_steps-2):
                    for t in range(s+1, self.time_steps-1):
                        den_vals[:, d, s, t-1, 0] = self.densities(points[:, s, d], s, t, d, 0)
                        den_vals[:, d, s, t-1, 1] = self.densities(points[:, t, d], s, t, d, 1)
            yield (points, den_vals)

    def build_markov_mmot(self):
        # TODO: Check dimensions matching and make sure that f, mu_0 and theta are supplied
        if self.densities is None:
            print('Potential ERROR: Densities are not supplied in Markov MMOT model')
            return 0

        s_mu_0 = tf.placeholder(shape=[None, self.time_steps, self.dimension], dtype=tf.float32)
        s_theta = tf.placeholder(shape=[None, self.time_steps, self.dimension], dtype=tf.float32)
        den_var = tf.placeholder(shape=[None, self.dimension, self.time_steps-2, self.time_steps-2], dtype=tf.float32)
        s0 = 0
        s1 = 0
        for t in range(self.time_steps):
            for i in range(self.dimension):
                s0 += tf.reduce_sum(ff_net(s_mu_0[:, t, i:i+1], 'stat_'+str(t)+'_'+str(i), feats=self.features,
                             n_layers=self.get_layer_num(1), hidden_dim=self.get_hidden_num(1), input_dim=1,
                             output_dim=1, activation=self.activation), axis=1)
                s1 += tf.reduce_sum(ff_net(s_theta[:, t, i:i+1], 'stat_'+str(t)+'_'+str(i), feats=self.features,
                             n_layers=self.get_layer_num(1), hidden_dim=self.get_hidden_num(1), input_dim=1,
                             output_dim=1, activation=self.activation), axis=1)

        trade_cost_mart = self.get_training_spec('martingale_trading_cost')
        for t1 in range(self.time_steps-1):
            for t2 in range(t1+1, self.time_steps):
                for i in range(self.dimension):
                    strategy = tf.reduce_sum(ff_net(
                        tf.reshape(s_theta[:, 0:(t1+1), :],
                                [tf.shape(s_theta)[0], (t1+1)*self.dimension]), 'dyn_'+str(t1)+'_'+str(t2)+'_'+str(i),
                                 feats=self.features, n_layers=self.get_layer_num((t1+1)*self.dimension), hidden_dim=
                                 self.get_hidden_num((t1+1)*self.dimension),
                                 input_dim=(t1+1)*self.dimension, output_dim=1), axis=1)
                    s1 += (s_theta[:, t2, i] - s_theta[:, t1, i]) * strategy
                    if trade_cost_mart > 0:
                        s1 -= trade_cost_mart * tf.square(strategy)

        # NOTE: Here, since we are in an MMOT setting, Markovian trading is for each dimension separately!
        # Trading cost / transaction costs are very specific at the moment. TODO: Implement more generally
        trade_cost = self.get_training_spec('markov_trading_cost')
        for d in range(self.dimension):
            for inc_step in range(1, self.time_steps - 1):
                for t in range(self.time_steps - 2 * inc_step):
                    input_x = s_theta[:, t:t+1, d]
                    input_x = tf.concat([input_x, s_theta[:, t+inc_step:t+inc_step+1, d]], axis=1)
                    t1 = tf.reduce_sum(ff_net(input_x, 'markov_'+str(inc_step)+'_'+str(t), feats=self.features,
                                 n_layers=self.get_layer_num(2), hidden_dim=self.get_hidden_num(2),
                                 input_dim=2, output_dim=1), axis=1)
                    input_y = s_theta[:, t+inc_step:t+inc_step+1, d]
                    input_y = tf.concat([input_y, s_theta[:, t+2*inc_step:t+2*inc_step+1, d]], axis=1)
                    t2 = tf.reduce_sum(ff_net(input_y, 'markov_'+str(inc_step)+'_'+str(t), feats=self.features,
                                 n_layers=self.get_layer_num(2), hidden_dim=self.get_hidden_num(2),
                                 input_dim=2, output_dim=1), axis=1)
                    s1 += t1 - den_var[:, d, inc_step-1, t] * t2
                    if trade_cost > 0:
                        s1 -= trade_cost * tf.square(t1)

        f_var = self.f(s_theta)
        diff = f_var - s1
        ints = tf.reduce_mean(s0)
        density_optimizer = penal_der(diff, self.gamma, type=self.penal_type)
        running_primal_2 = tf.reduce_mean(density_optimizer * f_var)
        criteria_term = ints - tf.reduce_mean(density_optimizer * s1)
        running_primal_1 = criteria_term + running_primal_2

        step_decrease = tf.Variable(0, trainable=False)
        objective_fun = ints + tf.reduce_mean(penal(diff, self.gamma, type=self.penal_type))
        rate = tf.train.exponential_decay(self.get_training_spec('rate'), step_decrease,
                                          self.get_training_spec('dec_interval'), self.get_training_spec('dec_rate'),
                                          staircase=False)

        train_op = tf.train.AdamOptimizer(learning_rate=rate, beta1=
                        self.get_training_spec('beta1'), beta2=self.get_training_spec('beta2')).minimize(objective_fun)

        obj_eval = self.f(s_theta)
        self.running_primal_1 = running_primal_1
        self.running_primal_2 = running_primal_2
        self.mu0_hedge = s0
        self.total_hedge = s1
        self.obj_eval = obj_eval
        self.density_optimizer = density_optimizer
        self.S_mu_0 = s_mu_0
        self.S_theta = s_theta
        self.objective_fun = objective_fun
        self.train_op = train_op
        self.step_decrease = step_decrease
        self.den_var = den_var
        saver = tf.train.Saver()
        self.saver = saver

    def build_markov_mmot_new(self):
        # TODO: Check dimensions matching and make sure that f, mu_0 and theta are supplied
        if self.densities is None:
            print('Potential ERROR: Densities are not supplied in Markov MMOT model')
            return 0

        s_mu_0 = tf.placeholder(shape=[None, self.time_steps, self.dimension], dtype=tf.float32)
        s_theta = tf.placeholder(shape=[None, self.time_steps, self.dimension], dtype=tf.float32)
        den_var = tf.placeholder(shape=[None, self.dimension, self.time_steps-2, self.time_steps-2, 2],
                                 dtype=tf.float32)

        s0 = 0
        s1 = 0
        for t in range(self.time_steps):
            for i in range(self.dimension):
                s0 += tf.reduce_sum(ff_net(s_mu_0[:, t, i:i+1], 'stat_'+str(t)+'_'+str(i), feats=self.features,
                             n_layers=self.get_layer_num(1), hidden_dim=self.get_hidden_num(1), input_dim=1,
                             output_dim=1, activation=self.activation), axis=1)
                s1 += tf.reduce_sum(ff_net(s_theta[:, t, i:i+1], 'stat_'+str(t)+'_'+str(i), feats=self.features,
                             n_layers=self.get_layer_num(1), hidden_dim=self.get_hidden_num(1), input_dim=1,
                             output_dim=1, activation=self.activation), axis=1)

        trade_cost_mart = self.get_training_spec('martingale_trading_cost')
        for t1 in range(self.time_steps-1):
            for t2 in range(t1+1, self.time_steps):
                for i in range(self.dimension):
                    strategy = tf.reduce_sum(ff_net(
                        tf.reshape(s_theta[:, 0:(t1+1), :],
                                [tf.shape(s_theta)[0], (t1+1)*self.dimension]), 'dyn_'+str(t1)+'_'+str(t2)+'_'+str(i),
                                 feats=self.features, n_layers=self.get_layer_num((t1+1)*self.dimension), hidden_dim=
                                 self.get_hidden_num((t1+1)*self.dimension),
                                 input_dim=(t1+1)*self.dimension, output_dim=1), axis=1)
                    s1 += (s_theta[:, t2, i] - s_theta[:, t1, i]) * strategy
                    if trade_cost_mart > 0:
                        s1 -= trade_cost_mart * tf.square(strategy)

        # NOTE: Here, since we are in an MMOT setting, Markovian trading is for each dimension separately!
        trade_cost = self.get_training_spec('markov_trading_cost')
        for d in range(self.dimension):
            for s in range(self.time_steps - 2):
                for t in range(s+1, self.time_steps - 1):
                    for tau in range(1, self.time_steps - t):
                        input_x = s_theta[:, s:s+1, d]
                        input_x = tf.concat([input_x, s_theta[:, s+tau:s+tau+1, d]], axis=1)
                        t1 = tf.reduce_sum(ff_net(input_x, 'markov_' + str(tau) + '_' + str(t) + '_'+str(s),
                                                  feats=self.features, n_layers=self.get_layer_num(2),
                                                  hidden_dim=self.get_hidden_num(2), input_dim=2, output_dim=1), axis=1)
                        input_y = s_theta[:, t:t+1, d]
                        input_y = tf.concat([input_y, s_theta[:, t+tau:t+tau+1, d]], axis=1)
                        t2 = tf.reduce_sum(ff_net(input_y, 'markov_' + str(tau) + '_' + str(t) + '_'+str(s),
                                                  feats=self.features, n_layers=self.get_layer_num(2),
                                                  hidden_dim=self.get_hidden_num(2), input_dim=2, output_dim=1), axis=1)
                        s1 += den_var[:, d, s, t-1, 0] * t1 - den_var[:, d, s, t-1, 1] * t2
                        if trade_cost > 0:
                            s1 -= trade_cost * (tf.square(t1)+tf.square(t2))

        f_var = self.f(s_theta)
        diff = f_var - s1
        ints = tf.reduce_mean(s0)
        density_optimizer = penal_der(diff, self.gamma, type=self.penal_type)
        running_primal_2 = tf.reduce_mean(density_optimizer * f_var)
        criteria_term = ints - tf.reduce_mean(density_optimizer * s1)
        running_primal_1 = criteria_term + running_primal_2

        step_decrease = tf.Variable(0, trainable=False)
        objective_fun = ints + tf.reduce_mean(penal(diff, self.gamma, type=self.penal_type))
        rate = tf.train.exponential_decay(self.get_training_spec('rate'), step_decrease,
                                          self.get_training_spec('dec_interval'), self.get_training_spec('dec_rate'),
                                          staircase=False)

        train_op = tf.train.AdamOptimizer(learning_rate=rate, beta1=
        self.get_training_spec('beta1'), beta2=self.get_training_spec('beta2')).minimize(objective_fun)

        obj_eval = self.f(s_theta)
        self.running_primal_1 = running_primal_1
        self.running_primal_2 = running_primal_2
        self.mu0_hedge = s0
        self.total_hedge = s1
        self.obj_eval = obj_eval
        self.density_optimizer = density_optimizer
        self.S_mu_0 = s_mu_0
        self.S_theta = s_theta
        self.objective_fun = objective_fun
        self.train_op = train_op
        self.step_decrease = step_decrease
        self.den_var = den_var
        saver = tf.train.Saver()
        self.saver = saver

    def build_markov_alone(self):
        # TODO: Check dimensions matching and make sure that f, mu_0 and theta are supplied
        if self.densities is None:
            print('Potential ERROR: Densities are not supplied in Markov MMOT model')
            return 0

        s_mu_0 = tf.placeholder(shape=[None, self.time_steps, self.dimension], dtype=tf.float32)
        s_theta = tf.placeholder(shape=[None, self.time_steps, self.dimension], dtype=tf.float32)
        den_var = tf.placeholder(shape=[None, self.dimension, self.time_steps-2, self.time_steps-2], dtype=tf.float32)
        s0 = 0
        s1 = 0
        for t in range(self.time_steps):
            for i in range(self.dimension):
                s0 += tf.reduce_sum(ff_net(s_mu_0[:, t, i:i+1], 'stat_'+str(t)+'_'+str(i), feats=self.features,
                             n_layers=self.get_layer_num(1), hidden_dim=self.get_hidden_num(1), input_dim=1,
                             output_dim=1, activation=self.activation), axis=1)
                s1 += tf.reduce_sum(ff_net(s_theta[:, t, i:i+1], 'stat_'+str(t)+'_'+str(i), feats=self.features,
                             n_layers=self.get_layer_num(1), hidden_dim=self.get_hidden_num(1), input_dim=1,
                             output_dim=1, activation=self.activation), axis=1)

        # NOTE: Here, since we are in an MMOT setting, Markovian trading is for each dimension separately!
        trade_cost = self.get_training_spec('markov_trading_cost')
        for d in range(self.dimension):
            for inc_step in range(1, self.time_steps - 1):
                for t in range(self.time_steps - 2 * inc_step):
                    input_x = s_theta[:, t:t+1, d]
                    input_x = tf.concat([input_x, s_theta[:, t+inc_step:t+inc_step+1, d]], axis=1)
                    t1 = tf.reduce_sum(ff_net(input_x, 'markov_'+str(inc_step)+'_'+str(t), feats=self.features,
                                 n_layers=self.get_layer_num(2), hidden_dim=self.get_hidden_num(2),
                                 input_dim=2, output_dim=1), axis=1)
                    input_y = s_theta[:, t+inc_step:t+inc_step+1, d]
                    input_y = tf.concat([input_y, s_theta[:, t+2*inc_step:t+2*inc_step+1, d]], axis=1)
                    t2 = tf.reduce_sum(ff_net(input_y, 'markov_'+str(inc_step)+'_'+str(t), feats=self.features,
                                 n_layers=self.get_layer_num(2), hidden_dim=self.get_hidden_num(2),
                                 input_dim=2, output_dim=1), axis=1)
                    s1 += t1 - den_var[:, d, inc_step-1, t] * t2
                    if trade_cost > 0:
                        s1 -= trade_cost * tf.square(t1)

        f_var = self.f(s_theta)
        diff = f_var - s1
        ints = tf.reduce_mean(s0)
        density_optimizer = penal_der(diff, self.gamma, type=self.penal_type)
        running_primal_2 = tf.reduce_mean(density_optimizer * f_var)
        criteria_term = ints - tf.reduce_mean(density_optimizer * s1)
        running_primal_1 = criteria_term + running_primal_2

        step_decrease = tf.Variable(0, trainable=False)
        objective_fun = ints + tf.reduce_mean(penal(diff, self.gamma, type=self.penal_type))
        rate = tf.train.exponential_decay(self.get_training_spec('rate'), step_decrease,
                                          self.get_training_spec('dec_interval'), self.get_training_spec('dec_rate'),
                                          staircase=False)

        train_op = tf.train.AdamOptimizer(learning_rate=rate, beta1=
        self.get_training_spec('beta1'), beta2=self.get_training_spec('beta2')).minimize(objective_fun)

        obj_eval = self.f(s_theta)
        self.running_primal_1 = running_primal_1
        self.running_primal_2 = running_primal_2
        self.mu0_hedge = s0
        self.total_hedge = s1
        self.obj_eval = obj_eval
        self.density_optimizer = density_optimizer
        self.S_mu_0 = s_mu_0
        self.S_theta = s_theta
        self.objective_fun = objective_fun
        self.train_op = train_op
        self.step_decrease = step_decrease
        self.den_var = den_var
        saver = tf.train.Saver()
        self.saver = saver

    def train_markov(self):
        n_train = self.get_training_spec('n_train')
        n_fine = self.get_training_spec('n_fine')
        n_report = self.get_training_spec('n_report')
        t0 = time.time()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            gen_mu_0 = self.mu_0(batch_size=self.get_training_spec('batch_size_mu_0'))
            gen_theta = self.markov_gen_theta(batch_size=self.get_training_spec('batch_size_theta'))
            vals = []
            vals_p1 = []
            vals_p2 = []
            for t in range(1, n_train+1):
                sample_mu_0 = next(gen_mu_0)
                (sample_theta, den_vals) = next(gen_theta)
                (c, p1v, p2v, _) = sess.run([self.objective_fun, self.running_primal_1, self.running_primal_2,
                                             self.train_op],
                                  feed_dict={self.S_mu_0: sample_mu_0, self.S_theta: sample_theta, self.den_var:
                                      den_vals})
                vals.append(c)
                vals_p1.append(p1v)
                vals_p2.append(p2v)

                if t % n_report == 0:
                    print('____________________________')
                    print('Time trained: ' + str(time.time() - t0))
                    print('Current step: ' + str(t))
                    print('Current value: ' + str(np.mean(vals[t-n_report:])))
                    print('Current primal 1 value: ' + str(np.mean(vals_p1[n_train + t - n_report:])))
                    print('Current primal 2 value: ' + str(np.mean(vals_p2[n_train + t - n_report:])))

            for t in range(1, n_fine+1):
                sample_mu_0 = next(gen_mu_0)
                (sample_theta, den_vals) = next(gen_theta)
                (c, p1v, p2v, _) = sess.run([self.objective_fun, self.running_primal_1, self.running_primal_2,
                                             self.train_op],
                                  feed_dict={self.S_mu_0: sample_mu_0, self.S_theta: sample_theta, self.den_var:
                                      den_vals, self.step_decrease: t})
                vals.append(c)
                vals_p1.append(p1v)
                vals_p2.append(p2v)

                if t % n_report == 0:
                    print('____________________________')
                    print('Time trained: ' + str(time.time() - t0))
                    print('Current step: ' + str(n_train + t))
                    print('Current value: ' + str(np.mean(vals[n_train+t-n_report:])))
                    print('Current primal 1 value: ' + str(np.mean(vals_p1[n_train + t - n_report:])))
                    print('Current primal 2 value: ' + str(np.mean(vals_p2[n_train + t - n_report:])))
            if self.saveit == 1:
                self.saver.save(sess, 'SavedModels/'+self.identifier)
        self.value_list = vals
        self.primal_1_value_list = vals_p1
        self.primal_2_value_list = vals_p2

    def train_markov_new(self):
        n_train = self.get_training_spec('n_train')
        n_fine = self.get_training_spec('n_fine')
        n_report = self.get_training_spec('n_report')
        t0 = time.time()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            gen_mu_0 = self.mu_0(batch_size=self.get_training_spec('batch_size_mu_0'))
            gen_theta = self.markov_gen_theta_new(batch_size=self.get_training_spec('batch_size_theta'))
            vals = []
            vals_p1 = []
            vals_p2 = []
            for t in range(1, n_train+1):
                sample_mu_0 = next(gen_mu_0)
                (sample_theta, den_vals) = next(gen_theta)

                (c, p1v, p2v, _) = sess.run([self.objective_fun, self.running_primal_1, self.running_primal_2,
                                             self.train_op], feed_dict={self.S_mu_0: sample_mu_0, self.S_theta:
                    sample_theta, self.den_var: den_vals})
                vals.append(c)
                vals_p1.append(p1v)
                vals_p2.append(p2v)

                vals.append(c)
                if t % n_report == 0:
                    print('____________________________')
                    print('Time trained: ' + str(time.time() - t0))
                    print('Current step: ' + str(t))
                    print('Current value: ' + str(np.mean(vals[t-n_report:])))
                    print('Current primal 1 value: ' + str(np.mean(vals_p1[t - n_report:])))
                    print('Current primal 2 value: ' + str(np.mean(vals_p2[t - n_report:])))

            for t in range(1, n_fine+1):
                sample_mu_0 = next(gen_mu_0)
                (sample_theta, den_vals) = next(gen_theta)
                (c, p1v, p2v, _) = sess.run([self.objective_fun, self.running_primal_1, self.running_primal_2,
                                             self.train_op],
                                  feed_dict={self.S_mu_0: sample_mu_0, self.S_theta: sample_theta, self.den_var:
                                      den_vals, self.step_decrease: t})
                vals.append(c)
                vals_p1.append(p1v)
                vals_p2.append(p2v)
                if t % n_report == 0:
                    print('____________________________')
                    print('Time trained: ' + str(time.time() - t0))
                    print('Current step: ' + str(n_train + t))
                    print('Current value: ' + str(np.mean(vals[n_train+t-n_report:])))
                    print('Current primal 1 value: ' + str(np.mean(vals_p1[n_train + t - n_report:])))
                    print('Current primal 2 value: ' + str(np.mean(vals_p2[n_train + t - n_report:])))
            if self.saveit == 1:
                self.saver.save(sess, 'SavedModels/'+self.identifier)
        self.value_list = vals
        self.primal_1_value_list = vals_p1
        self.primal_2_value_list = vals_p2

    def train_wasserstein(self):
        n_train = self.get_training_spec('n_train')
        n_fine = self.get_training_spec('n_fine')
        n_report = self.get_training_spec('n_report')
        n_lam_up = self.get_training_spec('n_lambda')
        rate_lambda = self.get_training_spec('rate_lambda')
        lam_fun = self.get_parameter('lam_fun')
        dec_rate = self.get_training_spec('dec_rate')
        dec_interval = self.get_training_spec('dec_interval')
        rho_val = self.get_parameter('rho_value')
        n_lamda_wait = self.get_training_spec('n_lambda_wait')

        t0 = time.time()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            gen_mu_0 = self.mu_0(batch_size=self.get_training_spec('batch_size_mu_0'))
            gen_theta = self.theta(batch_size=self.get_training_spec('batch_size_theta'))
            vals = []
            vals_p1 = []
            vals_p2 = []
            vals_met = []
            grads_lam = []
            lam_vals = [(0, self.get_training_spec('lambda_init'))]
            m0 = 0
            v0 = 0
            # l_train = 'grad_desc'
            # l_train = 'grad_desc'
            beta1 = self.get_training_spec('beta1')
            beta2 = self.get_training_spec('beta2')
            adam_ind = 0
            for t in range(1, n_train+n_fine+1):
                sample_mu_0 = next(gen_mu_0)
                sample_theta = next(gen_theta)
                (c, d1, d2, met_d, gl, _) = sess.run([self.objective_fun, self.running_primal_1, self.running_primal_2,
                                                  self.int_ctheta, self.grad_lambda, self.train_op],
                                  feed_dict={self.S_mu_0: sample_mu_0, self.S_theta: sample_theta,
                                             self.step_decrease: max(1, t-n_fine), self.rho: rho_val})
                vals_p1.append(d1)
                vals_p2.append(d2)
                vals.append(c)
                vals_met.append(met_d)
                grads_lam.append(gl)

                # if t > n_lamda_wait and t % n_lam_up == 0 and lam_fun != 'id':
                #     lambda_value = self.lambda_variable.eval()
                #     lam_vals.append((t, lambda_value))
                #     if t < n_train:
                #         if l_train == 'grad_desc':
                #             assign_op = self.lambda_variable.assign(lambda_value -
                #                                                     rate_lambda * np.mean(grads_lam[-n_lam_up:]))
                #         else:
                #             adam_ind += 1
                #             gvh = np.mean(grads_lam[-n_lam_up:])
                #             lrt = rate_lambda * np.sqrt(1-beta2**adam_ind) / (1-beta1**adam_ind)
                #             m0 = beta1 * m0 + (1-beta1) * gvh
                #             v0 = beta2 * v0 + (1-beta2) * gvh * gvh
                #             assign_op = self.lambda_variable.assign(lambda_value - lrt * m0 / (np.sqrt(v0)+(10**(-8))))
                #
                #     else:
                #         if l_train == 'grad_desc':
                #             assign_op = self.lambda_variable.assign(lambda_value - rate_lambda * (dec_rate **
                #                                               ((t-n_train)/dec_interval)) * np.mean(grads_lam[-n_lam_up:]))
                #         else:
                #             adam_ind += 1
                #             gvh = np.mean(grads_lam[-n_lam_up:])
                #             lrt = (rate_lambda * np.sqrt(1-beta2**adam_ind) / (1-beta1**adam_ind)) * (dec_rate **
                #                                               ((t-n_train)/dec_interval))
                #             m0 = beta1 * m0 + (1 - beta1) * gvh
                #             v0 = beta2 * v0 + (1 - beta2) * gvh * gvh
                #             assign_op = self.lambda_variable.assign(lambda_value - lrt * m0 / (np.sqrt(v0) + (10 ** (-8))))
                #
                #     sess.run(assign_op)

                if t > n_lamda_wait and t % n_lam_up == 0 and lam_fun != 'id':
                    lambda_value = self.lambda_variable.eval()
                    if t < n_train:
                        assign_op_lam = self.lambda_variable.assign(lambda_value -
                                                                rate_lambda * np.mean(grads_lam[-n_lam_up:]))
                    else:
                        assign_op_lam = self.lambda_variable.assign(lambda_value - rate_lambda * (dec_rate ** ((
                                                        t-n_train)/dec_interval)) * np.mean(grads_lam[-n_lam_up:]))
                    sess.run(assign_op_lam)
                    lam_vals.append((t, lambda_value))


                if t % n_report == 0:
                    if lam_fun == 'id':
                        lambda_value = 1/rho_val
                    else:
                        lambda_value = self.lambda_variable.eval()
                    print('____________________________')
                    print('Time trained: ' + str(time.time() - t0))
                    print('Current step: ' + str(t))
                    print('Current value of lambda: ' + str(lambda_value))
                    print('Current normal value: ' + str(np.mean(vals[t-n_report:])))
                    print('Current primal_1 value: ' + str(np.mean(vals_p1[t-n_report:])))
                    print('Current primal_2 value: ' + str(np.mean(vals_p2[t-n_report:])))
                    print('Current convergence criteria: ' + str(np.abs(np.mean(vals_p1[t-n_report:])-np.mean(
                        vals_p2[t-n_report:]))))
                    print('Current value for Wasserstein distance to reference: ' + str(np.mean(vals_met[t-n_report:])))
            if self.saveit == 1:
                self.saver.save(sess, 'SavedModels/'+self.identifier)
        self.value_list = vals
        self.primal_1_value_list = vals_p1
        self.primal_2_value_list = vals_p2
        self.vals_met = vals_met
        self.lam_vals = lam_vals

    def train_wasserstein_avar(self):
        n_train = self.get_training_spec('n_train')
        n_fine = self.get_training_spec('n_fine')
        n_report = self.get_training_spec('n_report')
        lam_fun = self.get_parameter('lam_fun')
        dec_rate = self.get_training_spec('dec_rate')
        dec_interval = self.get_training_spec('dec_interval')
        rho_val = self.get_parameter('rho_value')
        n_lamda_wait = self.get_training_spec('n_lambda_wait')
        n_lam_up = self.get_training_spec('n_lambda')
        rate_lambda = self.get_training_spec('rate_lambda')
        n_tau_wait = self.get_training_spec('n_tau_wait')
        n_tau_up = self.get_training_spec('n_tau')
        rate_tau = self.get_training_spec('rate_tau')
        avar_minmax = self.get_parameter('avar_minmax')
        lam_vals = [(0, self.get_training_spec('lambda_init'))]

        t0 = time.time()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            gen_mu_0 = self.mu_0(batch_size=self.get_training_spec('batch_size_mu_0'))
            gen_theta = self.theta(batch_size=self.get_training_spec('batch_size_theta'))
            vals = []
            vals_p1 = []
            vals_p2 = []
            vals_met = []
            grads_lam = []
            grads_tau = []
            for t in range(1, n_train+n_fine+1):
                sample_mu_0 = next(gen_mu_0)
                sample_theta = next(gen_theta)
                (c, d1, d2, met_d, gl, gt, _) = sess.run([self.objective_fun, self.running_primal_1,
                                                          self.running_primal_2, self.int_ctheta, self.grad_lambda,
                                                          self.grad_theta, self.train_op],
                                  feed_dict={self.S_mu_0: sample_mu_0, self.S_theta: sample_theta,
                                             self.step_decrease: max(0, t-n_fine), self.rho: rho_val})
                vals_p1.append(d1)
                vals_p2.append(d2)
                vals.append(c)
                vals_met.append(met_d)
                grads_lam.append(gl)
                grads_tau.append(gt)

                if t > n_lamda_wait and t % n_lam_up == 0 and lam_fun != 'id':
                    lambda_value = self.lambda_variable.eval()
                    if t < n_train:
                        assign_op_lam = self.lambda_variable.assign(lambda_value -
                                                                rate_lambda * np.mean(grads_lam[-n_lam_up:]))
                    else:
                        assign_op_lam = self.lambda_variable.assign(lambda_value - rate_lambda * (dec_rate ** ((
                                                        t-n_train)/dec_interval)) * np.mean(grads_lam[-n_lam_up:]))
                    sess.run(assign_op_lam)
                    lam_vals.append((t, lambda_value))

                if t > n_tau_wait and t % n_tau_up == 0:
                    tau_value = self.tau_variable.eval()
                    assign_op_tau = self.tau_variable.assign(tau_value - dec_rate ** (max(0, t-n_fine) / dec_interval)
                                                            * rate_tau * np.mean(grads_tau[-n_tau_up:]))
                    sess.run(assign_op_tau)

                if t % n_report == 0:
                    if lam_fun == 'id':
                        lambda_value = 1/rho_val
                    else:
                        lambda_value = self.lambda_variable.eval()
                    tau_value = self.tau_variable.eval()
                    print('____________________________')
                    print('Time trained: ' + str(time.time() - t0))
                    print('Current step: ' + str(t))
                    print('Current value of lambda: ' + str(lambda_value))
                    print('Current value of tau: ' + str(tau_value))
                    print('Current normal value: ' + str(np.mean(vals[t-n_report:])))
                    print('Current primal_1 value: ' + str(np.mean(vals_p1[t-n_report:])))
                    print('Current primal_2 value: ' + str(np.mean(vals_p2[t-n_report:])))
                    print('Current convergence criteria: ' + str(np.abs(np.mean(vals_p1[t-n_report:])-np.mean(
                        vals_p2[t-n_report:]))))
                    print('Current value for Wasserstein distance to reference: ' + str(np.mean(vals_met[t-n_report:])))
            if self.saveit == 1:
                self.saver.save(sess, 'SavedModels/'+self.identifier)
        self.value_list = vals
        self.primal_1_value_list = vals_p1
        self.primal_2_value_list = vals_p2
        self.vals_met = vals_met
        self.lam_vals = lam_vals

    def train_model(self):
        if self.prob_type == 'Markov' or self.prob_type == 'Markov_alone':
            self.train_markov()
            return 0
        if self.prob_type == 'Markov_new':
            self.train_markov_new()
            return 0
        if self.prob_type == 'Wasserstein_Marginals':
            self.train_wasserstein()
            return 0
        if self.prob_type == 'Wasserstein_Marginals_AVAR':
            self.train_wasserstein_avar()
            return 0
        n_train = self.get_training_spec('n_train')
        n_fine = self.get_training_spec('n_fine')
        n_report = self.get_training_spec('n_report')
        t0 = time.time()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            gen_mu_0 = self.mu_0(batch_size=self.get_training_spec('batch_size_mu_0'))
            gen_theta = self.theta(batch_size=self.get_training_spec('batch_size_theta'))
            vals = []
            vals_p1 = []
            vals_p2 = []
            for t in range(1, n_train+1):
                sample_mu_0 = next(gen_mu_0)
                sample_theta = next(gen_theta)
                (c, p1v, p2v, _) = sess.run([self.objective_fun, self.running_primal_1, self.running_primal_2,
                                             self.train_op],
                                  feed_dict={self.S_mu_0: sample_mu_0, self.S_theta: sample_theta})
                vals.append(c)
                vals_p1.append(p1v)
                vals_p2.append(p2v)
                if t % n_report == 0:
                    print('____________________________')
                    print('Time trained: ' + str(time.time() - t0))
                    print('Current step: ' + str(t))
                    print('Current value: ' + str(np.mean(vals[t-n_report:])))
                    print('Current primal 1 value: ' + str(np.mean(vals_p1[t - n_report:])))
                    print('Current primal 2 value: ' + str(np.mean(vals_p2[t - n_report:])))
            for t in range(1, n_fine+1):
                sample_mu_0 = next(gen_mu_0)
                sample_theta = next(gen_theta)
                (c, p1v, p2v, _) = sess.run([self.objective_fun, self.running_primal_1, self.running_primal_2,
                                             self.train_op],
                                  feed_dict={self.S_mu_0: sample_mu_0, self.S_theta: sample_theta, self.step_decrease: t})
                vals.append(c)
                vals_p1.append(p1v)
                vals_p2.append(p2v)
                if t % n_report == 0:
                    print('____________________________')
                    print('Time trained: ' + str(time.time() - t0))
                    print('Current step: ' + str(n_train + t))
                    print('Current value: ' + str(np.mean(vals[n_train+t-n_report:])))
                    print('Current primal 1 value: ' + str(np.mean(vals_p1[n_train + t - n_report:])))
                    print('Current primal 2 value: ' + str(np.mean(vals_p2[n_train + t - n_report:])))
            if self.saveit == 1:
                self.saver.save(sess, 'SavedModels/'+self.identifier)
        self.value_list = vals
        self.primal_1_value_list = vals_p1
        self.primal_2_value_list = vals_p2

    def gen_samples_markov(self, n_gen_samples=10000, max_density_av=1000):
        print('Generating samples...')
        t0 = time.time()
        n_report = self.get_training_spec('n_report')
        with tf.Session() as sess:
            self.saver.restore(sess, 'SavedModels/'+self.identifier)
            samples = np.zeros([0, self.time_steps, self.dimension])
            batch = self.get_training_spec('batch_size_theta')
            gen_theta = self.markov_gen_theta(batch_size=self.get_training_spec('batch_size_theta'))
            maxv = []
            ind = 0
            for t in range(max_density_av):
                ind += 1
                (sample_theta, den_vals) = next(gen_theta)
                dv = sess.run(self.density_optimizer, feed_dict={self.S_theta: sample_theta, self.den_var: den_vals})
                maxv.append(np.max(dv))
            while len(samples) < n_gen_samples:
                ind += 1
                (sample_theta, den_vals) = next(gen_theta)
                dv = sess.run(self.density_optimizer, feed_dict={self.S_theta: sample_theta, self.den_var: den_vals})
                maxv.append(np.max(dv))
                den_max = max(maxv[-max_density_av:])
                u = np.random.random_sample([batch])
                samples = np.append(samples, sample_theta[u * den_max <= dv, :, :], axis=0)
                if ind % n_report == 0:
                    print('Current generating iteration: ' + str(ind))
                    print('Current number of samples generated: ' + str(len(samples)))
        print('It took ' + str(time.time() - t0) + ' seconds to generate ' + str(n_gen_samples) + ' samples.')
        if self.samples is None:
            self.samples = samples
        else:
            self.samples = np.append(self.samples, samples, axis=0)

    def gen_samples_markov_new(self, n_gen_samples=10000, max_density_av=1000):
        print('Generating samples...')
        t0 = time.time()
        n_report = self.get_training_spec('n_report')
        with tf.Session() as sess:
            self.saver.restore(sess, 'SavedModels/'+self.identifier)
            samples = np.zeros([0, self.time_steps, self.dimension])
            batch = self.get_training_spec('batch_size_theta')
            gen_theta = self.markov_gen_theta_new(batch_size=self.get_training_spec('batch_size_theta'))
            maxv = []
            ind = 0
            for t in range(max_density_av):
                ind += 1
                (sample_theta, den_vals) = next(gen_theta)
                dv = sess.run(self.density_optimizer, feed_dict={self.S_theta: sample_theta, self.den_var: den_vals})
                maxv.append(np.max(dv))
            while len(samples) < n_gen_samples:
                ind += 1
                (sample_theta, den_vals) = next(gen_theta)
                dv = sess.run(self.density_optimizer, feed_dict={self.S_theta: sample_theta, self.den_var: den_vals})
                maxv.append(np.max(dv))
                den_max = max(maxv[-max_density_av:])
                u = np.random.random_sample([batch])
                samples = np.append(samples, sample_theta[u * den_max <= dv, :, :], axis=0)
                if ind % n_report == 0:
                    print('Current generating iteration: ' + str(ind))
                    print('Current number of samples generated: ' + str(len(samples)))
        print('It took ' + str(time.time() - t0) + ' seconds to generate ' + str(n_gen_samples) + ' samples.')
        if self.samples is None:
            self.samples = samples
        else:
            self.samples = np.append(self.samples, samples, axis=0)

    def gen_samples(self, n_gen_samples=10000, max_density_av=1000):
        if self.prob_type == 'Markov' or self.prob_type == 'Markov_alone':
            self.gen_samples_markov(n_gen_samples=n_gen_samples, max_density_av=max_density_av)
            return
        if self.prob_type == 'Markov_new':
            self.gen_samples_markov_new(n_gen_samples=n_gen_samples, max_density_av=max_density_av)
            return
        if self.prob_type == 'Wasserstein_Marginals' or self.prob_type == 'Wasserstein_Marginals_AVAR':
            self.gen_samples_wasserstein(n_gen_samples=n_gen_samples, max_density_av=max_density_av)
            return
        print('Generating samples...')
        t0 = time.time()
        n_report = self.get_training_spec('n_report')
        samples = np.zeros([0, self.time_steps, self.dimension])

        with tf.Session() as sess:
            self.saver.restore(sess, 'SavedModels/'+self.identifier)
            batch = self.get_training_spec('batch_size_theta')
            gen_theta = self.theta(batch_size=batch)
            maxv = []
            ind = 0
            for t in range(max_density_av):
                ind += 1
                sample_theta = next(gen_theta)
                dv = sess.run(self.density_optimizer, feed_dict={self.S_theta: sample_theta})
                maxv.append(np.max(dv))
            while len(samples) < n_gen_samples:
                ind += 1
                sample_theta = next(gen_theta)
                dv = sess.run(self.density_optimizer, feed_dict={self.S_theta: sample_theta})
                maxv.append(np.max(dv))
                den_max = max(maxv[-max_density_av:])
                u = np.random.random_sample([batch])
                samples = np.append(samples, sample_theta[u * den_max <= dv, :, :], axis=0)
                if ind % n_report == 0:
                    print('Current generating iteration: ' + str(ind))
                    print('Current number of samples generated: ' + str(len(samples)))
        print('It took ' + str(time.time() - t0) + ' seconds to generate ' + str(n_gen_samples) + ' samples.')
        if self.samples is None:
            self.samples = samples
        else:
            self.samples = np.append(self.samples, samples, axis=0)

    def gen_samples_wasserstein(self, n_gen_samples=10000, max_density_av=1000):
        if self.prob_type == 'Markov' or self.prob_type == 'Markov_alone':
            self.gen_samples_markov(n_gen_samples=n_gen_samples, max_density_av=max_density_av)
            return
        print('Generating samples...')
        t0 = time.time()
        n_report = self.get_training_spec('n_report')
        samples = np.zeros([0, self.time_steps, 2*self.dimension])
        rho_value = self.get_parameter('rho_value')

        with tf.Session() as sess:
            self.saver.restore(sess, 'SavedModels/'+self.identifier)
            batch = self.get_training_spec('batch_size_theta')
            gen_theta = self.theta(batch_size=batch)
            maxv = []
            ind = 0
            for t in range(max_density_av):
                ind += 1
                sample_theta = next(gen_theta)
                dv = sess.run(self.density_optimizer, feed_dict={self.S_theta: sample_theta, self.rho: rho_value})
                maxv.append(np.max(dv))
            while len(samples) < n_gen_samples:
                ind += 1
                sample_theta = next(gen_theta)
                dv = sess.run(self.density_optimizer, feed_dict={self.S_theta: sample_theta, self.rho: rho_value})
                maxv.append(np.max(dv))
                den_max = max(maxv[-max_density_av:])
                u = np.random.random_sample([batch])
                samples = np.append(samples, sample_theta[u * den_max <= dv, :, :], axis=0)
                if ind % n_report == 0:
                    print('Current generating iteration: ' + str(ind))
                    print('Current number of samples generated: ' + str(len(samples)))
        print('It took ' + str(time.time() - t0) + ' seconds to generate ' + str(n_gen_samples) + ' samples.')
        if self.samples is None:
            self.samples = samples
        else:
            self.samples = np.append(self.samples, samples, axis=0)

    def gen_samples_wasserstein_while_training(self, n_gen_samples=10000, max_density_av=1000, step_decrease=20000):
        print('Generating samples...')
        t0 = time.time()
        n_report = self.get_training_spec('n_report')
        samples = np.zeros([0, self.time_steps, 2*self.dimension])
        rho_value = self.get_parameter('rho_value')
        n_lam_up = self.get_training_spec('n_lambda')
        lam_fun = self.get_parameter('lam_fun')
        dec_rate = self.get_training_spec('dec_rate')
        dec_interval = self.get_training_spec('dec_interval')
        grads_lam = [0]
        rate_lambda = self.get_training_spec('rate_lambda')
        vals = []
        batch = self.get_training_spec('batch_size_theta')
        with tf.Session() as sess:
            self.saver.restore(sess, 'SavedModels/'+self.identifier)
            gen_mu_0 = self.mu_0(batch_size=self.get_training_spec('batch_size_mu_0'))
            gen_theta = self.theta(batch_size=self.get_training_spec('batch_size_theta'))
            maxv = []
            ind = 0
            for t in range(max_density_av):
                ind += 1
                sample_mu_0 = next(gen_mu_0)
                sample_theta = next(gen_theta)
                (c, dv, _, gl) = sess.run([self.objective_fun, self.density_optimizer, self.train_op, self.grad_lambda],
                                       feed_dict={self.S_theta: sample_theta, self.rho: rho_value,
                                                  self.step_decrease: step_decrease, self.S_mu_0: sample_mu_0})
                vals.append(c)
                grads_lam.append(gl)
                if ind % n_lam_up == 0 and lam_fun != 'id':
                    lambda_value = self.lambda_variable.eval()
                    assign_op = self.lambda_variable.assign(lambda_value - rate_lambda * (dec_rate **
                                                      (step_decrease/dec_interval)) * np.mean(grads_lam[-n_lam_up:]))
                    sess.run(assign_op)

                maxv.append(np.max(dv))
            while len(samples) < n_gen_samples:
                ind += 1
                sample_mu_0 = next(gen_mu_0)
                sample_theta = next(gen_theta)
                (c, dv, _, gl) = sess.run([self.objective_fun, self.density_optimizer, self.train_op, self.grad_lambda],
                                       feed_dict={self.S_theta: sample_theta, self.rho: rho_value,
                                                  self.step_decrease: step_decrease, self.S_mu_0: sample_mu_0})
                vals.append(c)
                grads_lam.append(gl)
                if ind % n_lam_up == 0 and lam_fun != 'id':
                    lambda_value = self.lambda_variable.eval()
                    assign_op = self.lambda_variable.assign(lambda_value - rate_lambda * (dec_rate **
                                                      (step_decrease/dec_interval)) * np.mean(grads_lam[-n_lam_up:]))
                    sess.run(assign_op)

                maxv.append(np.max(dv))
                den_max = max(maxv[-max_density_av:])
                u = np.random.random_sample([batch])
                samples = np.append(samples, sample_theta[u * den_max <= dv, :, :], axis=0)
                if ind % n_report == 0:
                    print('Current generating iteration: ' + str(ind))
                    print('Current number of samples generated: ' + str(len(samples)))
                    print('Current dual value: ' + str(np.mean(vals[-n_report:])))
        print('It took ' + str(time.time() - t0) + ' seconds to generate ' + str(n_gen_samples) + ' samples.')
        if self.samples is None:
            self.samples = samples
        else:
            self.samples = np.append(self.samples, samples, axis=0)

    def gen_samples_while_training(self, n_gen_samples=10000, max_density_av=1000, step_decrease=20000):
        if self.prob_type == 'Markov' or self.prob_type == 'Markov_alone':
            self.gen_samples_markov_while_training(n_gen_samples=n_gen_samples, max_density_av=max_density_av,
                                                   step_decrease=step_decrease)
            return
        if self.prob_type == 'Wasserstein_Marginals' or self.prob_type == 'Wasserstein_Marginals_AVAR':
            self.gen_samples_wasserstein_while_training(n_gen_samples=n_gen_samples, max_density_av=max_density_av,
                                                        step_decrease=step_decrease)
            return
        print('Generating samples...')
        t0 = time.time()
        n_report = self.get_training_spec('n_report')
        with tf.Session() as sess:
            self.saver.restore(sess, 'SavedModels/'+self.identifier)
            samples = np.zeros([0, self.time_steps, self.dimension])
            batch = self.get_training_spec('batch_size_theta')
            gen_mu_0 = self.mu_0(batch_size=self.get_training_spec('batch_size_mu_0'))
            gen_theta = self.theta(batch_size=self.get_training_spec('batch_size_theta'))
            maxv = []
            ind = 0
            for t in range(1, max_density_av+1):
                sample_mu_0 = next(gen_mu_0)
                sample_theta = next(gen_theta)
                ind += 1
                (dv, _) = sess.run([self.density_optimizer, self.train_op],
                                    feed_dict={self.S_mu_0: sample_mu_0, self.S_theta: sample_theta,
                                               self.step_decrease: step_decrease})
                maxv.append(np.max(dv))
            while len(samples) < n_gen_samples:
                ind += 1
                sample_mu_0 = next(gen_mu_0)
                sample_theta = next(gen_theta)
                (dv, _) = sess.run([self.density_optimizer, self.train_op],
                                    feed_dict={self.S_mu_0: sample_mu_0, self.S_theta: sample_theta,
                                               self.step_decrease: step_decrease})
                maxv.append(np.max(dv))
                den_max = max(maxv[-max_density_av:])
                u = np.random.random_sample([batch])
                samples = np.append(samples, sample_theta[u * den_max <= dv, :, :], axis=0)
                if ind % n_report == 0:
                    print('Current generating iteration: ' + str(ind))
                    print('Current number of samples generated: ' + str(len(samples)))
        print('It took ' + str(time.time() - t0) + ' seconds to generate ' + str(n_gen_samples) + ' samples.')
        if self.samples is None:
            self.samples = samples
        else:
            self.samples = np.append(self.samples, samples, axis=0)

    def gen_samples_markov_while_training(self, n_gen_samples=10000, max_density_av=1000, step_decrease=20000):
        print('Generating samples...')
        t0 = time.time()
        n_report = self.get_training_spec('n_report')
        with tf.Session() as sess:
            self.saver.restore(sess, 'SavedModels/'+self.identifier)
            samples = np.zeros([0, self.time_steps, self.dimension])
            batch = self.get_training_spec('batch_size_theta')
            gen_mu_0 = self.mu_0(batch_size=self.get_training_spec('batch_size_mu_0'))
            gen_theta = self.markov_gen_theta(batch_size=self.get_training_spec('batch_size_theta'))
            maxv = []
            ind = 0
            for t in range(max_density_av):
                ind += 1
                sample_mu_0 = next(gen_mu_0)
                (sample_theta, den_vals) = next(gen_theta)
                (dv, _) = sess.run([self.density_optimizer, self.train_op],
                                  feed_dict={self.S_mu_0: sample_mu_0, self.S_theta: sample_theta, self.den_var:
                                      den_vals, self.step_decrease: step_decrease})
                maxv.append(np.max(dv))
            while len(samples) < n_gen_samples:
                ind += 1
                sample_mu_0 = next(gen_mu_0)
                (sample_theta, den_vals) = next(gen_theta)
                (dv, _) = sess.run([self.density_optimizer, self.train_op],
                                  feed_dict={self.S_mu_0: sample_mu_0, self.S_theta: sample_theta, self.den_var:
                                      den_vals, self.step_decrease: step_decrease})
                maxv.append(np.max(dv))
                den_max = max(maxv[-max_density_av:])
                u = np.random.random_sample([batch])
                samples = np.append(samples, sample_theta[u * den_max <= dv, :, :], axis=0)
                if ind % n_report == 0:
                    print('Current generating iteration: ' + str(ind))
                    print('Current number of samples generated: ' + str(len(samples)))
        print('It took ' + str(time.time() - t0) + ' seconds to generate ' + str(n_gen_samples) + ' samples.')
        if self.samples is None:
            self.samples = samples
        else:
            self.samples = np.append(self.samples, samples, axis=0)

    def eval_static_hedge(self, points):
        with tf.Session() as sess:
            self.saver.restore(sess, 'SavedModels/'+self.identifier)
            fvals = sess.run(self.mu0_hedge, feed_dict={self.S_mu_0: points})
        return fvals

    def eval_total_hedge(self, points):
        with tf.Session() as sess:
            self.saver.restore(sess, 'SavedModels/'+self.identifier)
            fvals = sess.run(self.total_hedge, feed_dict={self.S_theta: points})
        return fvals

    def get_primal_value(self, n_samples):
        if len(self.samples) >= n_samples:
            with tf.Session() as sess:
                val_h = sess.run(self.obj_eval, feed_dict={self.S_theta: self.samples[-n_samples:, :, :]})
        else:
            self.gen_samples(n_gen_samples=n_samples - len(self.samples))
            with tf.Session() as sess:
                val_h = sess.run(self.obj_eval, feed_dict={self.S_theta: self.samples[-n_samples:, :, :]})
        return np.mean(val_h)

    def solve_LP(self, dis_each=50, n_samples=10**7, eps=10**-9, quant_arr_calc=0, minmax='max', mmot_eps=0,
                 given_ev=None, method='b'):
        from LP_Programs import u_quantization_bundle, lp_hom_mmot
        d = self.dimension
        t = self.time_steps
        print('Calculating quantization ... ')
        if self.quantile_funs:
            (a_list, w_list) = u_quantization_bundle(t, d, p_samp=self.mu_0, n=dis_each, n_samples=n_samples,
                                                     quantile_funs=self.quantile_funs, eps=eps,
                                                     quant_arr_calc=quant_arr_calc, given_ev=given_ev, method=method)
        else:
            (a_list, w_list) = u_quantization_bundle(t, d, p_samp=self.mu_0, n=dis_each, n_samples=n_samples,
                                                     given_ev=given_ev, method=method)
        print('Done!')
        a_w_list = [[(a_list[i][j], w_list[i][j]) for j in range(d)] for i in range(t)]
        obj_val, opti = lp_hom_mmot(a_w_list, self.f_np, mart=1, minmax=minmax, mmot_eps=mmot_eps)
        return obj_val, opti, a_list, w_list


if __name__ == '__main__':
    # def f_obj(s):
    #     return -tf.nn.relu(s[:, T-1, 0] - s[:, T-2, 0])

    T = 2
    d = 1
    K = 2
    P_TYPE = 'OT'  # 'MMOT', 'OT', 'Markov_new', 'Markov_alone'
    MINMAX = 1  # 1 means supremum (over measures) and -1 means infimum (over measures)

    def f_obj(x):
        return MINMAX * tf.nn.relu(tf.reduce_sum(x[:, T-1, :] - x[:, 0, :], axis=1))
    # (gen_fun, density_fun) = simple_random_mixture(K, T, d, den=1, each_dim_sep=1, hom_mc=1)
    (gen_fun, density_fun) = simple_random_mixture(K, T, d, den=1, each_dim_sep=1, hom_mc=0)

    training_spec = {'batch_size_mu_0': 2 ** 10, 'batch_size_theta': 2 ** 10, 'n_train': 20000,
                     'n_fine': 20000, 'markov_trading_cost': 0.1, 'n_report': 1000}

    bla = Hedging(theta=gen_fun, mu_0=gen_fun, f=f_obj, prob_type=P_TYPE, t=T, d=d, gamma=1000,
                  training_spec=training_spec, hidden=((0, 128), (1, 128)))
    print('Problem type: ' + bla.prob_type)
    bla.__setattr__('densities', density_fun)
    bla.build_graph()
    bla.train_model()
    N_SAMPLES = 10000
    bla.gen_samples(n_gen_samples=N_SAMPLES)
    import matplotlib.pyplot as plt
    plt.scatter(bla.samples[:N_SAMPLES, 1, 0], bla.samples[:, 2, 0], s=0.75)
    plt.savefig('Images/'+bla.identifier+'T12')
    plt.show()
    plt.scatter(bla.samples[:N_SAMPLES, 0, 0], bla.samples[:, 1, 0], s=0.75)
    plt.savefig('Images/'+bla.identifier+'T01')
    plt.show()
    print(bla.get_primal_value(N_SAMPLES))

