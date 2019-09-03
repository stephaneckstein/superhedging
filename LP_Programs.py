import numpy as np
from gurobipy import *


# General note:
# A discrete measure is represented as a list like element with two entries (x_list, p_list),
# where x_list is a list-like object of either numbers (if the measure is one dimensional) or numpy-arrays of all the
# same shape
# p_list is a list of non-negative numbers which sum to one


def u_quantization(p_samp=None, p_quant=None, n=100, n_samples=10**7, eps=10**(-9), quant_arr_calc=0):
    """
    computes the U-quantization of a given one dimensional measure. Notably preservers convex order (if calculated
    exactly).
    Source is the paper
    "Quantizations of probability measures and preservation of the convex order"
    by David M. Baker
    :param p_samp:
    :param p_quant:
    :param n:
    :param n_samples:
    :param eps: lowest value will be eps and highest value 1-eps where quantile function is evaluated
    :param quant_arr_calc: if p_quant is given, this specifies if p_quant can be evaluated on arrays (value 1)
    :return: discrete measure as described in General note at start of file
    """
    a = np.zeros(n)

    if p_quant:
        step = 1 / n_samples
        z = np.arange(0, 1 + step, step)
        z[0] = eps
        z[-1] = 1 - eps
        if quant_arr_calc:
            q_vec = p_quant(z)
        else:
            q_vec = np.zeros(n_samples)
            for i in range(n_samples):
                q_vec[i] = p_quant(z[i])
        small_samp = n_samples/n
        for i in range(n):
            a[i] = np.mean(q_vec[int(round(small_samp))*i : int(round(small_samp))*(i+1)])
    else:
        sampler = p_samp(n_samples)
        samp = next(sampler)
        q_vec = sorted(samp)
        small_samp = n_samples/n
        for i in range(n):
            a[i] = np.mean(q_vec[int(round(small_samp))*i : int(round(small_samp))*(i+1)])
    return (a, 1/n * np.ones(n))


def u_quantization_bundle(t, d, p_samp=None, n=100, n_samples=10**7, quantile_funs=None, eps=10**-9,
                          quant_arr_calc=0, given_ev=None, method='b'):
    """

    :param t:
    :param d:
    :param p_samp: generator for t x d many marginals
    :param n:
    :param n_samples:
    :param eps:
    :param quant_arr_calc:
    :param given_ev: np.array of size [d] which specifies the expected value for each marginal. Same for each time point
    :param method:
    :return:
    """
    if not hasattr(n, '__len__'):
        n = [[n for j in range(d)] for i in range(t)]

    a_list = [[np.zeros(n[i][j]) for j in range(d)] for i in range(t)]

    if quantile_funs:
        for i in range(t):
            for j in range(d):
                step = 1 / n_samples
                z = np.arange(0, 1 + step, step)
                z[0] = eps
                z[-1] = 1 - eps
                if method == 'a':
                    if quant_arr_calc:
                        q_vec = quantile_funs[i][j](z)
                    else:
                        q_vec = np.zeros(n_samples)
                        for k in range(n_samples):
                            q_vec[k] = quantile_funs[i][j](z[k])
                    small_samp = n_samples / n[i][j]
                    sh = 0
                    for k in range(n[i][j]):
                        a_list[i][j][k] = np.mean(q_vec[int(round(small_samp)) * k: int(round(small_samp)) * (k + 1)])
                        sh += a_list[i][j][k]
                    sh /= n[i][j]
                    if given_ev:
                        diff = given_ev[j] - sh
                        for k in range(n[i][j]):
                            a_list[i][j][k] += diff

                elif method == 'b':
                    sh = 0
                    for k in range(n[i][j]):
                        a_list[i][j][k] = quantile_funs[i][j]((2*(k+1)-1)/(2*n[i][j]))
                        sh += a_list[i][j][k]
                    sh /= n[i][j]
                    if given_ev:
                        diff = given_ev[j] - sh
                        # a_list[i][j][-1] += n[i][j] * diff
                        for k in range(n[i][j]):
                            a_list[i][j][k] += diff
    else:
        sampler = p_samp(n_samples)
        samp = next(sampler)
        for i in range(t):
            for j in range(d):
                q_vec = sorted(samp[:, i, j])
                small_samp = n_samples/n[i][j]
                sh = 0
                for k in range(n[i][j]):
                    if method == 'a':
                        a_list[i][j][k] = np.mean(q_vec[int(round(small_samp))*k : int(round(small_samp))*(k+1)])
                    elif method =='b':
                        a_list[i][j][k] = q_vec[int(round(((2*(k+1)-1)/(2*n[i][j])) * n_samples))]
                    sh += a_list[i][j][k]
                sh /= n[i][j]
                if given_ev:
                    diff = given_ev[j] - sh
                    for k in range(n[i][j]):
                        a_list[i][j][k] += diff

    w_list = [[1/n[i][j] * np.ones(n[i][j]) for j in range(d)] for i in range(t)]
    return(a_list, w_list)


def visualize_LP_sol(points1, points2, weights, name='testname', saveit=0):
    # points1 is np-array of size n_1, points2 of size n_2. weights is [n_1, n_2]
    import matplotlib.pyplot as plt
    pp1, pp2 = np.meshgrid(points1, points2)
    plt.scatter(pp1, pp2, s=weights * 10 / np.max(weights), marker='o')
    if saveit == 1:
        plt.savefig(name)
    plt.show()


def gromov_wasserstein(mu1, mu2, c_1, c_2, p=2):
    # Tried doing it with a quadratic objective but since it's not convex, gurobi is unhappy
    n_1 = len(mu1[1])
    n_2 = len(mu2[1])

    print('Building cos matrix...')
    cost_mat = np.zeros([n_1, n_2, n_1, n_2])
    cost_mat_test = np.zeros([n_1*n_2, n_1*n_2])
    for i1 in range(n_1):
        x = mu1[0][i1]
        for i1p in range(n_1):
            xp = mu1[0][i1p]
            for i2 in range(n_2):
                y = mu2[0][i2]
                for i2p in range(n_2):
                    yp = mu2[0][i2p]
                    cost_mat[i1, i2, i1p, i2p] = np.abs(c_1(x, xp) - c_2(y, yp)) ** p
                    cost_mat_test[i1*n_2 + i2, i1p*n_2+i2p] = np.abs(c_1(x, xp) - c_2(y, yp)) ** p
    print('Done!')
    print(cost_mat_test)
    print(np.linalg.eigvals(cost_mat_test))

    print('Initializing model...')
    m = Model('Primal')
    pi_var = m.addVars(n_1, n_2, lb=0, name='pi_var')
    print('Done')

    print('Adding marginal constraints...')
    m.addConstrs((pi_var.sum(i, '*') == mu1[1][i] for i in range(n_1)), name='xmarg')
    m.addConstrs((pi_var.sum('*', i) == mu2[1][i] for i in range(n_2)), name='ymarg')
    print('Done!')

    print('Setting objective function...')
    obj = QuadExpr()
    for i1 in range(n_1):
        for i2 in range(n_2):
            for i1p in range(n_1):
                for i2p in range(n_2):
                    obj += pi_var[i1, i2] * pi_var[i1p, i2p] * cost_mat[i1, i2, i1p, i2p]
    m.setObjective(obj, GRB.MINIMIZE)
    print('Done! Ready to solve:')

    m.optimize()
    objective_val = m.ObjVal
    optimizer = np.zeros([n_1, n_2])
    for i1 in range(n_1):
        for i2 in range(n_2):
            optimizer[i1, i2] = pi_var[i1, i2].x

    return objective_val, optimizer


def lp_hom_mmot(margs, f_np, mart=0, hom=0, mart_markovian=0, minmax='min', Delta=None, mmot_eps=0, hom_eps=0):
    """

    :param margs: list with t entries, (each having d entries, where each entry is a discrete measure)
    :param f_np: takes np array of shape [t, d] as input and returns value
    :param mart:
    :param hom: if hom=1, it is assumed that there is no redundancy in the support of each measure. So there should
    not be two dirac measures at the same point in one of the marginal descriptions.
    :return:
    """
    t = len(margs)
    d = len(margs[0])
    ind = np.ones(t * d, dtype=int)
    ind_gu = [1 for i in range(t*d)]
    for t_ind in range(t):
        for d_ind in range(d):
            samp_marg = len(margs[t_ind][d_ind][1])
            ind[t_ind * d + d_ind] = samp_marg
            ind_gu[t_ind * d + d_ind] = int(round(samp_marg))
    ind = tuple(ind)

    print('Building cost matrix...')
    # Build Cost matrix (or tensor)
    cost_mat = np.zeros(ind)
    input_arr = np.zeros([t, d])
    for index in np.ndindex(ind):
        for t_ind in range(t):
            for d_ind in range(d):
                input_arr[t_ind, d_ind] = margs[t_ind][d_ind][0][index[t_ind * d + d_ind]]
        cost_mat[index] = f_np(input_arr)
    print('Done!')

    print('Initializing model... ')
    # initialize model
    m = Model('Primal')
    pi_var = m.addVars(*ind_gu, lb=0, name='pi_var')
    print('Done!')

    print('Adding marginal constraints...')
    gurobi_sum_input = ['*' for i in range(t*d)]
    # Add marginal constraints
    for t_ind in range(t):
        for d_ind in range(d):
            for i_td in range(ind_gu[t_ind * d + d_ind]):
                sum_input = gurobi_sum_input.copy()
                sum_input[t_ind * d + d_ind] = i_td
                m.addConstr(pi_var.sum(*sum_input) == margs[t_ind][d_ind][1][i_td], 'marg_'+str(t_ind)+'_'+str(
                    d_ind)+'_'+str(i_td))
    print('Done!')

    # Add possible martingale constraint:
    if mart == 1:
        print('Adding martingale constraints...')
        if mmot_eps == 0:
            for d_ind in range(d):
                for t_1_ind in range(t-1):
                    for t_2_ind in range(t_1_ind+1, t):
                        for ind_1 in np.ndindex(ind[:(t_1_ind + 1) * d]):
                            sum_input = gurobi_sum_input.copy()
                            sum_input[:(t_1_ind + 1) * d] = ind_1
                            lhs = pi_var.sum(*sum_input) * margs[t_1_ind][d_ind][0][ind_1[t_1_ind * d + d_ind]]
                            rhs = LinExpr()
                            for ind_2 in np.ndindex(ind[(t_1_ind+1) * d:]):
                                rhs += pi_var[ind_1+ind_2] * margs[t_2_ind][d_ind][0][ind_2[(t_2_ind-t_1_ind-1) * d +
                                                                                            d_ind]]
                            m.addConstr(lhs == rhs, 'martingale_'+str(d_ind)+'_'+str(t_1_ind)+'_'+str(
                                t_2_ind)+'_'+"_".join([str(x) for x in ind_1]))
        else:
            for d_ind in range(d):
                for t_1_ind in range(t-1):
                    for t_2_ind in range(t_1_ind+1, t):
                        for ind_1 in np.ndindex(ind[:(t_1_ind + 1) * d]):
                            sum_input = gurobi_sum_input.copy()
                            sum_input[:(t_1_ind + 1) * d] = ind_1
                            lhs = pi_var.sum(*sum_input) * margs[t_1_ind][d_ind][0][ind_1[t_1_ind * d + d_ind]]
                            rhs = LinExpr()
                            for ind_2 in np.ndindex(ind[(t_1_ind+1) * d:]):
                                rhs += pi_var[ind_1+ind_2] * margs[t_2_ind][d_ind][0][ind_2[(t_2_ind-t_1_ind-1) * d +
                                                                                            d_ind]]
                            m.addConstr(lhs - rhs <= mmot_eps, 'martingale_'+str(d_ind)+'_'+str(
                                t_1_ind)+'_'+str(t_2_ind)+'_'+"_".join([str(x) for x in ind_1]))
                            m.addConstr(rhs - lhs <= mmot_eps, 'martingale_sw_'+str(d_ind)+'_'+str(
                                t_1_ind)+'_'+str(t_2_ind)+'_'+"_".join([str(x) for x in ind_1]))


        print('Done!')

    # Add possible markovian martingale constraint
    # Should be the constraint E[S_{t+1}|S_t] instead of E[S_{t+1}|F_t] as Labordere introduced
    if mart_markovian == 1:
        # TODO
        return 0

    # Add possible homogeneity constraint:
    # Possible TODO: A sorted structure for the supports of the marginals would allow for a faster setup of the
    # homogeneity constraints...
    if hom == 1:
        print('Adding homogeneity constraints...')
        if not Delta:
            sblub = 3
            if sblub == 17:
                Delta = []
                for s1 in range(t-2):
                    t1 = s1 + 2
                    for tau in range(1, t-t1):
                        Delta.append((s1, t1, tau))
                print('Setting Delta:')
                print(Delta)
            else:
                Delta = []
                for s1 in range(t-2):
                    for t1 in range(s1+1, t-1):
                        for tau in range(1, t-t1):
                            Delta.append((s1, t1, tau))
                print('Setting Delta:')
                print(Delta)
        for d_ind in range(d):
            for (s1, t1, tau) in Delta:
                for ind_1_x in range(ind_gu[s1*d+d_ind]):
                    for ind_2_x in range(ind_gu[t1*d+d_ind]):
                        if margs[s1][d_ind][0][ind_1_x] == margs[t1][d_ind][0][ind_2_x]:
                            for ind_1_y in range(ind_gu[(s1+tau)*d + d_ind]):
                                for ind_2_y in range(ind_gu[(t1+tau)*d + d_ind]):
                                    if margs[s1+tau][d_ind][0][ind_1_y] == margs[t1+tau][d_ind][0][ind_2_y]:
                                        # In this case the support of the later time steps is present for both s1+tau
                                        #  and t1+tau.
                                        sum_input_lhs = gurobi_sum_input.copy()
                                        sum_input_lhs[s1*d+d_ind] = ind_1_x
                                        sum_input_lhs[(s1+tau)*d+d_ind] = ind_1_y
                                        lhs = pi_var.sum(*sum_input_lhs) * margs[t1][d_ind][1][ind_2_x]
                                        sum_input_rhs = gurobi_sum_input.copy()
                                        sum_input_rhs[t1*d+d_ind] = ind_2_x
                                        sum_input_rhs[(t1+tau)*d+d_ind] = ind_2_y
                                        rhs = pi_var.sum(*sum_input_rhs) * margs[s1][d_ind][1][ind_1_x]
                                        if hom_eps == 0:
                                            m.addConstr(lhs == rhs, 'hom_'+str(s1)+'_'+str(t1)+'_'+str(tau)+'_'+str(ind_1_x)
                                                        +'_'+str(ind_2_x)+'_'+str(ind_1_y)+'_'+str(ind_2_y))
                                        else:
                                            m.addConstr(-hom_eps <= lhs-rhs <= hom_eps, 'hom_'+str(s1)+'_'+str(
                                                t1)+'_'+str(
                                                tau)+'_'+str(ind_1_x)
                                                        +'_'+str(ind_2_x)+'_'+str(ind_1_y)+'_'+str(ind_2_y))

                            for ind_1_y in range(ind_gu[(s1+tau)*d + d_ind]):
                                b = 0
                                for ind_2_y in range(ind_gu[(t1+tau)*d + d_ind]):
                                    if margs[s1 + tau][d_ind][0][ind_1_y] == margs[t1 + tau][d_ind][0][ind_2_y]:
                                        b = 1
                                        break
                                if b == 0:
                                    # In this case there is mass at a point at time s1+tau but not at time t1+tau. So
                                    #  at s1+tau it has to be 0
                                    sum_input = gurobi_sum_input.copy()
                                    sum_input[s1*d+d_ind] = ind_1_x
                                    sum_input[(s1+tau)*d+d_ind] = ind_1_y
                                    if hom_eps == 0:
                                        m.addConstr(margs[t1][d_ind][1][ind_2_x] * pi_var.sum(*sum_input) == 0, 'hom_l0_'+
                                                    str(s1)+'_'+str(t1)+'_'+str(tau)+'_'+str(ind_1_x)+'_'+str(
                                                    ind_2_x)+'_'+str(ind_1_y))
                                    else:
                                        m.addConstr(-hom_eps <= margs[t1][d_ind][1][ind_2_x] * pi_var.sum(*sum_input)
                                                    <= hom_eps, 'hom_l0_'+
                                                    str(s1)+'_'+str(t1)+'_'+str(tau)+'_'+str(ind_1_x)+'_'+str(
                                                    ind_2_x)+'_'+str(ind_1_y))


                            for ind_2_y in range(ind_gu[(t1 + tau) * d + d_ind]):
                                b = 0
                                for ind_1_y in range(ind_gu[(s1 + tau) * d + d_ind]):
                                    if margs[s1 + tau][d_ind][0][ind_1_y] == margs[t1 + tau][d_ind][0][ind_2_y]:
                                        b = 1
                                        break
                                if b == 0:
                                    # In this case there is mass at a point at time t1+tau but not at time s1+tau. So
                                    #  at s1+tau it has to be 0
                                    sum_input = gurobi_sum_input.copy()
                                    sum_input[t1*d+d_ind] = ind_2_x
                                    sum_input[(t1+tau)*d+d_ind] = ind_2_y
                                    if hom_eps == 0:
                                        m.addConstr(margs[s1][d_ind][1][ind_1_x] * pi_var.sum(*sum_input) == 0, 'hom_r0_'+
                                                    str(s1)+'_'+str(t1)+'_'+str(tau)+'_'+str(ind_1_x)+'_'+str(
                                                    ind_2_x)+'_'+str(ind_2_y))
                                    else:
                                        m.addConstr(-hom_eps<=margs[s1][d_ind][1][ind_1_x] * pi_var.sum(
                                            *sum_input)<=hom_eps, 'hom_r0_'+
                                                    str(s1)+'_'+str(t1)+'_'+str(tau)+'_'+str(ind_1_x)+'_'+str(
                                                    ind_2_x)+'_'+str(ind_2_y))

        print('Done!')
    if hom == 2:
        import itertools
        print('Adding multi-homogeneity constraints...')
        if not Delta:
            Delta = []
            for k in range(1, t-1):
                for s1 in range(t-k-1):
                    for t1 in range(s1+1, t-k):
                        for tau_list in itertools.combinations(range(1, t-t1), k):
                            Delta.append((s1, t1, tau_list))
            print('Setting Delta:')
            print(Delta)
        for d_ind in range(d):
            for (s1, t1, tau_list) in Delta:
                for ind_1_x in range(ind_gu[s1*d+d_ind]):
                    for ind_2_x in range(ind_gu[t1*d+d_ind]):
                        if margs[s1][d_ind][0][ind_1_x] == margs[t1][d_ind][0][ind_2_x]:
                            # at this point s and t are fix and the same

                            total_index = [[] for tau in tau_list]
                            ind_tau = 0
                            for tau in tau_list:
                                for ind_1_y in range(ind_gu[(s1 + tau) * d + d_ind]):
                                    for ind_2_y in range(ind_gu[(t1 + tau) * d + d_ind]):
                                        if margs[s1 + tau][d_ind][0][ind_1_y] == margs[t1 + tau][d_ind][0][ind_2_y]:
                                            total_index[ind_tau].append((ind_1_y, ind_2_y))
                                ind_tau += 1
                            # TODO: Check check double check

                            for combs in itertools.product(*total_index):
                                sum_input_lhs = gurobi_sum_input.copy()
                                sum_input_rhs = gurobi_sum_input.copy()
                                sum_input_lhs[s1 * d + d_ind] = ind_1_x
                                sum_input_rhs[t1 * d + d_ind] = ind_2_x

                                ind_tau = 0
                                for tau in tau_list:
                                    sum_input_lhs[(s1+tau)*d+d_ind] = combs[ind_tau][0]
                                    sum_input_rhs[(t1+tau)*d+d_ind] = combs[ind_tau][1]
                                    ind_tau += 1

                                lhs = pi_var.sum(*sum_input_lhs) * margs[t1][d_ind][1][ind_2_x]
                                rhs = pi_var.sum(*sum_input_rhs) * margs[s1][d_ind][1][ind_1_x]
                                m.addConstr(-hom_eps <= lhs - rhs <= hom_eps,
                                            'hom_' + '_'.join([str(bs) for bs in sum_input_lhs]) + '__' +
                                            '_'.join([str(bs) for bs in sum_input_rhs]))

                            # TODO: Constraint for when there is support missing
        print('Done!')
    # Specify objective function
    print('Setting objective function...')
    obj = LinExpr()
    for index in np.ndindex(ind):
        obj += cost_mat[index] * pi_var[index]

    if minmax == 'min':
        m.setObjective(obj, GRB.MINIMIZE)
    else:
        m.setObjective(obj, GRB.MAXIMIZE)

    print('Done! Ready to solve:')
    m.optimize()
    objective_val = m.ObjVal
    optimizer = np.zeros(ind)
    for index in np.ndindex(ind):
        optimizer[index] = pi_var[index].x

    return objective_val, optimizer


def brute_force_wasserstein_ball(barmu, mu_margs, rho, cost_fun, f_np, minmax='min', return_opti=0):
    # f_np should take [d] shape input
    # cost_fun should take two [d] shape inputs --> evaluated at (mu_bar, mu)!
    d = len(mu_margs)

    m = Model('Primal')
    ind = [int(len(barmu[1]))]
    for marg in mu_margs:
        ind.append(int(len(marg[1])))
    ind = tuple(ind)
    pi_var = m.addVars(*ind, lb=0, name='pi_var')

    obj = 0
    input_arr = np.zeros(d)
    cv = 0
    for index in np.ndindex(ind):
        for d_ind in range(d):
            input_arr[d_ind] = mu_margs[d_ind][0][index[1+d_ind]]
        obj += f_np(input_arr) * pi_var[index]
        xval = mu_margs[0][0][index[0]]
        cv += cost_fun(xval, input_arr) * pi_var[index]
    m.addConstr(cv <= rho, name='ineq')

    gurobi_sum_input = ['*' for i in range(1+d)]
    for i_td in range(ind[0]):
        sum_input = gurobi_sum_input.copy()
        sum_input[0] = i_td
        m.addConstr(pi_var.sum(*sum_input) == barmu[1][i_td], 'barmu'+'_'+str(i_td))

    for d_ind in range(d):
        for i_td in range(ind[d_ind]):
            sum_input = gurobi_sum_input.copy()
            sum_input[1+d_ind] = i_td
            m.addConstr(pi_var.sum(*sum_input) == mu_margs[d_ind][1][i_td], 'marg_'+str(d_ind)+'_'+str(i_td))

    if minmax == 'min':
        m.setObjective(obj, GRB.MINIMIZE)
    else:
        m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()
    objective_val = m.ObjVal
    if not return_opti:
        return objective_val
    else:
        opti = np.zeros(ind)
        cost_fun_vals = []
        cor_opti_vals = []
        f_vals = []
        th = 500/np.prod(ind)
        for index in np.ndindex(ind):
            opti[index] = pi_var[index].x
            if opti[index] >= th:
                for d_ind in range(d):
                    input_arr[d_ind] = mu_margs[d_ind][0][index[1 + d_ind]]
                f_vals.append(f_np(input_arr))
                xval = mu_margs[0][0][index[0]]
                cost_fun_vals.append(cost_fun(xval, input_arr))
                cor_opti_vals.append(opti[index])
        return objective_val, opti, cost_fun_vals, f_vals, cor_opti_vals


if __name__ == '__main__':
    # # u quantization_test:
    # def sampler(batch_size):
    #     while 1:
    #         yield np.random.random_sample(batch_size)
    # def q_fun(x):
    #     return x
    # (x, p) = u_quantization(p_samp=sampler)
    # print(x)
    # (x, p) = u_quantization(p_quant=q_fun, eps=0)  # eps can be set to zero as q_fun is defined at 0 and 1
    # #  and 1
    # print(x)

    # LP MMOT test:
    T = 2
    d = 2
    p = 2
    def f_spread(x):
        return np.abs(x[T-1, 0] - x[T-1, 1]) ** p

    K = 0
    def f_basket(x):
        return np.maximum(x[T-1, 0] + x[T-1, 1] - K, 0)

    # # marginals spread:
    # n = 10  # p = 3, n = 10, 11, 12, 13, 14, 15 values 31.1573, 31.1506, 31.1436, 31.1398, 31.1359, 31.1331
    # w = 1/(2*n) * np.ones(2*n+1)
    # w[0] = 1/(4*n)
    # w[-1] = 1/(4*n)
    # margs_00 = [np.linspace(-1, 1, 2*n+1), w]
    # margs_01 = [np.linspace(-1, 1, 2*n+1), w]
    # w2 = 1/(4*n) * np.ones(4*n+1)
    # w2[0] = 1/(8*n)
    # w2[-1] = 1/(8*n)
    # margs_11 = [np.linspace(-2, 2, 4*n+1), w2]
    # w3 = 1/(6*n) * np.ones(6*n+1)
    # w3[0] = 1/(12*n)
    # w3[-1] = 1/(12*n)
    # margs_10 = [np.linspace(-3, 3, 6*n+1), w3]

    # marginals basket:
    n = 5  # p = 3, n = 10, 11, 12, 13, 14, 15 values 31.1573, 31.1506, 31.1436, 31.1398, 31.1359, 31.1331
    w = 1/(2*n) * np.ones(2*n+1)
    w[0] = 1/(4*n)
    w[-1] = 1/(4*n)
    margs_01 = [np.linspace(-1, 1, 2*n+1), w]
    w2 = 1/(4*n) * np.ones(4*n+1)
    w2[0] = 1/(8*n)
    w2[-1] = 1/(8*n)
    margs_00 = [np.linspace(-2, 2, 4*n+1), w2]
    w3 = 1/(6*n) * np.ones(6*n+1)
    w3[0] = 1/(12*n)
    w3[-1] = 1/(12*n)
    margs_11 = [np.linspace(-3, 3, 6*n+1), w3]
    margs_10 = [np.linspace(-3, 3, 6*n+1), w3]


    margs = [[margs_00, margs_01], [margs_10, margs_11]]
    val, opti = lp_hom_mmot(margs, f_basket, minmax='min', mart=1, hom=0, mmot_eps=0)
    print(opti.shape)
    points_1 = margs_11[0]
    points_2 = margs_10[0]
    opti_2 = np.sum(opti, axis=(0, 1))
    visualize_LP_sol(points_1, points_2, opti_2, name='LP_K_0_min')


    # # Wasserstein ball brute force test:
    # N_mu_bar = 50
    # N_margs = 50
    # mu_bar_x = np.zeros([N_mu_bar, 2])
    # mu_bar_x[:, 0] = np.linspace(1/(2*N_mu_bar), 1-(1/(2*N_mu_bar)), N_mu_bar)
    # mu_bar_x[:, 1] = mu_bar_x[:, 0]
    # mu_bar = (mu_bar_x, 1/N_mu_bar * np.ones(N_mu_bar))
    # print(mu_bar_x[:, 0])
    # each_marg = (np.linspace(1/(2*N_margs), 1-(1/(2*N_margs)), N_margs), 1/N_margs * np.ones(N_margs))
    # print(each_marg[0])
    # mu_margs = [each_marg, each_marg]
    # def f_np(x):
    #     return np.sin(x[1] + x[0]) * np.cos(x[0])**2
    # def cost_fun(x, y):
    #     return np.sqrt(np.sum(np.abs(x-y)**2))
    # def cost_fun2(x, y):
    #     return np.sum(np.abs(x-y)**2)
    # obj_vals = []
    # obj_vals2 = []
    # for rho in np.linspace(0.3, 0.3, 1):
    #     opti_val, opti, cfvs, fvs, covs = brute_force_wasserstein_ball(mu_bar, mu_margs, rho, cost_fun, f_np,
    #                                                                  minmax='max',
    #                                                   return_opti=1)
    #     obj_vals.append(opti_val)
    #     rel_opti = np.sum(opti, axis=0)
    #     print(cfvs)
    # for rho in np.linspace(0.3, 0.3, 1):
    #     opti_val2, opti2, cfvs2, fvs2, covs2 = brute_force_wasserstein_ball(mu_bar, mu_margs, rho**2, cost_fun2, f_np,
    #                                                               minmax='max', return_opti=1)
    #     obj_vals2.append(opti_val2)
    #     rel_opti2 = np.sum(opti2, axis=0)
    #     print(cfvs2)
    #
    # print(np.array(cfvs))
    # print(np.array(covs))
    # print(np.array(cfvs2)**0.5)
    # print(np.array(covs2))
    #
    # print(np.array(cfvs)*np.array(covs))
    # print((np.array(cfvs2)**0.5) * np.array(covs2))
    #
    # print(np.sort(np.array(cfvs)))
    # print(np.sort(np.array(cfvs2)**0.5))
    #
    # visualize_LP_sol(each_marg[0], each_marg[0], rel_opti)
    # visualize_LP_sol(each_marg[0], each_marg[0], rel_opti2)

    # print(np.sort(np.array(cfvs))-np.sort(np.array(cfvs2)**5))

    # print(obj_vals)
    # import matplotlib.pyplot as plt
    # plt.plot(np.linspace(0, 0.6, 25), obj_vals)
    # plt.plot(np.linspace(0, 0.6, 25), obj_vals2)
    # plt.show()

    # # LP Hom Test:
    # from scipy.stats import binom
    # T_MAX = 10
    # T = 10
    # d = 1
    # K = 1
    # def f_forward(x):
    #     return np.maximum(x[-1, 0] - K * x[-2, 0], 0)
    # def f_asian(x):
    #     return np.maximum(np.sum(x)/T - K, 0)
    # margs = []
    # n_vars = 1
    # for t in range(T_MAX-T, T_MAX):
    #     n_vars *= t+1
    #     margs.append([[np.linspace(100-t, 100+t, t+1), np.ones(t+1) * 1/(t+1)]])
    # print(n_vars)
    # ov, opti = lp_hom_mmot(margs, f_forward, minmax='min', hom=1, mart=1)
    # ov2, opti2 = lp_hom_mmot(margs, f_forward, minmax='max', hom=1, mart=1)

    # from scipy.stats import binom
    # T = 2
    # d = 1
    # K = 1
    # def f_forward(x):
    #     return np.maximum(x[-1, 0] - K * x[-2, 0], 0)
    # margs = [[[[-1, 1], [0.5, 0.5]]],[[[-3, -1, 1, 3], [1/8, 3/8, 3/8, 1/8]]]]
    # ov, opti = lp_hom_mmot(margs, f_forward, minmax='min', hom=1, mart=1)
    # print(opti)
    # ov2, opti2 = lp_hom_mmot(margs, f_forward, minmax='max', hom=1, mart=1)
    # print(opti)

    # print(opti2)
    # wannabe gromov wasserstein --> doesn't work at the moment
    # mu1 = [[-1, 1], [0.5, 0.5]]
    # mu2 = [[-2, 2], [0.5, 0.5]]
    # def c_1(x, xp):
    #     return np.abs(x-xp)
    # def c_2(y, yp):
    #     return (y-yp)**2
    # gromov_wasserstein(mu1, mu2, c_1, c_2)

    # # LP Hom Test 2:
    # T = 3
    # d = 1
    # K = 1
    # def f_forward(x):
    #     return np.maximum(x[2, 0] - K * x[0, 0], 0)
    # margs = []
    # n_vars = 0
    # for t in range(T):
    #     n_vars += 2
    #     margs.append([[np.array([0, 1]), 1/2 * np.ones(2)]])
    # print(n_vars)
    # ov, opti = lp_hom_mmot(margs, f_forward, minmax='max', hom=2, mart=0)
    # print(opti)

    # # LP Hom Test 3:
    # T = 3
    # d = 1
    # K = 1
    # def f_forward(x):
    #     return np.maximum(x[2, 0] - K * x[0, 0], 0)
    # margs = [[[[10], [1]]], [[[9, 10, 11], [1/3, 1/3, 1/3]]], [[[8, 9, 10, 11, 12], [1/9, 2/9, 1/3, 2/9, 1/9]]]]
    # ov, opti = lp_hom_mmot(margs, f_forward, minmax='min', hom=2, mart=0)
    # print(opti)

