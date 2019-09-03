import numpy as np
import random
from scipy.stats import multivariate_normal


def simple_random_mixture(k=2, t=3, d=1, leftmost=-1, rightmost=1, den=0, each_dim_sep=0, hom_mc=0, p_given=0,
                          sigs_given=0, mus_given=0):
    """
    function to sample mixtures of normal distributions. for increasing t the 1dim marginals are in increasing convex
    order
    :param k:
    :param t:
    :param d:
    :param leftmost:
    :param rightmost:
    :param den:
    :param each_dim_sep:
    :param hom_mc:
    :param p_given:
    :param sigs_given:
    :param mus_given:
    :return: if den=0, returns a generator which takes batch_size as input and generates samples of size
    [batch_size, t, d]
    if den=1, returns a tuple with first entry as for den=0, and second entry a function which takes input (x, t,
    d) and produces the marginal densities for the corresponding samples the generator produces
    """
    p_list = []
    for i in range(t):
        p_here = np.ones(k)/k
        p_list.append(p_here)

    mus = []
    sigs = []
    np.random.seed(0)
    if hom_mc == 0:
        sig_list_1 = np.random.random_sample([t, d, k])
        sig_list_1 = np.cumsum(sig_list_1, axis=0)
    else:
        sig_list_1 = np.zeros([t, d, k])
        sigs0 = np.random.random_sample([d, k])
        inc_sigs = np.random.random_sample([d, k])
        sig_list_1[0, :, :] = sigs0
        for t_ind in range(1, t):
            sig_list_1[t_ind, :, :] = sig_list_1[t_ind-1, :, :] + inc_sigs
    np.random.seed(round(1000000 * random.random()))

    means_0 = np.linspace(leftmost, rightmost, k)
    for j in range(t):
        means = []
        sig_list = []
        for i in range(k):
            means.append(np.ones([d]) * means_0[i])
            sig_list.append(np.diag(sig_list_1[j, :, i]))
        mus.append(means)
        sigs.append(sig_list)

    if p_given:
        p_list = p_given
    if sigs_given:
        sigs = sigs_given
    if mus_given:
        mus = mus_given

    if den == 0:
        return mvar_normal_mixture(ps=p_list, mus=mus, sigs=sigs, k=k)
    else:
        gen_fun = mvar_normal_mixture(ps=p_list, mus=mus, sigs=sigs, k=k)
        density_fun = mvar_normal_mixture_den(ps=p_list, mus=mus, sigs=sigs, k=k, each_dim_sep=each_dim_sep)
        return((gen_fun, density_fun))


def simple_random_mixture_mt(k=2, t=3, d=1, leftmost=-1, rightmost=1, den=0, each_dim_sep=0, hom_mc=0):
    # same as simple_random_mixture but samples from a random walk instead of independently
    p_list = []
    for i in range(t):
        p_here = np.ones(k)/k
        p_list.append(p_here)

    mus = []
    sigs = []
    np.random.seed(0)
    if hom_mc == 0:
        sig_list_1 = np.random.random_sample([t, d, k])
        sig_list_1 = np.cumsum(sig_list_1, axis=0)
    else:
        sig_list_1 = np.zeros([t, d, k])
        sigs0 = np.random.random_sample([d, k])
        inc_sigs = np.random.random_sample([d, k])
        sig_list_1[0, :, :] = sigs0
        for t_ind in range(1, t):
            sig_list_1[t_ind, :, :] = sig_list_1[t_ind-1, :, :] + inc_sigs
    np.random.seed(round(1000000 * random.random()))

    means_0 = np.linspace(leftmost, rightmost, k)
    for j in range(t):
        means = []
        sig_list = []
        for i in range(k):
            means.append(np.ones([d]) * means_0[i])
            sig_list.append(np.diag(sig_list_1[j, :, i]))
        mus.append(means)
        sigs.append(sig_list)

    npz = np.zeros(d)
    def gen_fun(batch_size):
        while 1:
            data = np.zeros([batch_size, t, d])
            for bs in range(batch_size):
                ind = np.random.choice(range(k), p=p_list[0])
                data[bs, 0, :] = np.random.multivariate_normal(mus[0][ind], sigs[0][ind], 1)
                for i in range(1, t):
                    data[bs, i, :] = data[bs, i-1, :] + np.random.multivariate_normal(npz, sigs[i][ind] -
                                                                                      sigs[i-1][ind], 1)
            yield data
    return gen_fun


def mvar_normal_mixture_den(ps=((1, ), ), mus=(((0,), ), ), sigs=((np.reshape(np.array([1]), [1, 1]), ), ), k=1,
                            each_dim_sep=0):
    """

    :param ps:
    :param mus:
    :param sigs: TODO: Currently for d=1 its standard deviation and for d>1 its covariances! (easy to fix,
    but I don't want to change custom examples at the moment...)
    :param k:
    :return: a generator function that takes batchsize as input and produces output of shape [batchsize, t, d]
    """
    t = len(mus)
    if hasattr(mus[0][0], "__len__"):
        d = len(mus[0][0])
    else:
        d = 1

    sh = []
    if d == 1:
        for i in range(t):
            sh.append(np.array(sigs[i]) ** 2)
    else:
        sh = sigs.copy()

    if each_dim_sep == 0:
        def density_fun(x, t_ind):
            # x should be [size, d]
            den = np.zeros(len(x))
            for kind in range(k):
                den += ps[t_ind][kind] * multivariate_normal.pdf(x, mean=mus[t_ind][kind], cov=sh[t_ind][kind])
            return den
    else:
        def density_fun(x, t_ind, d):
            # x should be [size]
            den = np.zeros(len(x))
            for kind in range(k):
                den += ps[t_ind][kind] * multivariate_normal.pdf(x, mean=mus[t_ind][kind][d], cov=sh[t_ind][kind][d, d])
            return den

    return density_fun


def mvar_normal_mixture(ps=((1, ), ), mus=(((0,), ), ), sigs=((np.reshape(np.array([1]), [1, 1]), ), ), k=1):
    """

    :param ps:
    :param mus:
    :param sigs:
    :param k:
    :return: a generator function that takes batchsize as input and produces output of shape [batchsize, t, d]
    """
    t = len(mus)
    if hasattr(mus[0][0], "__len__"):
        d = len(mus[0][0])
    else:
        d = 1

    psc = []
    for i in range(t):
        csp = np.cumsum(ps[i])
        csp = np.concatenate(([0], csp))
        psc.append(csp)

    sh = []
    if d == 1:
        for i in range(t):
            sh.append(np.array(sigs[i]) ** 2)
    else:
        sh = sigs.copy()

    def gen_fun(batch_size):
        while 1:
            dataset = np.zeros([batch_size, t, d])
            for i in range(t):
                sel_idx = np.random.random_sample(batch_size)
                for kind in range(k):
                    idx = (sel_idx > psc[i][kind]) * (sel_idx < psc[i][kind+1])
                    ksamples = np.sum(idx)
                    dataset[idx, i, :] = np.random.multivariate_normal(mus[i][kind], sh[i][kind], ksamples)
            yield dataset
    return gen_fun


if __name__ == '__main__':
    # Tests:
    # Test 1:
    # gf = mvar_normal_mixture()
    # gen_gf = gf(10)
    # print(next(gen_gf))
    # print(next(gen_gf).shape)

    # Test 2:
    def f_obj(s):
        return np.maximum(s[:, 2, 0] - s[:, 1, 0], 0)

    T = 3
    d = 1
    K = 3

    sf = simple_random_mixture_mt(K, T, d, den=1, each_dim_sep=1, hom_mc=1)
    gen_sf = sf(20000)
    samp = next(gen_sf)
    print(np.mean(f_obj(samp)))
    import matplotlib.pyplot as plt
    plt.hist(samp[:, 0, 0], bins=100, normed=1)
    den_grid = np.linspace(-4, 4, 1000)
    # den_val = denf(den_grid, 0, 0)
    # plt.plot(den_grid, den_val)
    plt.show()