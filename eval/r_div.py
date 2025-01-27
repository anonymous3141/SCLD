import numpy as np
from scipy import stats

bandwidth = "scott"


def get_kde_estimates(bandwidth, data, data_test):
    kernel = stats.gaussian_kde(data.T, bandwidth)
    return kernel.evaluate(data_test.T)


def r_div(X, Y):
    XY = np.concatenate([X, Y], axis=0)
    logprob_1 = np.log(get_kde_estimates(bandwidth, XY, X)).mean()
    logprob_2 = np.log(get_kde_estimates(bandwidth, XY, Y)).mean()
    rd = abs(logprob_1 - logprob_2)
    return rd


def js_div(X, Y):
    XY = np.concatenate([X, Y], axis=0)
    logprob_pq = np.log(get_kde_estimates(bandwidth, XY, X)).mean()
    logprob_p = np.log(get_kde_estimates(bandwidth, X, X)).mean()
    logprob_qp = np.log(get_kde_estimates(bandwidth, XY, Y)).mean()
    logprob_q = np.log(get_kde_estimates(bandwidth, Y, Y)).mean()
    jsd = logprob_p + logprob_q - (logprob_pq + logprob_qp)
    return jsd
