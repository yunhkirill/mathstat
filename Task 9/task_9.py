import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, norm

def max_likelihood_function(sample_):
    def likelihood_function(variables):
        mean, sigma = variables
        counts = []
        for i in range(len(intervals) - 1):
            a, b = intervals[i], intervals[i+1]
            counts = np.append(counts, np.count_nonzero((sample_ < b) & (sample_ >= a)))
        return -np.prod((norm(mean, sigma).cdf(intervals[1:]) - norm(mean, sigma).cdf(intervals[:-1])) ** counts)
    
    
    initial_guess = [5.0, 3.0]
    result = minimize(likelihood_function, initial_guess, method='Nelder-Mead')
    return np.round(result.x, 3)

class EmpiricalDistribution:
    def __init__(self, sample):
        x, y = np.unique(sample, return_counts=True)
        y = (y/np.sum(y)).cumsum()
        y = np.hstack((np.array([0.]), y))
        assert len(x) == len(y) - 1
        self.x = np.array(x).astype(float)
        self.y = np.array(y).astype(float)
    
    def cdf(self, x):
        if x <= np.min(self.x):
            return 0.
        elif x > np.max(self.x):
            return 1.
        return self.y[np.where((self.x >= x) == True)[0][0]]
    
class EstimatedDistribution:
    def __init__(self, sample):
        self.mean, self.sigma  = max_likelihood_function(sample)
        self.params = (self.mean, self.sigma)
        self.norm = norm(*self.params)
        
    def cdf(self, x):
        return self.norm.cdf(x)

def delta_est_Kolmogorov(F_est, F_emp, N):
    def delta(x):
        return -abs(F_est.cdf(x) - F_emp.cdf(x))
    
    result = minimize(delta, [0], method='Nelder-Mead')
    return N**0.5 * -delta(result.x)[0]

def bootstrap_with_parameter(sample):
    n = np.size(sample)
    F_est = EstimatedDistribution(sample)
    F_emp = EmpiricalDistribution(sample)
    delta_est = delta_est_Kolmogorov(F_est, F_emp, N)
    params = F_est.mean, F_est.sigma
    
    deltas_est = []
    k = 0
    for _ in range(1_000):     
        sample_i = np.random.normal(*params, n)
        F_est_i = EstimatedDistribution(sample_i)
        F_emp_i = EmpiricalDistribution(sample_i)
        delta_est_i = delta_est_Kolmogorov(F_est_i, F_emp_i, N)
        k += delta_est_i > delta_est
        deltas_est.append(delta_est_i)

    p_value = k/1_000
    return delta_est, p_value

def plt_calc_bootstrap():
    delta_est, p_value = bootstrap_with_parameter(sample)
    print(f'Kolmagorov: p-value(delta > {np.round(delta_est, 3)} | H_0) = {np.round(p_value, 3)}')


data = np.arange(10)
data = np.vstack((data, [1/(np.max(data)-np.min(data))] * 10))
data = np.vstack((data, np.array([5, 8, 6, 12, 14, 18, 11, 6, 13, 7])))
N = int(np.sum(data[2]))
data = np.vstack((data, np.array([5, 8, 6, 12, 14, 18, 11, 6, 13, 7])/N))

intervals = np.hstack((np.array(-np.inf), data[0][1:], np.array(np.inf)))
sample = np.concatenate(np.array([[int(event)] * int(data[2][i]) for i, event in enumerate(data[0])], dtype=object)).astype(int)
max_mean, max_sigma = max_likelihood_function(sample)

probs = (norm(max_mean, max_sigma).cdf(intervals[1:]) - norm(max_mean, max_sigma).cdf(intervals[:-1]))
delta_est = np.sum((data[2] - N*probs)**2 / (N*probs))
k = 7

print(f'Pirson: p-value(delta > {np.round(delta_est, 3)} | H_0) = {np.round(chi2(k).sf(delta_est), 3)}')
plt_calc_bootstrap()
