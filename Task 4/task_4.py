import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

all_intervals = []

beta = 0.95
theta = 15
sample = np.around(np.random.uniform(low = theta, high = 2 * theta, size=100), 3)
print(sample)

#Точный доверительный интервал
x_max = np.max(sample)
t1 = round(x_max / (((1 + beta)/2) ** (1 / np.size(sample)) + 1), 3)
t2 = round(x_max / (((1 - beta)/2) ** (1 / np.size(sample)) + 1), 3)
all_intervals.append((t1, t2))
print(f'Exact Confidence Interval = ({t1}, {t2})')


#Асимптотический доверительный интервал
n = np.size(sample)
mean = np.mean(sample)
pow_mean = np.dot(sample, sample) / n

v1 = 2**0.5 * sp.special.erfinv(-beta)
v2 = -v1
t1 = round(3/2 * v1 * ((pow_mean-mean**2)/n)**0.5 + 2/3*mean, 3)
t2 = round(3/2 * v2 * ((pow_mean-mean**2)/n)**0.5 + 2/3*mean, 3)
all_intervals.append((t1, t2))
print(f'Asymptotic Confidence Interval = ({t1}, {t2})')


#Бутстраповский непараметрический доверительный интервал для theta_1
theta_est = 2/3 * np.mean(sample)
deltas = []
for i in range(1000):
    sample_i = np.random.choice(sample, size=np.size(sample))
    deltas.append(2/3 * np.mean(sample_i) - theta_est)
k1 = int((1 - beta) / 2 * 1000) - 1
k2 = int((1 + beta) / 2 * 1000) - 1
deltas.sort()
t1 = round(theta_est - deltas[k2], 3)
t2 = round(theta_est - deltas[k1], 3)
all_intervals.append((t1, t2))
print(f'Bootstrap Confidence Interval for theta_1 = ({t1}, {t2})')


#Бутстраповский параметрический доверительный интервал для theta_1
theta_est = 2/3 * np.mean(sample)
thetas = []
for i in range(10000):
    sample_i = np.random.uniform(low = theta, high = 2 * theta, size=100)
    thetas.append(2/3 * np.mean(sample_i))
thetas.sort()
k1 = int((1 - beta) / 2 * 10000) - 1
k2 = int((1 + beta) / 2 * 10000) - 1
t1 = round(thetas[k1], 3)
t2 = round(thetas[k2], 3)
all_intervals.append((t1, t2))
print(f'Bootstrap With Parameter Confidence Interval for theta_1 = ({t1}, {t2})')


#Бутстраповский непараметрический доверительный интервал для theta_2
theta_est = max(sample) / 2
deltas = []
for i in range(1000):
    sample_i = np.random.choice(sample, size=np.size(sample))
    deltas.append(max(sample_i) / 2 - theta_est)
k1 = int((1 - beta) / 2 * 1000) - 1
k2 = int((1 + beta) / 2 * 1000) - 1
deltas.sort()
t1 = round(theta_est - deltas[k2], 3)
t2 = round(theta_est - deltas[k1], 3)
all_intervals.append((t1, t2))
print(f'Bootstrap Confidence Interval for theta_2 = ({t1}, {t2})')


#Бутстраповский параметрический доверительный интервал для theta_2
theta_est = max(sample) / 2
thetas = []
for i in range(10000):
    sample_i = np.random.uniform(low = theta, high = 2 * theta, size=100)
    thetas.append(max(sample_i) / 2)
thetas.sort()
k1 = int((1 - beta) / 2 * 10000) - 1
k2 = int((1 + beta) / 2 * 10000) - 1
t1 = round(thetas[k1], 3)
t2 = round(thetas[k2], 3)
all_intervals.append((t1, t2))
print(f'Bootstrap With Parameter Confidence Interval for theta_2 = ({t1}, {t2})')


#Сравнение интервалов
plt.plot(all_intervals[0], (0, 0), label= "Точный доверительный интервал")
plt.plot(all_intervals[1], (1, 1), label= "Асимптотический доверительный интервал")
plt.plot(all_intervals[2], (2, 2), label= "Бутстраповский непараметрический доверительный интервал для theta_1, OMM")
plt.plot(all_intervals[3], (3, 3), label = "Бутстраповский параметрический доверительный интервал для theta_1, OMM")
plt.plot(all_intervals[4], (4, 4), label= "Бутстраповский непараметрический доверительный интервал для theta_2, OMP")
plt.plot(all_intervals[5], (5, 5), label= "Бутстраповский параметрический доверительный интервал для theta_2, OMP")
plt.legend(bbox_to_anchor=(1.005, 1), loc='upper left')
plt.show()