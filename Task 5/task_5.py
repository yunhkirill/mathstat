import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random



all_intervals = []

beta = 0.95
theta = 15
sample = np.around((1 - np.random.rand(100)) ** (1/(1 - theta)), 3)
print(sample)

#Доверительный интервал для медианы
waved_theta = 1 + np.size(sample)/(np.sum(np.log(sample)))
t1 = round((-1.96 * np.log(2) * (2 ** (1 / (waved_theta - 1)))) / ((waved_theta - 1)  * 100 ** 0.5) + 2 ** (1 / (waved_theta - 1)), 3)
t2 = round((1.96 * np.log(2) * (2 ** (1 / (waved_theta - 1)))) / ((waved_theta - 1)  * 100 ** 0.5) + 2 ** (1 / (waved_theta - 1)), 3)
all_intervals.append((t1, t2))
print(f'Confidence Interval for median = ({t1}, {t2})')


#Асимптотический доверительный интервал
waved_theta = 1 + np.size(sample)/(np.sum(np.log(sample)))
t1 = round(-1.96 * (waved_theta - 1) / (100 ** 0.5) + waved_theta, 3)
t2 = round(1.96 * (waved_theta - 1) / (100 ** 0.5) + waved_theta, 3)
all_intervals.append((t1, t2))
print(f'Asymptotic Confidence Interval = ({t1}, {t2})')


#Бутстраповский непараметрический доверительный интервал
real_theta = 1 + np.size(sample)/(np.sum(np.log(sample)))
deltas = []
for i in range(1000):
    sample_i = np.random.choice(sample, size=np.size(sample))
    deltas.append(1 + np.size(sample_i)/(np.sum(np.log(sample_i))) - real_theta)
k1 = int((1 - beta) / 2 * 1000) - 1
k2 = int((1 + beta) / 2 * 1000) - 1
deltas.sort()
t1 = round(real_theta - deltas[k2], 3)
t2 = round(real_theta - deltas[k1], 3)
all_intervals.append((t1, t2))
print(f'Bootstrap Confidence Interval = ({t1}, {t2})')


#Бутстраповский параметрический доверительный интервал
real_theta = 1 + np.size(sample)/(np.sum(np.log(sample)))
_theta = real_theta
thetas = []
for i in range(10000):
    sample_i = np.random.uniform(low = _theta, high = 2 * _theta, size=100)
    thetas.append(1 + np.size(sample_i)/(np.sum(np.log(sample_i))))
thetas.sort()
k1 = int((1 - beta) / 2 * 10000) - 1
k2 = int((1 + beta) / 2 * 10000) - 1
t1 = round(thetas[k1], 3)
t2 = round(thetas[k2], 3)
all_intervals.append((t1, t2))
print(f'Bootstrap With Parameter Confidence Interval = ({t1}, {t2})')


#Сравнение интервалов
plt.plot(all_intervals[0], (0, 0), label= "Доверительный интервал для медианы")
plt.plot(all_intervals[1], (1, 1), label= "Асимптотический доверительный интервал")
plt.plot(all_intervals[2], (2, 2), label= "Бутстраповский непараметрический доверительный интервал")
plt.plot(all_intervals[3], (3, 3), label= 'Бутстраповский параметрический доверительный интервал')
plt.legend(bbox_to_anchor=(1.005, 1), loc='upper left')
plt.show()