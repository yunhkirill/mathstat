
import numpy as np
import statistics
import scipy.stats as sps
import matplotlib.pyplot as plt
import math

sample = np.around(np.random.exponential(size=25), 3)
print(sample)
print("Mode = ", sps.mode(sample)[0])
print("Median = ", statistics.median(sample))
print("Range of data (razmah) = ", max(sample) - min(sample))
print("Skewness (asymmetry coefficient) = ", sps.skew(sample))

#Построение эмпирической функциии распределения
sorted_data = np.sort(sample)
n = len(sorted_data)
ecdf_y = np.arange(1, n + 1) / n
plt.step(sorted_data, ecdf_y, where='post')
plt.xlabel('Значение')
plt.ylabel('ЭФР')
plt.title('Эмпирическая функция распределения')
plt.show()


#Построение гистограммы
plt.hist(sample, bins=30, density=True, alpha=0.6, color='g', edgecolor='black')
plt.title('Гистограмма')
plt.xlabel('Значение')
plt.ylabel('Плотность')
plt.show()


#Построение boxplot
plt.boxplot(sample, vert=False)
plt.title('Boxplot')
plt.xlabel('Значение')
plt.yticks([])
plt.show()


#Сравнение теор. плотности распр. среднего арифметического с бустраповской оценкой
x = np.arange(0, 2.01, 0.01)
plt.plot(x, (sample.size**sample.size)/math.factorial(sample.size - 1) * x**(sample.size - 1) * np.e**(-sample.size*x))
data = np.random.exponential(size=1000)
num_bootstrap_samples = 1000
bootstrap_means = []
for _ in range(num_bootstrap_samples):
    bootstrap_sample = np.random.choice(data, size=sample.size, replace=True)
    bootstrap_mean = np.mean(bootstrap_sample)
    bootstrap_means.append(bootstrap_mean)
plt.hist(bootstrap_means, bins=25, density=True, alpha=0.6, color='g', edgecolor='black')
plt.show()


#Построение бутстрэповской оценки плотности распределения коэффициента асимметрии
def compute_skewness(data):
    n = len(data)
    mean = np.mean(data)
    skewness = (1/n) * np.sum((data - mean)**3) / np.std(data)**3
    return skewness

sample_size = 100
data = np.random.exponential(size=sample_size)
num_bootstrap_samples = 1000
bootstrap_skewness = []
for _ in range(num_bootstrap_samples):
    bootstrap_sample = np.random.choice(data, size=sample_size, replace=True)
    bootstrap_skewness.append(compute_skewness(bootstrap_sample))
hist, bins = np.histogram(bootstrap_skewness, bins=30, density=True)
widths = np.diff(bins)
max_density = np.max(hist)
hist_normalized = hist / max_density
plt.bar(bins[:-1], hist_normalized, width=widths, color='g', alpha=0.6, edgecolor='black')
plt.title('Бутстрэповская оценка плотности распределения коэффициента асимметрии')
plt.xlabel('Коэффициент асимметрии')
plt.ylabel('Плотность')
plt.show()