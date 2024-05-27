import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import norm 
import statistics
import matplotlib as mpl

from sklearn.mixture import GaussianMixture

mpl.use('TkAgg')

from scipy.stats import norm


dist_1 = np.random.normal(10, 3, 1000)
dist_2 = np.random.normal(30, 5, 4000)
dist_3 = np.random.normal(45, 6, 500)

multimodal_dist = np.concatenate((dist_1, dist_2, dist_3), axis=0)





fig, axes = plt.subplots(nrows=3, ncols=1, sharex='col', figsize=(6.4, 7))



axes[1].hist(multimodal_dist, bins=50, alpha=0.5)

x = np.linspace(min(multimodal_dist), max(multimodal_dist), 100)

for mean, covariance, weight in zip(means, standard_deviations, weights):
    pdf = weight * norm.pdf(x, mean, std)
    plt.plot(x.reshape(-1, 1), pdf.reshape(-1, 1), alpha=0.5)

plt.show()