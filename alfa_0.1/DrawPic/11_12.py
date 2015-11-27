print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
#from sklearn.linear_model import SGDClassifier

# we create 40 separable points
rng = np.random.RandomState(0)
n_samples_1 = 500
n_samples_2 = 100
X = np.r_[1.5 * rng.randn(n_samples_1, 2) + [-4,0],
          0.5 * rng.randn(n_samples_2, 2) + [1, 2]]

Xx = np.array([X[:,0]/4,X[:,1]])
Xx =  Xx.T


y = [0] * (n_samples_1) + [1] * (n_samples_2)

# fit the model and get the separating hyperplane


xx = np.linspace(-5, 5)
yy = -2 * xx + 2


wyy = -1 * xx + 1

# plot separating hyperplanes and samples
plt.figure(figsize=(8,8))
h0 = plt.plot(xx, yy, 'k-', label='consider without dif')
h1 = plt.plot(xx, wyy, 'k--', label='consider with dif')
s = [20*4 for n in range(len(y))]
plt.scatter(Xx[:500, 0], Xx[:500, 1], c='y', s=s[:500])
plt.scatter(Xx[500:, 0], Xx[500:, 1], c='r',marker='*',s=s[500:])
plt.legend()
plt.axis('tight')
plt.show()