import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from numpy.linalg import inv, qr


# creating ndarray
lyst = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]]
print(lyst)
arr = np.array(lyst)
print(arr)
zero = np.zeros((2, 3))
print(zero)
range_row = np.arange(5)
print(range_row)

# data Type for ndarray
arr = np.array(np.arange(4), dtype=np.int)
print(arr)
arr1 = arr.astype(np.float)
print(arr1)

# operations between array and scalar
arr = np.array(np.arange(6).reshape((2, 3)), dtype=np.float)
arr += 1
print(arr)
arr1 = 1/arr
print(arr1)
arr2 = arr**2
print(arr2)

# indexing and slicing
arr = np.arange(10)
print(arr)
arr_slice = arr[5:8]
arr_slice[:] = 12
print(arr)
arr_slice = arr[5:8].copy()
arr_slice[:] = 64
print(arr)
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr2d[2])
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(arr3d[0])

# boolean indexing
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
print(data[names == 'Bob'])
data[data < 0] = 0
print(data)

# transposing array and swapping axes
arr = np.arange(15).reshape((3, 5))
print(arr)
print(arr.T)
arr = np.arange(16).reshape((2, 2, 4))
print(arr)
print(arr.transpose(0, 2, 1))
print(arr.swapaxes(1, 2))
print(arr.swapaxes(0, 2))

# fast element-wise array functions
arr = np.arange(10)
print(np.sqrt(arr))
print(np.exp(arr))
print(np.max(arr))

# data processing using array
points = np.arange(-5, 5, 0.2)
xs, ys = np.meshgrid(points, points)
z = np.sqrt(xs**2 + 3*ys**2)
plt.contour(xs, ys, z)
plt.show()

# Expressing Conditional Logic as Array
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
result = np.where(cond, xarr, yarr)
print(result)

# mathematical and statstical Methods
arr = np.random.randn(2, 3)
print(arr)
print(arr.mean(axis=1))
print(arr.sum(axis=1))

# Methods for Boolean Arrays
bools = np.array([False, False, True, False])
print(bools.any())
print(bools.all())

# sorting
arr = np.random.randn(4, 3)
print(arr)
arr.sort(axis=1)
print(arr)

# unique and other set logic
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
print(np.unique(names))
print(sorted(set(names)))

# file input and output with arrays
arr = np.arange(10)
np.save('some_array', arr)
arr1 = np.load('some_array.npy')
print(arr1)

np.savez('array_archive.npz', a=arr, b=arr+1)
arch = np.load('array_archive.npz')
print(arch['a'])
print(arch['b'])

# saving and loading text files
arr = np.loadtxt('arr_text.txt', delimiter=',')
print(arr)

# linear algebra
X = np.random.randn(3, 3)
mat = X.T.dot(X)
print(mat)
x_inv = inv(mat)
print(x_inv)

q, r = qr(mat)
print(q)
print(r)

# random number generation
samples = []
mu = [2, 1]
sigama = [[3, 0],
          [0, 3]]
z = []
for _ in range(50):
    sample = np.random.multivariate_normal(mean=mu, cov=sigama)
    samples.append(sample)

samples = np.array(samples)

x1_min, x1_max = samples[:, 0].min() - 1, samples[:, 0].max() + 1
x2_min, x2_max = samples[:, 1].min() - 1, samples[:, 1].max() + 1
xs, ys = np.meshgrid(np.arange(x1_min, x1_max, 0.1), np.arange(x2_min, x2_max, 0.5))

z = np.array([xs.ravel(), ys.ravel()]).T
zs = np.diag(np.exp(-(z - mu).dot(inv(sigama)).dot((z - mu).T)))
zs = zs.reshape(xs.shape)
plt.contour(xs, ys, zs)
plt.scatter(samples[:, 0], samples[:, 1])
plt.show()


# random walk
nwalks = 5000
nsteps = 1000
draws = np.random.randint(0, 2, size=(nwalks, nsteps))
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(1)
print(walks)
