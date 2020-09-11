import numpy as np
import cmath
import math
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


res = 1080
view = np.array([[-2, 1], [-1.5, 1.5]])


def function(z, c):
    return z**2 + c


def recursive(f, n):
    def f1(x, *args):
        for i in range(n):
            x = f(x, *args)
        return x
    return f1


l = []
density = np.zeros((res, res), dtype=np.float32)


cc = 0
pts = 10_000
its = 1500
for i in tqdm(range(pts)):
    z = np.random.rand(2) * (view[:, 1] - view[:, 0]) + view[:, 0]
    z = z[0] + 1j*z[1]
    c = z
    z = function(0, c)

    n = 0
    l1 = []
    while np.linalg.norm(z) < 4 and n < its:
        l1.append(np.array((z.real, z.imag, n)))
        z = function(z, c)
        n += 1
        cc += 1
    if np.linalg.norm(z) < 4:
        l.extend(l1)
print(cc, cc/(pts*its))


for z in tqdm(l):
    zz = z[0:2]
    it = z[2]
    n = res * (zz - view[:, 0]) / (view[:, 1] - view[:, 0])

    if (n >= res - 1).any() or (n < 0).any():
        continue

    x = math.floor(n[0])
    y = math.floor(n[1])

    if True:
        #density[x, y] += 1*math.sqrt(it)
        density[x, y] += 1
    else:
        xF = float(n[0]) % 1.0
        yF = float(n[1]) % 1.0
        density[x, y] += (xF + yF)/4
        density[x + 1, y] += (1 - xF + yF)/4
        density[x, y + 1] += (xF + 1 - yF)/4
        density[x + 1, y + 1] += (2 - xF - yF)/4

final = density.copy()/50
final = np.sqrt(density)

vmax = 5
plt.imshow(final, cmap='Greys_r', vmin=0, vmax=vmax)
plt.show()
plt.imsave('t7.png', final, cmap='Greys_r', vmin=0, vmax=vmax)
