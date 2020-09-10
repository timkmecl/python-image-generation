import cmath
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


''' Parameters '''
# Grid settings
gReal = [-10, 10]
gImag = [-10, 10]
gPerU = 2

# Resolution
res = 1080

pPerLine = 500
steps = 20

# Zoom/focus
f = 12
m = 0.002
zoom = 2.2

# Camera
pos = np.array((6, 7, 5))
direc = np.array((0, 0, 0))
top = np.array((0, 0.0001, -1))

# Function
function = np.tan




def normed(v):
    return v / np.linalg.norm(v)


def projection(v, pos, dirV, topU, leftU, f, m, l):
    point = v - pos

    d = np.linalg.norm(point, axis=1)[:,np.newaxis]
    pointU = point/d

    psi = np.arccos(pointU.dot(dirV))

    x = psi < math.pi/2
    d1 = d[x]
    pointU1 = pointU[x]

    v = np.stack((pointU1.dot(topU), pointU1.dot(leftU)), axis=0)

    x = (np.abs(v[0]) < 1) & (np.abs(v[1]) < 1)
    d1 = d1[x]
    v1 = v.T[x]

    r = (np.random.rand(*d1.shape) ** 2) * (m * (np.abs(f - d1)) ** math.e) / d1
    fi = np.random.rand(d1.shape[0]) * math.tau
    vRand = r * np.stack((np.cos(fi), np.sin(fi)), axis=1)
    l.extend(list(v1 + vRand))



dirV = normed(direc - pos)
topU = normed(top - top.dot(dirV) * dirV)
leftU = normed(np.cross(dirV, topU))
size = 1 / zoom

l = []

bf = ((steps * pPerLine) / 1000) / ((res) / 500)
density = np.zeros((res, res), dtype=np.float32)
print(bf)


def vis():
    global density
    count = 0
    print("vis")
    for z in l2:
        if np.linalg.norm(z) > size:
            continue

        n = res * (z + size) / (2*size)

        if n[0] >= res-1 or n[1] >= res-1:
            continue

        x = math.floor(n[0])
        y = math.floor(n[1])
        #density[x, y] += 1
        xF = float(n[0])%1.0
        yF = float(n[1])%1.0
        density[x, y] += (xF + yF)/4
        density[x + 1, y] += (1 - xF + yF)/4
        density[x, y + 1] += (xF + 1 - yF)/4
        density[x + 1, y + 1] += (2 - xF - yF)/4
        count += 1

    density2 = density.copy()
    density2 /= progress
    density2 /= (count / len(l2))
    density2 /= bf
    #density = np.log(density + 1)
    density2 = np.sqrt(density)

    print("fact", progress*bf*count/len(l2))
    ff = math.sqrt(progress*bf*count/len(l2))

    plt.clf()
    plt.imshow(density2, cmap='Greys_r', vmin=0.1/ff, vmax=max(1/ff, math.sqrt(6)))
    plt.title(f'{100*progress}%')
    plt.draw()
    plt.pause(0.05)


# plt.ion()
for i in tqdm(range(steps)):
    l = []
    for j in range(gImag[0]*gPerU, gImag[1]*gPerU + 1):
        re = np.random.rand(pPerLine) * (gReal[1] - gReal[0]) + gReal[0]
        im = np.full_like(re, j / gPerU)
        z = re + 1j * im
        zz = function(z)

        pts = np.stack((np.real(zz), np.imag(zz), im), axis=1)

        projection(pts, pos, dirV, topU, leftU, f, m, l)

    for j in range(gReal[0]*gPerU, gReal[1]*gPerU + 1):
        re = np.full_like(re, j / gPerU)
        im = np.random.rand(pPerLine) * (gImag[1] - gImag[0]) + gImag[0]
        z = re + 1j * im
        zz = function(z)

        pts = np.stack((np.real(zz), np.imag(zz), im), axis=1)

        projection(pts, pos, dirV, topU, leftU, f, m, l)
    
    l2 = np.array(l)
    '''
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.scatter(l[:, 0], l[:, 1], s=0.2, c='white', alpha = 0.2)
    plt.xlim(-1/zoom, 1/zoom)
    plt.ylim(-1/zoom, 1/zoom)
    ax.set_facecolor("black")
    plt.show()'''

    progress = (i+1) / steps

    vis()
plt.show()
