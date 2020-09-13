import numpy as np
import cmath
import math
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import os


''' Parameters '''

# Range on Re and Im axes
view = np.array([[-2, 1], [-1.5, 1.5]])

# Iterated function


def function(z, c):
    return z**2 + c


# Initital sequence element
z0 = -0.7+0.5j

# Number of trajectories and iterations
pts = 1_000_000
its = 5_000


''' Code '''

# Generates inital points for trajectories


def inital_points(pts, view):
    z1 = np.random.rand(pts, 2) * (view[:, 1] - view[:, 0]) + view[:, 0]
    z1 = z1[:, 0] + 1j*z1[:, 1]
    return z1


# Calculates trajectories for inital points
def calculate_trajectories(z0, z1, func, pts, its, remove_cardioid=False):
    c = np.copy(z1)
    if remove_cardioid:
        x = np.logical_not(
            (np.abs(c)**2) * (8 * np.abs(c)**2 - 3) <= 3/32 - c.real)
        c = c[x]

    z = np.full_like(c, z0)
    ind = np.arange(z.shape[0])

    n = 0
    l1 = []

    for i in tqdm(range(its)):
        z = function(z, c)
        n += 1
        l1.append([z.copy(), ind.copy(), n])

        x = np.abs(z) < 2
        c = c[x]
        z = z[x]
        ind = ind[x]

    return l1, ind


# Generates a list of coordinates from previous function's output
def change_data_shape(l1, ind):
    l = []
    ll = []
    for a in tqdm(l1[1:]):
        x = np.logical_not(np.isin(a[1], ind))
        b = a[0][x]
        #l2 = np.stack((a[0][x].real, a[0][x].imag, np.full(a[0][x].shape, a[2]))).T
        # l.extend(l2.tolist())
        l2 = [[b[i].real, b[i].imag, a[2]] for i in range(b.shape[0])]
        l.extend(l2)

    ll = [[e[0] for e in l], [e[1] for e in l], [e[2] for e in l]]

    return ll


# Generates a 2d histogram of points and point*current_iteration
def to_histogram(res, ll):
    H = np.histogram2d(ll[0], ll[1], bins=res, range=view)
    density = H[0]
    H2 = np.histogram2d(ll[0], ll[1], bins=res, range=view, weights=ll[2])
    density2 = H2[0]

    return density, density2


# Draws an image
def draw(density, density2, divide_by=1, show_2=True, vmax=1, show=True, save=False, filename="ttt.png"):
    if show_2:
        final = density2.copy()/divide_by
    else:
        final = density.copy()/divide_by

    final = np.sqrt(final)

    if show:
        plt.imshow(final, cmap='Greys_r', vmin=0, vmax=vmax)
        plt.show()
    if save:
        plt.imsave(os.path.join(os.path.dirname(__file__), 'img', filename), final, cmap='Greys_r', vmin=0, vmax=vmax)


def main():
    z1 = inital_points(pts, view)
    r = 30
    for i in range(r):
        print(i)
        phi = (math.pi * i/r) / 10
        z0 = (math.cos(phi) + 1j * math.sin(phi)) * 1
        l = change_data_shape(*calculate_trajectories(z0, z1, function, pts, its))
        hist = to_histogram(1080, l)
        draw(*hist, 35, show=False, show_2=False, save=True, filename=f'an2/a{i}.png')
        draw(*hist, 7000, show=False, save=True, filename=f'an1/it{i}.png')


if __name__ == '__main__':
    main()
