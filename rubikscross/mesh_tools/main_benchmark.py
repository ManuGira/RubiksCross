import time

import cv2
import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

import inverse_map
import inverse_map_jit
import inverse_map_barycentric
import pickle
import os

def get_distortion_map(N, force_generate=False):
    pickle_filname = f'distorion_map_{N}.pickle'
    if not force_generate and os.path.exists(pickle_filname):
        with open(pickle_filname, 'rb') as handle:
            xmap, ymap = pickle.load(handle)
            return xmap, ymap

    sh = (N, N)
    t = np.random.normal(size=sh)
    dx = ndi.gaussian_filter(t, 40, order=(0, 1))
    dy = ndi.gaussian_filter(t, 40, order=(1, 0))
    dx *= 20 / dx.max()
    dy *= 20 / dy.max()
    yy, xx = np.indices(sh)
    xmap = (xx - dx).astype(np.float32)
    ymap = (yy - dy).astype(np.float32)

    with open(pickle_filname, 'wb') as handle:
        pickle.dump((xmap, ymap), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return xmap, ymap


def generate_test_image(N):
    sh = (N, N)
    img = np.zeros(sh)
    img[::10, :] = 1
    img[:, ::10] = 1

    p0 = N//10
    size = N*8//10
    th = 10
    img[p0:p0+size, p0:p0+th] = 1
    img[p0:p0+th, p0:p0+size*2//3] = 1
    img[p0+size//2:p0+size//2+th, p0:p0+size//2] = 1

    img = ndi.gaussian_filter(img, 0.5)
    return img


def benchmark(invert_map_function):
    dts = []
    print("warmp up")
    t0 = time.time()
    xmap, ymap = get_distortion_map(4)
    res, _ = invert_map_function(xmap, ymap)
    dt = time.time() - t0
    dts.append(dt)
    print(f"Warmup time ({xmap.shape}, {xmap.dtype} -> {res.shape}, {res.dtype}): {dt:.3f} s")

    for N in range(100, 1001, 100):
        xmap, ymap = get_distortion_map(N)

        t0 = time.time()
        xmap1, ymap1 = invert_map_function(xmap, ymap)
        dt = time.time() - t0
        dts.append(dt)
        print(f"Computation time ({xmap.shape}, {xmap.dtype} -> {xmap1.shape}, {xmap1.dtype}): {dt:.3f} s")

    return dts


def demo(invert_map_function):
    N = 500
    xmap, ymap = get_distortion_map(N)
    img = generate_test_image(N)
    warped = cv2.remap(img, xmap, ymap, cv2.INTER_LINEAR)

    cv2.imshow("Warped", warped)
    cv2.waitKeyEx(1)

    xmap_inv, ymap_inv = invert_map_function(xmap, ymap)
    unwarped = cv2.remap(warped,  xmap_inv, ymap_inv, cv2.INTER_LINEAR)

    cv2.imshow("Unwarped", unwarped)
    cv2.waitKeyEx(0)


def main():
    dts = benchmark(inverse_map.invert_map)
    plt.plot(dts[1:])

    dts = benchmark(inverse_map_jit.invert_map)
    plt.plot(dts[1:])

    dts = benchmark(inverse_map_barycentric.invert_map)
    plt.plot(dts[1:])

    plt.show()

if __name__ == '__main__':
    # demo(inverse_map_barycentric.invert_map)
    main()
