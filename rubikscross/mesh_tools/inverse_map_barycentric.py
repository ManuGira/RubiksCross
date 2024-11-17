import numpy as np
import math

import numba

@numba.njit()
def vertex_index_buffer(h: int, w: int):
    N = (h - 1) * (w - 1) * 2
    n = 0
    triangle_vib = np.empty((N, 3), dtype=np.int32)
    for y in range(h - 1):
        for x in range(w - 1):
            i0 = y * w + x
            i1 = i0 + 1
            i2 = i0 + w
            i3 = i2 + 1
            triangle_vib[n, :] = i0, i1, i2
            triangle_vib[n + 1, :] = i2, i1, i3
            n += 2
    return triangle_vib

@numba.njit()
def invert_map(xmap, ymap):
    h, w = xmap.shape
    xmap_inv = np.zeros_like(xmap, dtype=np.float32) - 1
    ymap_inv = np.zeros_like(ymap, dtype=np.float32) - 1

    triangle_vib = vertex_index_buffer(h, w)

    for k0, k1, k2 in triangle_vib:
        x0 = xmap.ravel()[k0]
        x1 = xmap.ravel()[k1]
        x2 = xmap.ravel()[k2]

        y0 = ymap.ravel()[k0]
        y1 = ymap.ravel()[k1]
        y2 = ymap.ravel()[k2]

        # barycentric coordinates
        y12 = y1 - y2
        x02 = x0 - x2
        x21 = x2 - x1
        y02 = y0 - y2
        # TODO: be carefull when norm is 0
        norm = y12 * x02 + x21 * y02

        i0 = k0 // w
        i1 = k1 // w
        i2 = k2 // w

        j0 = k0 % w
        j1 = k1 % w
        j2 = k2 % w

        # find surrounding rectangle
        xmin = min(min(x0, x1), x2)
        ymin = min(min(y0, y1), y2)
        xmax = max(max(x0, x1), x2)
        ymax = max(max(y0, y1), y2)

        xmin = max(int(math.floor(xmin)), 0)
        ymin = max(int(math.floor(ymin)), 0)
        xmax = min(int(math.ceil(xmax)), w - 1)
        ymax = min(int(math.ceil(ymax)), h - 1)
        # xmax += 1
        # ymax += 1
        for px in range(xmin, xmax):
            pwx0 = y12 * (px - x2)
            pwx1 = -y02 * (px - x2)
            for py in range(ymin, ymax):
                if norm == 0:
                    xmap_inv[py, px] = j0
                    ymap_inv[py, px] = i0
                else:
                    # compute non-normalized weights of barycentric coordinates
                    w0 = (pwx0 + x21 * (py - y2))/norm
                    w1 = (pwx1 + x02 * (py - y2))/norm
                    w2 = 1 - w0 - w1
                    #
                    # w0 = 1
                    # w1 = 0
                    # w2 = 0
                    # norm = 1

                    if w0 < 0 or w1 < 0 or w2 < 0:
                        continue

                    xmap_inv[py, px] = (j0 * w0 + j1 * w1 + j2 * w2)
                    ymap_inv[py, px] = (i0 * w0 + i1 * w1 + i2 * w2)

    # invert maps
    # for i in range(dst_size):
    #     for j in range(dst_size):
    #         x, y = int(xmap[i, j]), int(ymap[i, j])
    #         # x, y = int(round(mx0[i, j])), int(round(my0[i, j]))
    #         # x = min(x+k, 47)
    #         xmap_inv[y, x] = j
    #         ymap_inv[y, x] = i

    return xmap_inv, ymap_inv
