import numpy as np

# https://stackoverflow.com/a/65566295
def bilinear_inverse(p, vertices, numiter=4):
    """
    Compute the inverse of the bilinear map from the unit square
    [(0,0), (1,0), (1,1), (0,1)]
    to the quadrilateral vertices = [p0, p1, p2, p4]

    Parameters:
    ----------
    p: array of shape (2, ...)
        Points on which the inverse transforms are applied.
    vertices: array of shape (4, 2, ...)
        Coordinates of the vertices mapped to the unit square corners
    numiter:
        Number of Newton interations

    Returns:
    --------
    s: array of shape (2, ...)
        Mapped points.

    This is a (more general) python implementation of the matlab implementation
    suggested in https://stackoverflow.com/a/18332009/1560876
    """

    p = np.asarray(p)
    v = np.asarray(vertices)
    sh = p.shape[1:]
    if v.ndim == 2:
        v = np.expand_dims(v, axis=tuple(range(2, 2 + len(sh))))

    # Start in the center
    s = .5 * np.ones((2,) + sh)
    s0, s1 = s
    for k in range(numiter):
        # Residual
        r = v[0] * (1 - s0) * (1 - s1) + v[1] * s0 * (1 - s1) + v[2] * s0 * s1 + v[3] * (1 - s0) * s1 - p

        # Jacobian
        J11 = -v[0, 0] * (1 - s1) + v[1, 0] * (1 - s1) + v[2, 0] * s1 - v[3, 0] * s1
        J21 = -v[0, 1] * (1 - s1) + v[1, 1] * (1 - s1) + v[2, 1] * s1 - v[3, 1] * s1
        J12 = -v[0, 0] * (1 - s0) - v[1, 0] * s0 + v[2, 0] * s0 + v[3, 0] * (1 - s0)
        J22 = -v[0, 1] * (1 - s0) - v[1, 1] * s0 + v[2, 1] * s0 + v[3, 1] * (1 - s0)

        inv_detJ = 1. / (J11 * J22 - J12 * J21)

        s0 -= inv_detJ * (J22 * r[0] - J12 * r[1])
        s1 -= inv_detJ * (-J21 * r[0] + J11 * r[1])

    return s


def invert_map(xmap, ymap, diagnostics=False):
    """
    Generate the inverse of deformation map defined by (xmap, ymap) using inverse bilinear interpolation.
    """

    dtype = xmap.dtype

    # Generate quadrilaterals from mapped grid points.
    quads = np.array([[ymap[:-1, :-1], xmap[:-1, :-1]],
                      [ymap[1:, :-1], xmap[1:, :-1]],
                      [ymap[1:, 1:], xmap[1:, 1:]],
                      [ymap[:-1, 1:], xmap[:-1, 1:]]])

    # Range of indices possibly within each quadrilateral
    x0 = np.floor(quads[:, 1, ...].min(axis=0)).astype(int)
    x1 = np.ceil(quads[:, 1, ...].max(axis=0)).astype(int)
    y0 = np.floor(quads[:, 0, ...].min(axis=0)).astype(int)
    y1 = np.ceil(quads[:, 0, ...].max(axis=0)).astype(int)

    # Quad indices
    i0, j0 = np.indices(x0.shape)

    # Offset of destination map
    x0_offset = x0.min()
    y0_offset = y0.min()

    # Index range in x and y (per quad)
    xN = x1 - x0 + 1
    yN = y1 - y0 + 1

    # Shape of destination array
    sh_dest = (1 + x1.max() - x0_offset, 1 + y1.max() - y0_offset)

    # Coordinates of destination array
    yy_dest, xx_dest = np.indices(sh_dest)

    xmap1 = np.zeros(sh_dest, dtype=dtype)
    ymap1 = np.zeros(sh_dest, dtype=dtype)
    TN = np.zeros(sh_dest, dtype=int)

    # Smallish number to avoid missing point lying on edges
    epsilon = .01

    # Loop through indices possibly within quads
    for ix in range(xN.max()):
        for iy in range(yN.max()):
            # Work only with quads whose bounding box contain indices
            valid = (xN > ix) * (yN > iy)

            # Local points to check
            p = np.array([y0[valid] + ix, x0[valid] + iy])

            # Map the position of the point in the quad
            s = bilinear_inverse(p, quads[:, :, valid])

            # s out of unit square means p out of quad
            # Keep some epsilon around to avoid missing edges
            in_quad = np.all((s > -epsilon) * (s < (1 + epsilon)), axis=0)

            # Add found indices
            ii = p[0, in_quad] - y0_offset
            jj = p[1, in_quad] - x0_offset

            ymap1[ii, jj] += i0[valid][in_quad] + s[0][in_quad]
            xmap1[ii, jj] += j0[valid][in_quad] + s[1][in_quad]

            # Increment count
            TN[ii, jj] += 1

    ymap1 /= TN + (TN == 0)
    xmap1 /= TN + (TN == 0)

    if diagnostics:
        diag = {'x_offset': x0_offset,
                'y_offset': y0_offset,
                'mask': TN > 0}
        return xmap1, ymap1, diag
    else:
        return xmap1, ymap1


def main():
    import cv2
    from scipy import ndimage as ndi
    import time

    print("Generate deformation field map")
    N = 500
    sh = (N, N)
    t = np.random.normal(size=sh)
    dx = ndi.gaussian_filter(t, 40, order=(0, 1))
    dy = ndi.gaussian_filter(t, 40, order=(1, 0))
    dx *= 30 / dx.max()
    dy *= 30 / dy.max()
    yy, xx = np.indices(sh)
    xmap = (xx - dx).astype(np.float32)
    ymap = (yy - dy).astype(np.float32)

    print("Generate Test image")
    img = np.zeros(sh)
    img[::10, :] = 1
    img[:, ::10] = 1
    img = ndi.gaussian_filter(img, 0.5)

    print("Apply forward mapping")
    warped = cv2.remap(img, xmap, ymap, cv2.INTER_LINEAR)
    cv2.imshow("warped", warped)
    cv2.waitKeyEx(1)

    print("warmp up")
    t0 = time.time()
    xmap_warmup = xmap[::N // 3, ::N // 3]
    ymap_warmup = ymap[::N // 3, ::N // 3]
    invert_map(xmap_warmup, ymap_warmup)
    print(f"Warmup time ({xmap_warmup.shape}, {xmap_warmup.dtype}): {time.time() - t0:.3f} s")

    print("Invert mapping...")
    t0 = time.time()
    xmap1, ymap1 = invert_map(xmap, ymap)
    print(f"Computation time: {time.time() - t0:.3f} s")
    unwarped = cv2.remap(warped, xmap1, ymap1, cv2.INTER_LINEAR)
    cv2.imshow("unwarped", unwarped)
    cv2.waitKeyEx(0)


if __name__ == '__main__':
    main()
