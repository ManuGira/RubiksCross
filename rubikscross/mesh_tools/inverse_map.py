import numpy as np


# https://stackoverflow.com/a/65566295
def bilinear_inverse(p, vertices, numiter=4):
    """
    Compute the inverse of the bilinear map from the unit square
    [(0,0), (1,0), (1,1), (0,1)]
    to the quadrilateral vertices = [p0, p1, p2, p3]

    Parameters:
    ----------
    p: array of shape (2, ...).
    vertices: array of shape (4, 2, ...).
    numiter: Number of Newton iterations.

    Returns:
    --------
    s: array of shape (2, ...).
    """
    map_dtype = vertices.dtype
    p = np.asarray(p)
    v = np.asarray(vertices)
    sh = p.shape[1:]
    if v.ndim == 2:
        expanded_v = np.empty((4, 2) + sh, dtype=map_dtype)
        for i in range(4):
            for j in range(2):
                expanded_v[i, j, ...] = v[i, j]
        v = expanded_v

    # Start in the center
    s = np.empty((2,) + sh, dtype=map_dtype)
    s[0, ...] = 0.5  # s0
    s[1, ...] = 0.5  # s1
    s0, s1 = s[0], s[1]
    for k in range(numiter):
        # Residual
        r0 = (v[0, 0] * (1 - s0) * (1 - s1) + v[1, 0] * s0 * (1 - s1) + v[2, 0] * s0 * s1 + v[3, 0] * (1 - s0) * s1 - p[0])
        r1 = (v[0, 1] * (1 - s0) * (1 - s1) + v[1, 1] * s0 * (1 - s1) + v[2, 1] * s0 * s1 + v[3, 1] * (1 - s0) * s1 - p[1])

        # Jacobian
        J11 = (-v[0, 0] * (1 - s1) + v[1, 0] * (1 - s1) + v[2, 0] * s1 - v[3, 0] * s1)
        J21 = (-v[0, 1] * (1 - s1) + v[1, 1] * (1 - s1) + v[2, 1] * s1 - v[3, 1] * s1)
        J12 = (-v[0, 0] * (1 - s0) - v[1, 0] * s0 + v[2, 0] * s0 + v[3, 0] * (1 - s0))
        J22 = (-v[0, 1] * (1 - s0) - v[1, 1] * s0 + v[2, 1] * s0 + v[3, 1] * (1 - s0))

        detJ = J11 * J22 - J12 * J21
        inv_detJ = 1. / detJ

        s0 -= inv_detJ * (J22 * r0 - J12 * r1)
        s1 -= inv_detJ * (-J21 * r0 + J11 * r1)

    return s


def quad_loops(i0, j0, quads, x0, x0_offset, x1, xN, y0, y0_offset, y1, yN):
    map_type = quads.dtype

    # Shape of destination array
    sh_dest = (1 + y1.max() - y0_offset, 1 + x1.max() - x0_offset)

    xmap1 = np.zeros(sh_dest, dtype=map_type)
    ymap1 = np.zeros(sh_dest, dtype=map_type)
    TN = np.zeros(sh_dest, dtype=np.int32)
    # Smallish number to avoid missing point lying on edges
    epsilon = 0.01
    # Loop through indices possibly within quads
    for ix in range(xN.max()):
        for iy in range(yN.max()):
            # Work only with quads whose bounding box contain indices
            valid = (xN > ix) * (yN > iy)

            # Local points to check
            p = np.array([y0[valid] + ix, x0[valid] + iy])

            # Map the position of the point in the quad
            valid_quads = quads[:, :, valid]

            s = bilinear_inverse(p, valid_quads)

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
    return TN, xmap1, ymap1


def invert_map(xmap, ymap, diagnostics=False):
    """
    Generate the inverse of deformation map defined by (xmap, ymap) using inverse bilinear interpolation.
    """
    # import matplotlib.pyplot as plt

    # Generate quadrilaterals from mapped grid points.
    quads = np.array([
        [ymap[:-1, :-1], xmap[:-1, :-1]],
        [ymap[1:, :-1], xmap[1:, :-1]],
        [ymap[1:, 1:], xmap[1:, 1:]],
        [ymap[:-1, 1:], xmap[:-1, 1:]],
    ])

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

    TN, xmap1, ymap1 = quad_loops(i0, j0, quads, x0, x0_offset, x1, xN, y0, y0_offset, y1, yN)

    if diagnostics:
        diag = {'x_offset': x0_offset,
                'y_offset': y0_offset,
                'mask': TN > 0}
        return xmap1, ymap1, diag
    else:
        return xmap1, ymap1
